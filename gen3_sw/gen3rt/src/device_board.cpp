// BoardDevice: see gen3rt/board.h. Built on the board only (make BOARD=1).
#include "gen3rt/board.h"
#include "gen3rt/cmd.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <atomic>
#include <stdexcept>
#include <string>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

namespace g3rt {

namespace {
constexpr uint64_t REG_BASE = 0xA0000000ULL, REG_SPAN = 1u << 20;
constexpr uint32_t R_CTRL = 0x00, R_RING_LO = 0x01, R_RING_HI = 0x02, R_CMD_WR = 0x03, R_CPL_RD = 0x04,
                   R_RING_LG = 0x06, R_TAKEN = 0x08, R_CPL_WR = 0x09, R_CYCLES = 0x0B, R_STATE = 0x0C, R_ID = 0x0D,
                   R_PORT = 0x10, CPL_RING = 0x2000;
// diagnostics (gen3_fpga/node_board_v3/rtl/node_host.sv): parent tags, {last taken / next command op and tag},
// unit states, {node_mem, fetcher, NI} states
constexpr uint32_t R_DBG[] = {0x07, 0x0A, 0x0E, 0x0F};
// MMIO publication/consumption orders normal cacheable BO accesses with the device.
void io_barrier() {
#if defined(__aarch64__)
    asm volatile("dmb osh" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

double now_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) + 1e-9 * double(t.tv_nsec);
}
}  // namespace

struct BoardDevice::Impl {
    int memfd = -1;
    volatile uint32_t* regs = nullptr;
    xrt::device dev;
    xrt::bo bo;
    uint8_t* host = nullptr;
    uint64_t phys = 0;
    size_t size = 0, next = 0;
    uint64_t ring = 0;                 // command ring (Node address), entries 16 B
    uint32_t ring_n = 0;
    uint32_t cmd_wr = 0, cpl_rd = 0;
    FILE* trace = nullptr;
    unsigned spin_us = 0;
    void snapshot(const char* reason) {
        if (!trace) return;
        std::fflush(trace);
        std::fprintf(stderr,"GEN3_SNAPSHOT %s written=%u read=%u",reason,cmd_wr,cpl_rd);
        for (uint32_t r : {R_TAKEN,R_CPL_WR,R_STATE,R_CYCLES,0x10u,0x11u,0x12u,0x13u})
            std::fprintf(stderr," r%02x=%u",r,reg(r));
        std::fprintf(stderr," %s\n",diagnostics().c_str());
    }
    std::string diagnostics() const {
        std::string out = "dbg";
        char b[16];
        for (uint32_t r : R_DBG) { std::snprintf(b, sizeof b, " %08x", reg(r)); out += b; }
        return out;
    }
    uint32_t reg(uint32_t w) const { return regs[w * 4]; }
    void set(uint32_t w, uint32_t v) { regs[w * 4] = v; }
};

BoardDevice::BoardDevice(size_t pool, int ring_lg) : d_(new Impl) {
    if (const char* value=std::getenv("G3RT_SYNC_SPIN_US")) {
        char* end=nullptr;errno=0;const unsigned long n=std::strtoul(value,&end,10);
        if(errno || end==value || *end || n>1000)throw std::invalid_argument("G3RT_SYNC_SPIN_US: integer 0..1000");
        d_->spin_us=unsigned(n);
    }
    if (const char* value = std::getenv("G3RT_BOARD_POOL_MIB")) {
        char* end=nullptr; errno=0;
        const unsigned long mib=std::strtoul(value,&end,10);
        if (errno || end==value || *end || mib<32 || mib>4096)
            throw std::invalid_argument("G3RT_BOARD_POOL_MIB must be an integer in 32..4096");
        pool=size_t(mib)<<20;
    }
    d_->memfd = ::open("/dev/mem", O_RDWR | O_SYNC);
    if (d_->memfd < 0) throw std::runtime_error("open /dev/mem (needs root)");
    void* p = ::mmap(nullptr, REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, d_->memfd, off_t(REG_BASE));
    if (p == MAP_FAILED) throw std::runtime_error("mmap of the Node registers");
    d_->regs = static_cast<volatile uint32_t*>(p);
    if (d_->reg(R_ID) != 0x47334E03u)
        throw std::runtime_error("no v3 Node at 0xA000_0000 (id " + std::to_string(d_->reg(R_ID)) + ")");
    d_->dev = xrt::device(0);
    d_->bo = xrt::bo(d_->dev, pool, xrt::bo::flags::cacheable, 0);
    d_->host = d_->bo.map<uint8_t*>();
    d_->phys = d_->bo.address();
    d_->size = pool;
    std::memset(d_->host, 0, pool);
    d_->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, pool, 0);
    // the command ring at the start of the pool
    d_->ring_n = 1u << ring_lg;
    d_->ring = d_->phys;
    d_->next = (size_t(d_->ring_n) * 16 + 4095) / 4096 * 4096;
    // reset the Node, set the ring, release
    d_->set(R_CTRL, 1);
    usleep(1000);
    d_->set(R_RING_LO, uint32_t(d_->ring));
    d_->set(R_RING_HI, uint32_t(d_->ring >> 32));
    d_->set(R_RING_LG, uint32_t(ring_lg));
    d_->set(R_CMD_WR, 0);
    d_->set(R_CPL_RD, 0);
    usleep(1000);
    d_->set(R_CTRL, 0);
    for (int i = 0; i < 1000 && (d_->reg(R_STATE) & 1); ++i) usleep(100);
    if (d_->reg(R_STATE) & 1) throw std::runtime_error("the Node stays in reset (core clock locked?)");
    usleep(1000);
    if (const char* path=std::getenv("G3RT_BOARD_COMMAND_TRACE")) {
        d_->trace=std::fopen(path,"wb");
        if (!d_->trace) throw std::runtime_error("cannot open command trace");
        d_->snapshot("start");
    }
    // R51's UB RAM survives the Node reset. Switching model geometries exposed
    // NaN results from retained UB contents; establish a clean initial state.
    // The BO is already zero-filled. Borrow unused pool space until this transfer
    // retires, then leave it available to subsequent device allocations.
    constexpr size_t ub_bytes = 512u << 10;
    if (d_->size - d_->next < ub_bytes) throw std::runtime_error("pool too small for UB initialization");
    const uint16_t ub = binding(0xFE, 1);
    submit({cmd_bind(0, {0xFE, 1, 0, 16384, Order::P8, Kind::Static, true}),
            cmd_move_in(0, ub, 0, 16384, d_->phys + d_->next, true),
            cmd_fence(0), cmd_unbind(0, ub)});
    sync();
    for (const auto& c : take_completions())
        if (c.status != g3::ST_OK) throw std::runtime_error("UB initialization command failed");
}

BoardDevice::~BoardDevice() {
    if (d_->trace) { d_->snapshot("end"); std::fclose(d_->trace); }
    if (d_->regs) ::munmap(const_cast<uint32_t*>(d_->regs), REG_SPAN);
    if (d_->memfd >= 0) ::close(d_->memfd);
    delete d_;
}

uint64_t BoardDevice::alloc(size_t bytes) {
    const size_t a = d_->next;
    d_->next += (bytes + 4095) / 4096 * 4096;
    if (d_->next > d_->size) throw std::runtime_error("device memory pool exhausted");
    return d_->phys + a;
}

void BoardDevice::write(uint64_t addr, const void* src, size_t bytes) {
    const size_t off = size_t(addr - d_->phys);
    std::memcpy(d_->host + off, src, bytes);
    d_->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, off);
}

void BoardDevice::read(uint64_t addr, void* dst, size_t bytes) {
    const size_t off = size_t(addr - d_->phys);
    d_->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, off);
    std::memcpy(dst, d_->host + off, bytes);
}

const void* BoardDevice::read_in_place(uint64_t addr, size_t bytes) {
    const size_t off = size_t(addr - d_->phys);
    d_->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, off);
    return d_->host + off;
}

void* BoardDevice::write_in_place(uint64_t addr, size_t bytes) {
    const size_t off = size_t(addr - d_->phys);
    if (off + bytes > d_->size) throw std::out_of_range("BoardDevice::write_in_place");
    return d_->host + off;
}

void BoardDevice::publish(uint64_t addr, size_t bytes) {
    d_->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, size_t(addr - d_->phys));
}

void BoardDevice::poll_completions() {
    const uint32_t wr = d_->reg(R_CPL_WR) & 0xFFFF;
    io_barrier();
    const uint32_t before = d_->cpl_rd;
    while ((d_->cpl_rd & 0xFFFF) != wr) {
        const uint32_t w = CPL_RING + (d_->cpl_rd & 255);
        const uint32_t data = d_->regs[w * 4], st = d_->regs[w * 4 + 1];
        g3::Completion c;
        c.data = data;
        c.status = uint8_t(st & 0xF);
        c.tag = uint8_t((st >> 4) & 0xFF);
        c.cycle = d_->regs[w * 4 + 3];
        cpl_.push_back(c);
        if (d_->trace) std::fprintf(stderr,"GEN3_CPL count=%u tag=%u status=%u data=%u cycle=%u\n",d_->cpl_rd,unsigned(c.tag),unsigned(c.status),c.data,c.cycle);
        ++d_->cpl_rd;
    }
    if (d_->cpl_rd != before) { io_barrier(); d_->set(R_CPL_RD, d_->cpl_rd & 0xFFFF); }
}

void BoardDevice::submit(const std::vector<Cmd128>& cmds) {
    if (d_->trace) { std::fwrite(cmds.data(),sizeof(Cmd128),cmds.size(),d_->trace); std::fflush(d_->trace); }
    size_t i = 0;
    const double start = now_s();
    unsigned polls = 0;
    while (i < cmds.size()) {
        // room: entries the Node has not taken
        uint32_t taken = d_->reg(R_TAKEN) & 0xFFFF;
        uint32_t room = d_->ring_n - 1 - ((d_->cmd_wr - taken) & 0xFFFF);
        while (room == 0) {
            if ((++polls & 1023) == 0 && now_s() - start > 15.0)
                throw std::runtime_error("command ring timeout: state=" + std::to_string(d_->reg(R_STATE)) +
                                         " taken=" + std::to_string(taken) + " written=" + std::to_string(d_->cmd_wr));
            poll_completions();
            taken = d_->reg(R_TAKEN) & 0xFFFF;
            room = d_->ring_n - 1 - ((d_->cmd_wr - taken) & 0xFFFF);
        }
        const size_t first = d_->cmd_wr & (d_->ring_n - 1);
        size_t n = std::min<size_t>({room, cmds.size() - i, d_->ring_n - first});
        uint8_t* dst = d_->host + (d_->ring - d_->phys) + first * 16;
        static_assert(sizeof(Cmd128) == 16, "command ring ABI");
        std::memcpy(dst, cmds.data() + i, n * sizeof(Cmd128));
        // whole 32-byte words around the entries
        const size_t b0 = (first * 16) & ~size_t(31), b1 = ((first + n) * 16 + 31) & ~size_t(31);
        d_->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, b1 - b0, size_t(d_->ring - d_->phys) + b0);
        io_barrier();
        d_->cmd_wr += uint32_t(n);
        d_->set(R_CMD_WR, d_->cmd_wr & 0xFFFF);
        i += n;
    }
}

void BoardDevice::sync() {
    // a reporting FENCE (tag 255, kept for this) completes after every command before it
    const uint8_t tag = 255;
    Cmd128 f;
    f.lo = uint64_t(g3::OP_FENCE) | uint64_t(tag) << 8;
    submit({f});
    await(tag, true);
}

void BoardDevice::wait(uint8_t tag) { await(tag, false); }

// polls the completions until one with `tag` is in (erase: dropped, as sync's own FENCE)
void BoardDevice::await(uint8_t tag, bool erase) {
    const double start = now_s();
    size_t scanned = 0;
    unsigned polls = 0;
    for (;;) {
        poll_completions();
        for (size_t k = scanned; k < cpl_.size(); ++k) {
            if (cpl_[k].status != g3::ST_OK)
                throw std::runtime_error("Node completion error: tag=" + std::to_string(cpl_[k].tag) +
                                         " status=" + std::string(g3::status_name(cpl_[k].status)));
            if (cpl_[k].tag == tag) { if (erase) cpl_.erase(cpl_.begin() + long(k)); return; }
        }
        scanned = cpl_.size();
        if ((++polls & 1023) == 0 && now_s() - start > 15.0) {
            d_->snapshot("sync_timeout");
            throw std::runtime_error("Node sync timeout: state=" + std::to_string(d_->reg(R_STATE)) +
                                     " taken=" + std::to_string(d_->reg(R_TAKEN)) +
                                     " written=" + std::to_string(d_->cmd_wr) +
                                     " completion=" + std::to_string(d_->reg(R_CPL_WR)) +
                                     " read=" + std::to_string(d_->cpl_rd) +
                                     " read_target=" + std::to_string(d_->reg(R_CPL_RD)) +
                                     " write_target=" + std::to_string(d_->reg(R_CMD_WR)) + " " +
                                     d_->diagnostics());
        }
        // Brief polling can avoid the scheduler overshoot of a 5 us sleep.
        // Long operations retain the original sleeping wait and timeout.
        if(!d_->spin_us || (now_s()-start)*1e6>=d_->spin_us)usleep(5);
#if defined(__aarch64__)
        else asm volatile("yield" ::: "memory");
#endif
    }
}

uint32_t BoardDevice::cycles() { return d_->reg(R_CYCLES); }
uint32_t BoardDevice::port_words(int p) { return d_->reg(R_PORT + uint32_t(p)); }

double BoardDevice::core_mhz() {
    const uint32_t c0 = cycles();
    const double t0 = now_s();
    usleep(20000);
    const uint32_t c1 = cycles();
    const double t1 = now_s();
    return double(c1 - c0) / (t1 - t0) / 1e6;
}

}  // namespace g3rt
