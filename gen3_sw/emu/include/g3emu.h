// Gen3 Node v3 command-level emulator: the 128-bit command ABI (docs/gen3/node_v3.md s3, gen3_rtl/include/g3_defs.vh)
// executed in program order over a unified buffer (UB) and a byte-addressed external memory, bit-exact with the
// RTL (gen3_fpga/node) in what it writes and in the completion statuses. Cycles are not modelled.
//
// Program-order semantics: each command runs to completion before the next one is looked at; host pushes and
// MOVE_IN write their data when they come in the program. The RTL's watermarks and hazard rules make a valid program
// equivalent to this. A WAIT completes as soon as its condition holds (possibly after later commands).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "g3emu_kernels.h"

namespace g3 {

struct Word {
    uint8_t b[32] = {0};   // byte 0 = bits 7:0 of the 256-bit word
    uint32_t u32(int i) const { return uint32_t(b[4 * i]) | uint32_t(b[4 * i + 1]) << 8 | uint32_t(b[4 * i + 2]) << 16 |
                                       uint32_t(b[4 * i + 3]) << 24; }
    void set_u32(int i, uint32_t v) { for (int k = 0; k < 4; ++k) b[4 * i + k] = uint8_t(v >> (8 * k)); }
};

struct Cmd128 {
    uint64_t lo = 0, hi = 0;   // bits 63:0, 127:64
    uint64_t field(int lsb, int width) const;
    bool bit(int i) const { return field(i, 1) != 0; }
};

enum Status : uint8_t {
    ST_OK = 0, ST_UNSUPPORTED = 1, ST_RESERVED = 2, ST_RANGE = 3, ST_BINDING = 4, ST_ORDER = 5, ST_CAPACITY = 6,
    ST_POISONED = 7, ST_ABORTED = 8, ST_EXECUTION = 9
};
const char* status_name(int status);

enum Opcode : uint8_t {
    OP_BIND = 0x01, OP_UNBIND = 0x02, OP_MOVE_OUT = 0x08, OP_MOVE_IN = 0x09, OP_GEMM = 0x10, OP_ATTN = 0x11,
    OP_VECTOR = 0x12, OP_FENCE = 0x20, OP_QUIESCE = 0x21, OP_RESUME = 0x22, OP_WAIT = 0x23, OP_CTX_DROP = 0x24,
    OP_ABORT = 0x25, OP_CLEAR = 0x26, OP_CSR = 0x30
};
enum Kind : uint8_t { KIND_STATIC = 0, KIND_APPEND = 1, KIND_STREAM = 2 };
enum WriterKind : uint8_t { W_NONE = 0, W_EXT = 1, W_PARENT = 2 };
enum Epilogue : uint8_t { EPI_NONE = 0, EPI_ADD = 1, EPI_MUL = 2, EPI_SILU = 3, EPI_GELU = 4, EPI_SILU_MUL = 5,
                          EPI_ROPE = 6 };

struct Completion {
    uint8_t tag = 0;
    uint8_t status = 0;
    uint32_t data = 0;
    bool pending = false;   // execute(): a WAIT whose condition does not hold yet (it completes later)
    uint32_t cycle = 0;     // the board: the Node's cycle counter when it completed (the emulator: 0)
};

// One binding table (URF) entry, as g3_ctrl keeps it. Layout {tr, order}: 00 P8, 01 P32, 10 F32, 11 V^T.
struct Binding {
    bool valid = false;
    uint8_t gen = 0;
    uint16_t base = 0;     // UB word
    uint16_t ext = 0;      // words
    bool order = false;    // BIND bit 56
    bool tr = false;       // BIND bit 127
    uint8_t kind = KIND_STATIC;
    uint8_t alen = 0;      // array length (element 0), 1 for the other elements
    uint16_t wm = 0;       // watermark: words (STATIC) or rows (APPEND)
    uint8_t writer = W_NONE;
    bool poison = false;
};

// Sparse byte-addressed external memory in 32-byte words (absent words read as zero).
class ExtMemory {
public:
    static constexpr int PAGE_BITS = 10;
    const Word* find(uint64_t word_addr) const;
    Word read(uint64_t word_addr) const;
    void write(uint64_t word_addr, const Word& w);
    void write_words(uint64_t word_addr, const Word* w, size_t n);
    void read_words(uint64_t word_addr, Word* out, size_t n) const;
    std::vector<uint64_t> addresses() const;   // written / preloaded words, ascending
    size_t size() const;
private:
    struct Page {
        Word w[1 << PAGE_BITS];
        uint64_t present[(1 << PAGE_BITS) / 64] = {0};
    };
    std::unordered_map<uint64_t, std::unique_ptr<Page>> pages_;
};

struct NodeOptions {
    // Attention inputs as the Python oracle sees them (decoded, then re-quantised; V^T keys >= keys zeroed). The RTL
    // feeds the stored codes and scales (default).
    bool attn_requantize = false;
    // diagnostics: which operands attn_requantize re-quantises (bit 0 Q, bit 1 K, bit 2 V^T)
    int attn_requantize_mask = 7;
    // print every command with its completion status to stderr
    bool trace = false;
    // VECTOR as the v3 Node's g3_vru_stub.sv (default): admitted, result released (watermark 0), status UNSUPPORTED.
    // false: gen3_rtl's g3_vru (RMSNORM).
    bool vector_stub = true;
    // worker threads for large GEMMs (output columns split; every value is computed alone, so results do not depend
    // on it)
    int threads = 1;
};

class Node {
public:
    static constexpr int UB_WORDS = 16384;                       // v3: 4 URAM banks x 4096 words
    static constexpr uint64_t NODE_BASE = 0x0000'1000'0000ULL;   // binding window base of host pushes

    explicit Node(NodeOptions opt = {});

    // memories
    const Word& ub(uint32_t addr) const { return ub_[addr & (UB_WORDS - 1)]; }
    void ub_write(uint32_t addr, const Word& w, uint32_t strobe = 0xFFFFFFFFu);
    ExtMemory& ext() { return ext_; }
    const ExtMemory& ext() const { return ext_; }

    // host operations of a program
    void preload(uint64_t byte_addr, const Word* words, size_t n);
    // write into a binding window (NODE_BASE + gen << 27 + idx << 19 + word << 5) in bursts of 16 words
    void push(uint64_t window_addr, const Word* words, size_t n);
    void push_binding(uint8_t idx, uint8_t gen, uint32_t word, const Word* words, size_t n) {
        push(NODE_BASE + (uint64_t(gen) << 27) + (uint64_t(idx) << 19) + (uint64_t(word) << 5), words, n);
    }
    // admit and execute one command; returns its completion (also appended to completions(); a WAIT that is still
    // pending is returned with pending = true and appended when it completes)
    Completion execute(const Cmd128& cmd);

    const std::vector<Completion>& completions() const { return completions_; }
    const Binding& binding(int idx) const { return urf_[idx & 255]; }
    // STREAM bindings 0xF8..0xFF: source word address (byte address / 32) and length in words
    uint64_t stream_source(int idx) const { return sd_src_[idx & 7]; }
    uint32_t stream_length(int idx) const { return sd_len_[idx & 7]; }
    uint32_t stream_quarters(int idx) const { return sd_qs_[idx & 7]; }
    size_t pending_waits() const { return waiters_.size(); }
    std::vector<std::string>& warnings() { return warnings_; }

private:
    struct Waiter {
        uint8_t tag, idx, gen;
        bool mode;
    };
    NodeOptions opt_;
    std::vector<Word> ub_;
    ExtMemory ext_;
    std::array<Binding, 256> urf_{};
    std::array<uint32_t, 2048> owner_{};     // granule (8 words) -> {valid, idx, gen}
    std::array<uint64_t, 8> sd_src_{};       // STREAM bindings 0xF8..0xFF: source byte address, length (words)
    std::array<uint32_t, 8> sd_len_{};
    std::array<uint32_t, 8> sd_qs_{};        // GEMM B in quarters: quarter stride in 16 KiB units (0: bands in order)
    uint32_t rms_mem_[32] = {0};             // fold 1/rms memory of the output engine (persistent, as in the RTL)
    std::vector<Waiter> waiters_;
    std::vector<Completion> completions_;
    std::vector<std::string> warnings_;

    void complete(uint8_t tag, uint8_t status, uint32_t data = 0) { completions_.push_back({tag, status, data}); }
    void event(uint8_t idx, uint8_t gen, uint32_t wm, bool final_);
    void check_waiters();
    void warn(const std::string& s);

    uint8_t exec_bind(const Cmd128& c);
    uint8_t exec_unbind(const Cmd128& c);
    uint8_t exec_move_out(const Cmd128& c);
    uint8_t exec_move_in(const Cmd128& c);
    uint8_t check_gemm(const Cmd128& c) const;
    void run_gemm(const Cmd128& c);
    uint8_t check_attn(const Cmd128& c) const;
    void run_attn(const Cmd128& c);
    uint8_t check_vector(const Cmd128& c) const;
    void run_vector(const Cmd128& c);
    bool source_pending(uint8_t idx) const;
};

// ---------------------------------------------------------------------------------------------------------------
// Program files (tb/gen2/node_program.py, tb/gen3/node/g3_node_tb.sv): program.hex records
//   01 CMD  02 PUSH {addr[67:20], n[19:0]} + n DATA  03 WAIT tag  04 MARK  05 END  06 PRELOAD (as PUSH)  ff DATA
// ---------------------------------------------------------------------------------------------------------------
struct ProgramRecord {
    uint8_t kind = 0;
    Word payload;
};
bool read_program(const std::string& path, std::vector<ProgramRecord>& recs, std::string* err);
// Run a program: preloads first (before reset, as the bench), then the records in order up to END.
void run_program(Node& node, const std::vector<ProgramRecord>& recs);
// hmem_out.hex: "<word address hex> <64 hex digits>" per external memory word, ascending
bool write_hmem(const Node& node, const std::string& path);
// completions.txt: "<sequence> <tag> <status> <data>" per completion, in completion order
bool write_completions(const Node& node, const std::string& path);

}  // namespace g3
