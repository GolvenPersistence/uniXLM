#include <cstring>
#include <stdexcept>

#include "gen3rt/device.h"

namespace g3rt {

EmuDevice::EmuDevice(g3::NodeOptions opt) : node_(opt) {}

uint64_t EmuDevice::alloc(size_t bytes) {
    const uint64_t a = next_;
    next_ += (uint64_t(bytes) + 4095) / 4096 * 4096;
    return a;
}

void EmuDevice::write(uint64_t addr, const void* src, size_t bytes) {
    if (addr & 31) throw std::invalid_argument("EmuDevice::write: address not 32-byte aligned");
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint64_t w = addr >> 5;
    size_t full = bytes / 32;
    std::vector<g3::Word> buf(full);
    if (full) std::memcpy(buf.data(), s, full * 32);
    node_.ext().write_words(w, buf.data(), full);
    if (bytes % 32) {
        g3::Word t = node_.ext().read(w + full);
        std::memcpy(t.b, s + full * 32, bytes % 32);
        node_.ext().write(w + full, t);
    }
}

void EmuDevice::read(uint64_t addr, void* dst, size_t bytes) {
    if (addr & 31) throw std::invalid_argument("EmuDevice::read: address not 32-byte aligned");
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint64_t w = addr >> 5;
    const size_t n = (bytes + 31) / 32;
    std::vector<g3::Word> buf(n);
    node_.ext().read_words(w, buf.data(), n);
    std::memcpy(d, buf.data(), bytes);
}

void EmuDevice::submit(const std::vector<Cmd128>& cmds) {
    for (const Cmd128& c : cmds) {
        const g3::Completion r = node_.execute(c);
        if (!r.pending && (r.tag != 0 || r.status != g3::ST_OK)) cpl_.push_back(r);
    }
}

}  // namespace g3rt
