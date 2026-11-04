// A Gen3 Node as gen3rt drives it: device memory at the Node's (physical) byte addresses, and a command stream.
// EmuDevice runs the bit-exact emulator (gen3_sw/emu); BoardDevice (board.h) the PYNQ-ZU overlay.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "g3emu.h"

namespace g3rt {

using g3::Cmd128;

class Device {
public:
    virtual ~Device() = default;
    // device memory: Node byte addresses; allocations are 4 KiB aligned and zero filled
    virtual uint64_t alloc(size_t bytes) = 0;
    virtual void write(uint64_t addr, const void* src, size_t bytes) = 0;   // addr 32-byte aligned
    virtual void read(uint64_t addr, void* dst, size_t bytes) = 0;
    // Device memory the host uses in place (a device with a host mapping; else nullptr, and read / write it is).
    // read_in_place: [addr, addr + bytes), which the Node wrote, made visible to host reads, and its host address.
    // write_in_place: the host address of [addr, addr + bytes); publish makes the host's writes there visible to the
    // Node. Ranges 64-byte aligned (cache lines); host threads may use them.
    virtual const void* read_in_place(uint64_t addr, size_t bytes) { (void)addr; (void)bytes; return nullptr; }
    virtual void* write_in_place(uint64_t addr, size_t bytes) { (void)addr; (void)bytes; return nullptr; }
    virtual void publish(uint64_t addr, size_t bytes) { (void)addr; (void)bytes; }
    // commands run in order; completions of commands with a tag (and of failed commands) are collected
    virtual void submit(const std::vector<Cmd128>& cmds) = 0;
    virtual void sync() = 0;                                // every submitted command completed
    virtual uint32_t cycles() { return 0; }                 // the Node's cycle counter (the board; else 0)
    virtual uint32_t port_words(int) { return 0; }          // words read on read port p (the board; else 0)
    std::vector<g3::Completion> take_completions() { auto c = std::move(cpl_); cpl_.clear(); return c; }
    virtual std::string name() const = 0;
    // until the completion of the command with `tag` (1..254, reported) has come in, the Node running on; it
    // stays for take_completions. The emulator has run every command when submit returns.
    virtual void wait(uint8_t tag) { (void)tag; sync(); }
protected:
    std::vector<g3::Completion> cpl_;
};

class EmuDevice : public Device {
public:
    explicit EmuDevice(g3::NodeOptions opt = {});
    uint64_t alloc(size_t bytes) override;
    void write(uint64_t addr, const void* src, size_t bytes) override;
    void read(uint64_t addr, void* dst, size_t bytes) override;
    void submit(const std::vector<Cmd128>& cmds) override;
    void sync() override {}
    std::string name() const override { return "emulator"; }
    g3::Node& node() { return node_; }
private:
    g3::Node node_;
    uint64_t next_ = 0x1'0000'0000ULL;
};

}  // namespace g3rt
