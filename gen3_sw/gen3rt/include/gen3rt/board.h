// The v3 Node on PYNQ-ZU (gen3_fpga/node_board_v3): registers and the completion ring through the AXI BRAM
// controller window at 0xA000_0000 (/dev/mem; node_host.sv's word map), device memory in one cacheable XRT buffer
// (physical addresses are Node addresses), commands in a ring in that buffer (doorbell: the command write count).
// The bitstream is loaded and pl_clk0 set to 100 MHz before (board/load_overlay.py). Linux with XRT only.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gen3rt/device.h"

namespace g3rt {

class BoardDevice : public Device {
public:
    // pool: bytes of device memory (one XRT buffer); ring_lg: log2 of the command ring's entries
    explicit BoardDevice(size_t pool = 384u << 20, int ring_lg = 14);
    ~BoardDevice() override;
    uint64_t alloc(size_t bytes) override;
    void write(uint64_t addr, const void* src, size_t bytes) override;
    void read(uint64_t addr, void* dst, size_t bytes) override;
    const void* read_in_place(uint64_t addr, size_t bytes) override;
    void* write_in_place(uint64_t addr, size_t bytes) override;
    void publish(uint64_t addr, size_t bytes) override;
    void submit(const std::vector<Cmd128>& cmds) override;
    void sync() override;
    std::string name() const override { return "PYNQ-ZU"; }
    void wait(uint8_t tag) override;
    // core cycles (the Node's counter), read words per port
    uint32_t cycles() override;
    uint32_t port_words(int p) override;
    double core_mhz();               // measured against CLOCK_MONOTONIC
private:
    struct Impl;
    Impl* d_;
    void poll_completions();
    void await(uint8_t tag, bool erase);
};

}  // namespace g3rt
