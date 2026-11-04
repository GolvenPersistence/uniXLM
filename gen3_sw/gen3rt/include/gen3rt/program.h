// Building Node command streams: the binding table's generations (they persist on the Node from one submit to the
// next), command tags, and the BIND / GEMM / ATTN / MOVE helpers.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "gen3rt/cmd.h"

namespace g3rt {

struct Bound {
    uint8_t idx = 0, gen = 0;
    uint32_t base = 0, extent = 0;
    int count = 1;
    uint16_t b(int element = 0) const { return binding(uint8_t(idx + element), gen); }
};

class Program {
public:
    std::vector<Cmd128> cmds;
    std::vector<const char*> roles;     // per command: the role given by role() when it was added (profiles)
    bool report_all = false;            // every command reports its completion (profiles)
    void role(const char* r) { role_ = r; }
    void clear() { cmds.clear(); roles.clear(); }

    uint8_t gen_of(uint8_t idx) const { return gens_[idx]; }
    uint8_t next_gen(uint8_t idx) const { return uint8_t(gens_[idx] % 255 + 1); }
    // a tag that reports its completion (1..254; 255 is the devices' own), or 0 (silent)
    uint8_t tag(bool report) {
        if (!report && !report_all) return 0;
        const uint8_t t = next_tag_;
        next_tag_ = uint8_t(next_tag_ % 254 + 1);
        return t;
    }
    // BIND: a.gen == 0 takes the index's next generation; every element of an array gets it
    Bound bind(BindArgs a, bool report = false) {
        if (a.gen == 0) a.gen = next_gen(a.idx);
        for (int e = 0; e < a.count; ++e) gens_[uint8_t(a.idx + e)] = a.gen;
        push(cmd_bind(tag(report), a));
        return Bound{a.idx, a.gen, a.base, a.extent, a.count};
    }
    uint8_t gemm(const GemmArgs& g, bool report = false) { const uint8_t t = tag(report); push(cmd_gemm(t, g)); return t; }
    uint8_t attn(const AttnArgs& a, bool report = false) { const uint8_t t = tag(report); push(cmd_attn(t, a)); return t; }
    uint8_t move_out(uint16_t src, uint32_t off, uint32_t words, uint64_t addr, bool report = false) {
        const uint8_t t = tag(report);
        push(cmd_move_out(t, src, off, words, addr));
        return t;
    }
    uint8_t move_in(uint16_t dst, uint32_t off, uint32_t words, uint64_t addr, bool last = false, bool report = false) {
        const uint8_t t = tag(report);
        push(cmd_move_in(t, dst, off, words, addr, last));
        return t;
    }
    uint8_t fence(bool report = false) { const uint8_t t = tag(report); push(cmd_fence(t)); return t; }
private:
    void push(const Cmd128& c) { cmds.push_back(c); if (report_all) roles.push_back(role_); }
    const char* role_ = "";
    std::array<uint8_t, 256> gens_{};
    uint8_t next_tag_ = 1;
};

}  // namespace g3rt
