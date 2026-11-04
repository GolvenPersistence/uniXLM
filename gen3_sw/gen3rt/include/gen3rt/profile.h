// Cycles of the Node per command role, from the completion records of runs in which every command reported
// (Program::report_all): in program order, command i is charged the cycles from the latest completion before it
// (or the submit) to its own completion; commands that overlap an earlier one cost what they add. Board only (the
// completion records carry the Node's cycle counter there).
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "gen3rt/cmd.h"
#include "g3emu.h"

namespace g3rt {

class Profile {
public:
    ~Profile();
    // one run: its commands (with their roles), the cycle counter at the submit, its completions in arrival order;
    // ports: the 4 read-port word counters before the run, then after
    void add(const std::vector<Cmd128>& cmds, const std::vector<const char*>& roles, uint32_t start,
             const std::vector<g3::Completion>& cpl, const uint32_t* ports);
    void add(const std::vector<Cmd128>& cmds, const std::vector<const char*>& roles, uint32_t start,
             const std::vector<g3::Completion>& cpl);   // no port counters (ggml-gen3 links against this one)
    // every later run to `path`: "G3PROF1\n", then per run {u32 commands, u32 start cycle, u32 ports[8]} and per
    // command {u64 lo, u64 hi, i64 completion cycle - start (-1: none), u8 role length, role}
    bool open_dump(const std::string& path);
    void print(std::FILE* f, double mhz) const;
    uint64_t total() const { return total_; }
private:
    struct Entry { uint64_t cycles = 0, count = 0; };
    std::map<std::string, Entry> roles_;
    uint64_t total_ = 0;
    std::FILE* dump_ = nullptr;
};

}  // namespace g3rt
