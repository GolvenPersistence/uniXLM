#include "gen3rt/profile.h"

#include <algorithm>
#include <cstring>
#include <deque>

namespace g3rt {

Profile::~Profile() { if (dump_) std::fclose(dump_); }

bool Profile::open_dump(const std::string& path) {
    dump_ = std::fopen(path.c_str(), "wb");
    if (dump_) std::fwrite("G3PROF1\n", 1, 8, dump_);
    return dump_ != nullptr;
}

void Profile::add(const std::vector<Cmd128>& cmds, const std::vector<const char*>& roles, uint32_t start,
                  const std::vector<g3::Completion>& cpl) {
    add(cmds, roles, start, cpl, nullptr);
}

void Profile::add(const std::vector<Cmd128>& cmds, const std::vector<const char*>& roles, uint32_t start,
                  const std::vector<g3::Completion>& cpl, const uint32_t* ports) {
    // completion cycle of every command: a tag's completions come in its commands' order
    std::vector<std::deque<size_t>> by_tag(256);
    for (size_t i = 0; i < cmds.size(); ++i) by_tag[(cmds[i].lo >> 8) & 0xFF].push_back(i);
    std::vector<int64_t> at(cmds.size(), -1);
    for (const g3::Completion& c : cpl) {
        std::deque<size_t>& q = by_tag[c.tag];
        if (q.empty()) continue;
        at[q.front()] = int64_t(uint32_t(c.cycle - start));          // relative to the submit (mod 2^32)
        q.pop_front();
    }
    if (dump_) {
        const uint32_t head[10] = {uint32_t(cmds.size()), start, ports ? ports[0] : 0, ports ? ports[1] : 0,
                                   ports ? ports[2] : 0, ports ? ports[3] : 0, ports ? ports[4] : 0,
                                   ports ? ports[5] : 0, ports ? ports[6] : 0, ports ? ports[7] : 0};
        std::fwrite(head, 4, 10, dump_);
        for (size_t i = 0; i < cmds.size(); ++i) {
            const char* r = i < roles.size() && roles[i] ? roles[i] : "";
            const uint8_t n = uint8_t(std::min<size_t>(std::strlen(r), 255));
            std::fwrite(&cmds[i].lo, 8, 1, dump_); std::fwrite(&cmds[i].hi, 8, 1, dump_);
            std::fwrite(&at[i], 8, 1, dump_); std::fwrite(&n, 1, 1, dump_); std::fwrite(r, 1, n, dump_);
        }
    }
    int64_t prev = 0;
    for (size_t i = 0; i < cmds.size(); ++i) {
        if (at[i] < 0) continue;
        Entry& e = roles_[i < roles.size() && roles[i] ? roles[i] : ""];
        e.count += 1;
        if (at[i] > prev) {
            e.cycles += uint64_t(at[i] - prev);
            total_ += uint64_t(at[i] - prev);
            prev = at[i];
        }
    }
}

void Profile::print(std::FILE* f, double mhz) const {
    std::vector<std::pair<std::string, Entry>> v(roles_.begin(), roles_.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.cycles > b.second.cycles; });
    std::fprintf(f, "profile: %llu cycles (%.2f ms at %.1f MHz)\n", static_cast<unsigned long long>(total_),
                 mhz > 0 ? double(total_) / mhz / 1e3 : 0.0, mhz);
    for (const auto& [name, e] : v)
        std::fprintf(f, "  %-12s %12llu cycles %6.1f %%  %8llu commands\n", name.empty() ? "(none)" : name.c_str(),
                     static_cast<unsigned long long>(e.cycles), total_ ? 100.0 * double(e.cycles) / double(total_) : 0.0,
                     static_cast<unsigned long long>(e.count));
}

}  // namespace g3rt
