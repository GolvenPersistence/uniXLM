// program.hex reader / runner and the result files (the formats of tb/gen3/node/g3_node_tb.sv).
#include <cstdio>
#include <fstream>
#include <string>

#include "g3emu.h"

namespace g3 {

namespace {
int hexval(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

Cmd128 low128(const Word& w) {
    Cmd128 c;
    for (int i = 7; i >= 0; --i) {
        c.lo = (c.lo << 8) | w.b[i];
        c.hi = (c.hi << 8) | w.b[8 + i];
    }
    return c;
}
}  // namespace

bool read_program(const std::string& path, std::vector<ProgramRecord>& recs, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    std::string line;
    size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        if (line.empty()) continue;
        if (line.size() != 66) {
            if (err) *err = path + ":" + std::to_string(lineno) + ": expected 66 hex digits";
            return false;
        }
        ProgramRecord r;
        const int k0 = hexval(line[0]), k1 = hexval(line[1]);
        if (k0 < 0 || k1 < 0) {
            if (err) *err = path + ":" + std::to_string(lineno) + ": bad hex";
            return false;
        }
        r.kind = uint8_t(k0 * 16 + k1);
        for (int i = 0; i < 32; ++i) {
            // payload hex digits 2..65, most significant byte first: byte 31 at digits 2-3
            const int hi = hexval(line[2 + 2 * (31 - i)]), lo = hexval(line[3 + 2 * (31 - i)]);
            if (hi < 0 || lo < 0) {
                if (err) *err = path + ":" + std::to_string(lineno) + ": bad hex";
                return false;
            }
            r.payload.b[i] = uint8_t(hi * 16 + lo);
        }
        recs.push_back(r);
    }
    return true;
}

void run_program(Node& node, const std::vector<ProgramRecord>& recs) {
    // preloads land in external memory before the run
    for (size_t pc = 0; pc < recs.size();) {
        const Cmd128 p = low128(recs[pc].payload);
        if (recs[pc].kind == 0x06) {
            const size_t n = size_t(p.field(0, 20));
            std::vector<Word> words;
            for (size_t k = 0; k < n && pc + 1 + k < recs.size(); ++k) words.push_back(recs[pc + 1 + k].payload);
            node.preload(p.field(20, 48), words.data(), words.size());
            pc += 1 + n;
        } else if (recs[pc].kind == 0x02) {
            pc += 1 + size_t(p.field(0, 20));
        } else {
            ++pc;
        }
    }
    for (size_t pc = 0; pc < recs.size();) {
        const ProgramRecord& r = recs[pc];
        const Cmd128 p = low128(r.payload);
        switch (r.kind) {
        case 0x01:
            node.execute(p);
            ++pc;
            break;
        case 0x02: {
            const size_t n = size_t(p.field(0, 20));
            std::vector<Word> words;
            for (size_t k = 0; k < n && pc + 1 + k < recs.size(); ++k) words.push_back(recs[pc + 1 + k].payload);
            node.push(p.field(20, 48), words.data(), words.size());
            pc += 1 + n;
            break;
        }
        case 0x05:
            return;
        case 0x06:
            pc += 1 + size_t(p.field(0, 20));
            break;
        default:   // 03 WAIT (program order already waits), 04 MARK, stray DATA
            ++pc;
            break;
        }
    }
}

bool write_hmem(const Node& node, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    for (uint64_t a : node.ext().addresses()) {
        const Word* w = node.ext().find(a);
        std::fprintf(f, "%llx ", static_cast<unsigned long long>(a));
        for (int i = 31; i >= 0; --i) std::fprintf(f, "%02x", w->b[i]);
        std::fputc('\n', f);
    }
    std::fclose(f);
    return true;
}

bool write_completions(const Node& node, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    size_t i = 0;
    for (const Completion& c : node.completions())
        std::fprintf(f, "%zu %u %u %u\n", i++, unsigned(c.tag), unsigned(c.status), unsigned(c.data));
    std::fclose(f);
    return true;
}

}  // namespace g3
