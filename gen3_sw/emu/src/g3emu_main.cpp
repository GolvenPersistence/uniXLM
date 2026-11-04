// g3emu: run a Gen3 Node program (program.hex) on the command-level emulator.
//
//   g3emu run <case_dir> [--out DIR] [--attn-requantize] [--vector-rmsnorm] [--threads N] [--trace] [--quiet]
//
// Reads <case_dir>/program.hex, writes DIR/hmem_out.hex and DIR/completions.txt (DIR defaults to <case_dir>/emu_out).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

#include "g3emu.h"

#ifdef _WIN32
#include <direct.h>
static int make_dir(const std::string& p) { return _mkdir(p.c_str()); }
#else
static int make_dir(const std::string& p) { return mkdir(p.c_str(), 0755); }
#endif

static void make_dirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i)
        if (i == path.size() || path[i] == '/' || path[i] == '\\') make_dir(path.substr(0, i));
}

static int usage() {
    std::fprintf(stderr, "usage: g3emu run <case_dir> [--out DIR] [--attn-requantize] [--vector-rmsnorm] [--threads N]\n"
                         "                 [--trace] [--quiet]\n");
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[1], "run") != 0) return usage();
    std::string dir = argv[2], out;
    g3::NodeOptions opt;
    bool quiet = false;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--attn-requantize")) opt.attn_requantize = true;
        else if (!std::strcmp(argv[i], "--attn-requantize-mask") && i + 1 < argc) {
            opt.attn_requantize = true;
            opt.attn_requantize_mask = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--quiet")) quiet = true;
        else if (!std::strcmp(argv[i], "--trace")) opt.trace = true;
        else if (!std::strcmp(argv[i], "--vector-rmsnorm")) opt.vector_stub = false;
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) opt.threads = std::atoi(argv[++i]);
        else return usage();
    }
    if (out.empty()) out = dir + "/emu_out";
    std::vector<g3::ProgramRecord> recs;
    std::string err;
    if (!g3::read_program(dir + "/program.hex", recs, &err)) {
        std::fprintf(stderr, "g3emu: %s\n", err.c_str());
        return 1;
    }
    g3::Node node(opt);
    const auto t0 = std::chrono::steady_clock::now();
    g3::run_program(node, recs);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    make_dirs(out);
    if (!g3::write_hmem(node, out + "/hmem_out.hex") || !g3::write_completions(node, out + "/completions.txt")) {
        std::fprintf(stderr, "g3emu: cannot write results to %s\n", out.c_str());
        return 1;
    }
    for (const std::string& w : node.warnings()) std::fprintf(stderr, "g3emu: warning: %s\n", w.c_str());
    if (!quiet)
        std::printf("g3emu: %zu records, %zu completions, %zu pending waits, %zu external words, %.3f s\n", recs.size(),
                    node.completions().size(), node.pending_waits(), node.ext().size(), sec);
    return 0;
}
