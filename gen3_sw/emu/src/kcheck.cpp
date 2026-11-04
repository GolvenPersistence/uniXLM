// g3emu_kcheck: compare the C++ kernels with the Python reference outputs of tools/kernel_vectors.py.
//
//   g3emu_kcheck <dir>     (dir holds quant.txt gemm.txt epi.txt fold.txt attn.txt vt.txt)
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "g3emu_kernels.h"

using namespace g3;
using Sections = std::map<std::string, std::vector<uint64_t>>;

static bool load(const std::string& path, Sections& s) {
    std::ifstream in(path);
    if (!in) return false;
    std::string tok, name;
    size_t left = 0;
    while (in >> tok) {
        if (tok[0] == '@') {
            name = tok.substr(1);
            in >> left;
            s[name].reserve(left);
        } else {
            s[name].push_back(std::stoull(tok, nullptr, 16));
        }
    }
    return true;
}

template <class T>
static std::vector<T> as(const std::vector<uint64_t>& v) {
    return std::vector<T>(v.begin(), v.end());
}

struct Tally {
    long total = 0, bad = 0;
    std::string first;
    void check(bool ok, const std::string& what) {
        ++total;
        if (!ok) {
            if (bad == 0) first = what;
            ++bad;
        }
    }
    bool report(const char* name) const {
        std::printf("%-8s %s  %ld/%ld equal%s%s\n", name, bad ? "FAIL" : "PASS", total - bad, total,
                    bad ? "  first mismatch: " : "", bad ? first.c_str() : "");
        return bad == 0;
    }
};

static std::string hx(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%llx", (unsigned long long)v);
    return b;
}

static bool check_quant(Sections& s) {
    Tally t;
    const auto vals = as<uint32_t>(s["values"]);
    const auto codes = s["codes"], scales = s["scales"];
    const size_t n = scales.size();
    for (size_t i = 0; i < n; ++i) {
        uint8_t c[32], sc;
        quantize_block32(&vals[32 * i], c, &sc);
        bool ok = sc == scales[i];
        for (int k = 0; k < 32; ++k) ok &= c[k] == codes[32 * i + k];
        t.check(ok, "block " + std::to_string(i) + " scale " + hx(sc) + " vs " + hx(scales[i]));
    }
    const auto dc = s["dec_codes"], ds = s["dec_scales"], dv = s["dec_out"];
    for (size_t i = 0; i < dc.size(); ++i) {
        const uint32_t r = decode_e4m3(uint8_t(dc[i]), uint8_t(ds[i]));
        t.check(r == dv[i], "decode code " + hx(dc[i]) + " scale " + hx(ds[i]) + ": " + hx(r) + " vs " + hx(dv[i]));
    }
    return t.report("quant");
}

static bool check_gemm(Sections& s) {
    Tally t;
    const int cases = int(s["cases"][0]);
    for (int i = 0; i < cases; ++i) {
        const std::string p = "g" + std::to_string(i) + ".";
        const auto sh = s[p + "shape"];
        const int m = int(sh[0]), n = int(sh[1]), g = int(sh[2]);
        const auto a = as<uint8_t>(s[p + "a"]), sa = as<uint8_t>(s[p + "sa"]);
        const auto b = as<uint8_t>(s[p + "b"]), sb = as<uint8_t>(s[p + "sb"]);
        const auto out = s[p + "out"];
        std::vector<uint32_t> r(size_t(m) * n);
        gemm_bf48(m, n, g, a.data(), sa.data(), b.data(), sb.data(), r.data(), n);
        for (int k = 0; k < m * n; ++k)
            t.check(r[k] == out[k], "case " + std::to_string(i) + " [" + std::to_string(k / n) + "," +
                                        std::to_string(k % n) + "] " + hx(r[k]) + " vs " + hx(out[k]));
    }
    return t.report("gemm");
}

static bool check_epi(Sections& s) {
    Tally t;
    const auto x = s["act_in"], so = s["silu_out"], go = s["gelu_out"];
    for (size_t i = 0; i < x.size(); ++i) {
        const uint32_t a = silu_pwl(uint32_t(x[i])), b = gelu_pwl(uint32_t(x[i]));
        t.check(a == so[i], "silu(" + hx(x[i]) + ") " + hx(a) + " vs " + hx(so[i]));
        t.check(b == go[i], "gelu(" + hx(x[i]) + ") " + hx(b) + " vs " + hx(go[i]));
    }
    const auto rx = as<uint32_t>(s["rope_x"]), rc = as<uint32_t>(s["rope_c"]);
    const auto ro = s["rope_out"];
    std::vector<uint32_t> r(rx.size());
    rope_pairs(rx.data(), rc.data(), r.data(), int(rx.size()));
    for (size_t i = 0; i < r.size(); ++i) t.check(r[i] == ro[i], "rope " + std::to_string(i));
    const auto px = s["ar_x"], py = s["ar_y"], ao = s["add_out"], mo = s["mul_out"];
    for (size_t i = 0; i < px.size(); ++i) {
        const uint32_t a = f32_add(uint32_t(px[i]), uint32_t(py[i])), m = f32_mul(uint32_t(px[i]), uint32_t(py[i]));
        t.check(a == ao[i], "add " + hx(px[i]) + " + " + hx(py[i]) + ": " + hx(a) + " vs " + hx(ao[i]));
        t.check(m == mo[i], "mul " + hx(px[i]) + " * " + hx(py[i]) + ": " + hx(m) + " vs " + hx(mo[i]));
    }
    return t.report("epilogue");
}

static bool check_fold(Sections& s) {
    Tally t;
    const int cases = int(s["cases"][0]);
    for (int i = 0; i < cases; ++i) {
        const std::string p = "f" + std::to_string(i) + ".";
        const auto sh = s[p + "shape"];
        const int r = int(sh[0]), w = int(sh[1]), eps = int(sh[2]);
        const auto x = as<uint32_t>(s[p + "x"]), gm = as<uint32_t>(s[p + "gamma"]);
        const auto rms = s[p + "rms"], norm = s[p + "norm"];
        std::vector<uint32_t> out(w);
        for (int k = 0; k < r; ++k) {
            const uint32_t v = inv_rms_row(&x[size_t(k) * w], w, eps);
            t.check(v == rms[k], "rms case " + std::to_string(i) + " row " + std::to_string(k) + " " + hx(v) + " vs " +
                                     hx(rms[k]));
            rmsnorm_row(&x[size_t(k) * w], gm.data(), w, eps, out.data());
            for (int j = 0; j < w; ++j) t.check(out[j] == norm[size_t(k) * w + j], "norm case " + std::to_string(i));
        }
    }
    return t.report("fold");
}

static bool check_attn(Sections& s) {
    Tally t;
    const int cases = int(s["cases"][0]);
    for (int i = 0; i < cases; ++i) {
        const std::string p = "a" + std::to_string(i) + ".";
        const auto sh = s[p + "shape"];
        const int m = int(sh[0]), n = int(sh[1]), dh = int(sh[2]), qbase = int(sh[3]);
        const int cap = (n + 31) / 32 * 32, G = dh / 32;
        const auto q = as<uint8_t>(s[p + "q"]), qs = as<uint8_t>(s[p + "qs"]);
        const auto k = as<uint8_t>(s[p + "k"]), ks = as<uint8_t>(s[p + "ks"]);
        const auto vt = as<uint8_t>(s[p + "vt"]), vts = as<uint8_t>(s[p + "vts"]);
        const auto out = s[p + "out"], outs = s[p + "outs"];
        AttnKV kv;
        kv.dh = dh; kv.cap = cap;
        kv.k_codes = k.data(); kv.k_scales = ks.data(); kv.vt_codes = vt.data(); kv.vt_scales = vts.data();
        std::vector<uint8_t> oc(dh);
        for (int r = 0; r < m; ++r) {
            const long lim = long(qbase) + r + 1;
            const int visible = int(lim < 0 ? 0 : lim > n ? n : lim);
            uint8_t os;
            attention_row(kv, &q[size_t(r) * dh], &qs[size_t(r) * G], n, visible, oc.data(), &os);
            bool ok = os == outs[r];
            int bad_d = -1;
            for (int d = 0; d < dh; ++d)
                if (oc[d] != out[size_t(r) * dh + d] && bad_d < 0) { ok = false; bad_d = d; }
            t.check(ok, "case " + std::to_string(i) + " row " + std::to_string(r) + " scale " + hx(os) + " vs " +
                            hx(outs[r]) + (bad_d >= 0 ? " dim " + std::to_string(bad_d) : std::string()));
        }
    }
    return t.report("attn");
}

static bool check_vt(Sections& s) {
    Tally t;
    const auto c = s["rq_code"], so = s["rq_so"], sn = s["rq_sn"], ro = s["rq_out"];
    for (size_t i = 0; i < c.size(); ++i) {
        const uint8_t r = vt_requant_code(uint8_t(c[i]), uint8_t(so[i]), uint8_t(sn[i]));
        t.check(r == ro[i], "requant code " + hx(c[i]) + " " + hx(so[i]) + "->" + hx(sn[i]) + ": " + hx(r) + " vs " +
                                hx(ro[i]));
    }
    const int seqs = int(s["seqs"][0]);
    for (int i = 0; i < seqs; ++i) {
        const std::string p = "s" + std::to_string(i) + ".";
        const auto sh = s[p + "shape"];
        const int dh = int(sh[0]), cap = int(sh[1]), pre = int(sh[2]), total = int(sh[3]);
        const auto v = as<uint32_t>(s[p + "v"]);
        if (pre > total || total > cap || v.size() != size_t(total) * dh) {
            t.check(false, "sequence " + std::to_string(i) + ": bad shape");
            continue;
        }
        const auto ec = s[p + "codes"], es = s[p + "scales"];
        std::vector<uint8_t> codes(size_t(dh) * cap, 0), scales(size_t(dh) * (cap / 32), 127);
        auto block = [&](int t0, int m) {   // whole cells from FP32 (append_block)
            for (int j0 = 0; j0 < m; j0 += 32) {
                const int j = (t0 + j0) / 32;
                for (int d = 0; d < dh; ++d) {
                    uint32_t col[32] = {0};
                    for (int k = 0; k < 32 && j0 + k < m; ++k) col[k] = v[size_t(t0 + j0 + k) * dh + d];
                    uint8_t cc[32], sc;
                    quantize_block32(col, cc, &sc);
                    std::memcpy(&codes[size_t(d) * cap + 32 * j], cc, 32);
                    scales[size_t(d) * (cap / 32) + j] = sc;
                }
            }
        };
        if (pre) block(0, pre);
        for (int tk = pre; tk < total; ++tk) {
            if (tk % 32 == 0) { block(tk, 1); continue; }
            const int j = tk / 32, kt = tk % 32;
            for (int d = 0; d < dh; ++d) {
                uint8_t* cell = &codes[size_t(d) * cap + 32 * j];
                const uint8_t old = scales[size_t(d) * (cap / 32) + j];
                bool zero = true;
                for (int k = 0; k < 32; ++k) zero &= (cell[k] & 0x7F) == 0;
                const uint32_t x = v[size_t(tk) * dh + d];
                const uint8_t nw = vt_tail_scale(old, zero, x);
                for (int k = 0; k < 32; ++k) cell[k] = k == kt ? quantize_e4m3(x, nw) : vt_requant_code(cell[k], old, nw);
                scales[size_t(d) * (cap / 32) + j] = nw;
            }
        }
        bool ok = true;
        for (size_t k = 0; k < codes.size(); ++k) ok &= codes[k] == ec[k];
        for (size_t k = 0; k < scales.size(); ++k) ok &= scales[k] == es[k];
        t.check(ok, "sequence " + std::to_string(i));
    }
    return t.report("vt");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: g3emu_kcheck <dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    struct T { const char* file; bool (*fn)(Sections&); };
    const T tests[] = {{"quant.txt", check_quant}, {"gemm.txt", check_gemm}, {"epi.txt", check_epi},
                       {"fold.txt", check_fold}, {"attn.txt", check_attn}, {"vt.txt", check_vt}};
    int failed = 0;
    for (const T& t : tests) {
        Sections s;
        if (!load(dir + "/" + t.file, s)) {
            std::printf("%-8s MISSING %s\n", t.file, (dir + "/" + t.file).c_str());
            ++failed;
            continue;
        }
        failed += !t.fn(s);
    }
    return failed ? 1 : 0;
}
