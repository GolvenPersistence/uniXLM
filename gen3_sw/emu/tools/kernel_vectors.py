"""Kernel test vectors for the C++ emulator: random inputs and the Python reference outputs (g2lib, core_vectors,
g3lib, attention_numerics with G3_ACC=bf48), written as text files that build/g3emu_kcheck compares bit for bit.

    python gen3_sw/emu/tools/kernel_vectors.py [--out gen3_sw/emu/out/kernels] [--seed 1]
    gen3_sw/emu/build/g3emu_kcheck gen3_sw/emu/out/kernels

File format: sections "@<name> <count>" followed by <count> hexadecimal values.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["G3_ACC"] = "bf48"           # the v3 numerics (bf48 accumulation) in g2lib and Gen3Attention

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tb" / "gen2"))
sys.path.insert(0, str(REPO / "tb" / "gen3"))
import g2lib  # noqa: E402
import g3lib  # noqa: E402
from core_vectors import activation_bits, rope_bits  # noqa: E402


class Out:
    def __init__(self, path: Path):
        self.lines = []
        self.path = path

    def sec(self, name: str, values) -> None:
        vals = [int(v) & 0xFFFFFFFFFFFFFFFF for v in np.asarray(values).reshape(-1).tolist()]
        self.lines.append(f"@{name} {len(vals)}")
        for i in range(0, len(vals), 16):
            self.lines.append(" ".join(f"{v:x}" for v in vals[i:i + 16]))

    def write(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")
        print("wrote", self.path)


def special_f32() -> list[int]:
    return [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00400000, 0x00800000, 0x80800000,
            0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0x3F800000, 0xBF800000,
            0x41000000, 0xC1000000, 0x40800000, 0xC0800000, 0x40000000, 0xC0000000, 0x3D800000, 0x43E00000,
            0x43E10000, 0x43F00000, 0x3B000000, 0x3A800000, 0x39800000, 0x0C000000, 0x01000000, 0x00000400]


def wide_f32(rng, n: int) -> np.ndarray:
    v = rng.normal(size=n) * np.exp2(rng.integers(-140, 128, size=n))
    with np.errstate(over="ignore"):
        return g2lib.f32_bits(v.astype(np.float32))


def gen_quant(rng, out_dir: Path) -> None:
    o = Out(out_dir / "quant.txt")
    rows = []
    for i in range(3000):
        kind = i % 6
        if kind == 0:
            r = wide_f32(rng, 32)
        elif kind == 1:      # one binade per block, random spread
            e = rng.integers(-130, 120)
            r = g2lib.f32_bits((rng.normal(size=32) * 2.0 ** float(e)).astype(np.float32))
        elif kind == 2:      # specials mixed in
            r = wide_f32(rng, 32)
            sp = special_f32()
            for j in rng.integers(0, 32, size=4):
                r[j] = sp[rng.integers(0, len(sp))]
        elif kind == 3:      # E4M3 boundary values at a scale: midpoints and exact levels
            s = rng.integers(1, 250)
            q = g2lib._E4M3_INT[rng.integers(0, 127, size=32)].astype(np.float64)
            mid = q + rng.integers(0, 2, size=32) * 0.5 * np.maximum(1, 2.0 ** np.floor(np.log2(np.maximum(q, 1)) - 3))
            v = mid * np.exp2(float(s) - 136) * np.where(rng.integers(0, 2, size=32) == 1, -1, 1)
            with np.errstate(over="ignore", under="ignore"):
                r = g2lib.f32_bits(v.astype(np.float32))
        elif kind == 4:      # subnormal / tiny blocks
            r = (rng.integers(0, 1 << 23, size=32) | (rng.integers(0, 2, size=32) << 31)).astype(np.uint32)
        else:                # zeros and a few values
            r = np.zeros(32, dtype=np.uint32)
            r[rng.integers(0, 32, size=3)] = wide_f32(rng, 3)
        rows.append(np.asarray(r, dtype=np.uint32))
    vals = np.stack(rows)
    codes = np.zeros(vals.shape, dtype=np.int64)
    scales = np.zeros(len(vals), dtype=np.int64)
    for i, r in enumerate(vals):
        s, c = g2lib.quantize_row(r)
        scales[i] = s
        codes[i] = c
    o.sec("values", vals)
    o.sec("codes", codes)
    o.sec("scales", scales)
    dc, ds, dv = [], [], []
    for s in [0, 1, 2, 8, 9, 50, 100, 118, 119, 127, 128, 136, 150, 200, 253, 254, 255]:
        for c in range(256):
            dc.append(c); ds.append(s); dv.append(g2lib.decode_e4m3_fp32(c, s))
    o.sec("dec_codes", dc)
    o.sec("dec_scales", ds)
    o.sec("dec_out", dv)
    o.write()


def gen_gemm(rng, out_dir: Path) -> None:
    o = Out(out_dir / "gemm.txt")
    cases = []
    for (m, n, g, lo, hi) in ((5, 7, 3, 116, 128), (16, 33, 4, 100, 140), (1, 64, 18, 116, 128), (33, 40, 2, 60, 200),
                              (8, 8, 6, 0, 12), (8, 8, 6, 245, 254), (4, 6, 40, 120, 124), (12, 20, 9, 0, 254)):
        a, sa = g2lib.random_mxfp8(rng, m, 32 * g, lo, hi)
        b, sb = g2lib.random_mxfp8(rng, n, 32 * g, lo, hi)
        cases.append((a, sa, b, sb))
    # renormalisation pressure: many groups of large equal-sign products at one exponent
    g = 80
    a = np.full((2, 32 * g), 0x7E, dtype=np.int64); b = np.full((3, 32 * g), 0x7E, dtype=np.int64)
    b[1] = 0x7D; b[2, ::3] = 0xFE
    cases.append((a, np.full((2, g), 120, dtype=np.int64), b, np.full((3, g), 121, dtype=np.int64)))
    # NaN codes and 0xFF scales
    a, sa = g2lib.random_mxfp8(rng, 6, 96, 116, 128)
    b, sb = g2lib.random_mxfp8(rng, 5, 96, 116, 128)
    a[1, 40] = 0x7F; b[2, 3] = 0xFF; sa[3, 2] = 0xFF; sb[4, 0] = 0xFF
    cases.append((a, sa, b, sb))
    o.sec("cases", [len(cases)])
    for i, (a, sa, b, sb) in enumerate(cases):
        out = g2lib.gemm_bf48(a, sa, b, sb)
        o.sec(f"g{i}.shape", [a.shape[0], b.shape[0], a.shape[1] // 32])
        o.sec(f"g{i}.a", a); o.sec(f"g{i}.sa", sa); o.sec(f"g{i}.b", b); o.sec(f"g{i}.sb", sb)
        o.sec(f"g{i}.out", out)
    o.write()


def gen_epi(rng, out_dir: Path) -> None:
    o = Out(out_dir / "epi.txt")
    x = np.concatenate([g2lib.f32_bits(rng.uniform(-10, 10, size=4000).astype(np.float32)),
                        g2lib.f32_bits(rng.uniform(-0.2, 0.2, size=500).astype(np.float32)),
                        wide_f32(rng, 500), np.array(special_f32(), dtype=np.uint32),
                        g2lib.f32_bits(np.array([1.9999999, 2.0, 3.9999998, 4.0, 7.9999995, 8.0, -1.9999999, -2.0,
                                                 -4.0, -8.0, 0.0625, 0.125, -0.0625], dtype=np.float32))])
    x = x.astype(np.uint32)
    o.sec("act_in", x)
    o.sec("silu_out", activation_bits(x, gelu=False))
    o.sec("gelu_out", activation_bits(x, gelu=True))
    rx = g2lib.f32_bits((rng.normal(size=(1, 256)) * 5).astype(np.float32))
    rc = g2lib.f32_bits(rng.uniform(-1, 1, size=(1, 256)).astype(np.float32))
    rx[0, :8] = special_f32()[:8]
    o.sec("rope_x", rx); o.sec("rope_c", rc); o.sec("rope_out", rope_bits(rx, rc))
    p = np.concatenate([wide_f32(rng, 3000), np.array(special_f32(), dtype=np.uint32)]).astype(np.uint32)
    q = np.concatenate([wide_f32(rng, 3000), np.array(special_f32()[::-1], dtype=np.uint32)]).astype(np.uint32)
    o.sec("ar_x", p); o.sec("ar_y", q)
    o.sec("add_out", g2lib.fp32_add_bits(p, q)); o.sec("mul_out", g2lib.fp32_mul_bits(p, q))
    o.write()


def gen_fold(rng, out_dir: Path) -> None:
    o = Out(out_dir / "fold.txt")
    cases = [(8, 64, 2), (5, 128, 0), (3, 576, 3), (4, 320, 1), (2, 192, 2)]
    o.sec("cases", [len(cases)])
    for i, (r, w, eps) in enumerate(cases):
        xc, xs = g2lib.random_mxfp8(rng, r, w, 110, 135)
        xb = g2lib.decode_matrix_fp32(xc, xs)
        gc, gs = g2lib.random_mxfp8(rng, 1, w, 120, 128)
        gb = g2lib.decode_matrix_fp32(gc, gs)[0]
        o.sec(f"f{i}.shape", [r, w, eps])
        o.sec(f"f{i}.x", xb); o.sec(f"f{i}.gamma", gb)
        o.sec(f"f{i}.rms", g3lib.inv_rms_rows(xb, eps))
        o.sec(f"f{i}.norm", g2lib.rmsnorm_rows(xb, gb, eps))
    o.write()


def gen_attn(rng, out_dir: Path) -> None:
    from attention_numerics import Gen3Attention
    na = Gen3Attention(REPO)
    ar = na.ar
    o = Out(out_dir / "attn.txt")
    cases = [(32, 32, 64, 0, 1.0), (16, 90, 128, 100, 3.0), (1, 33, 64, 32, 2.0), (24, 150, 64, 126, 6.0),
             (8, 64, 224, 100, 1.5), (16, 48, 160, 20, 4.0), (5, 77, 64, 72, 6.0), (3, 129, 96, 126, 12.0),
             (32, 256, 64, 300, 2.0), (4, 200, 32, 10, 30.0)]
    o.sec("cases", [len(cases)])
    for i, (m, n, dh, qbase, spread) in enumerate(cases):
        causal = qbase < n - 1
        q = (rng.normal(size=(m, dh)) * spread).astype(np.float32)
        k = (rng.normal(size=(n, dh)) * spread).astype(np.float32)
        v = (rng.normal(size=(n, dh)) * np.exp(rng.normal(size=(n, 1)))).astype(np.float32)
        _, info = na.run(q, k, v, causal, qbase, block=64)
        cap = (n + 31) // 32 * 32
        qc, qs, _, _ = ar.quantize(q, "mxfp8")
        kp = np.zeros((cap, dh), dtype=np.float32); kp[:n] = k
        kc, ks, _, _ = ar.quantize(kp, "mxfp8")
        vv = np.zeros((dh, cap), dtype=np.float32); vv[:, :n] = v.T
        vc, vs, _, _ = ar.quantize(vv, "mxfp8")
        o.sec(f"a{i}.shape", [m, n, dh, qbase])
        o.sec(f"a{i}.q", qc); o.sec(f"a{i}.qs", qs)
        o.sec(f"a{i}.k", kc); o.sec(f"a{i}.ks", ks)
        o.sec(f"a{i}.vt", vc); o.sec(f"a{i}.vts", vs)
        o.sec(f"a{i}.out", info["codes"]); o.sec(f"a{i}.outs", info["scales"])
    o.write()


def gen_vt(rng, out_dir: Path) -> None:
    o = Out(out_dir / "vt.txt")
    # requantisation of a stored code to a larger scale vs decode + quantise
    codes, so_, sn_, exp = [], [], [], []
    pairs = [(a, b) for a in (0, 1, 7, 50, 100, 118, 119, 120, 127, 140, 200, 250) for b in range(a + 1, min(a + 20, 255))]
    pairs += [(a, 254) for a in (0, 100, 200, 253)] + [(10, 255), (255, 255)]
    overflow = 0
    for s_o, s_n in pairs:
        for c in range(256):
            codes.append(c); so_.append(s_o); sn_.append(s_n)
            if (c & 0x7F) == 0 or s_n == s_o:      # VtCache re-rounds nonzero codes when the scale changes
                exp.append(c)
                continue
            dec = g2lib.decode_e4m3_fp32(c, s_o)
            if (dec & 0x7FFFFFFF) == 0x7F800000 and s_o != 0xFF and s_n != 0xFF:
                # VtCache decodes through FP32, which overflows for scales >= 247; the RTL re-rounds in the code
                # domain. The same quantisation, shifted to scale 136 (decode exact), is the exact reference.
                overflow += 1
                exp.append(int(g3lib._quant(g2lib.decode_e4m3_fp32(c, 136), 136 + (s_n - s_o))))
            else:
                exp.append(int(g3lib._quant(dec, s_n)))
    print(f"vt requant: {overflow} cases with an FP32-overflowing decode use the exact (shifted-scale) reference")
    o.sec("rq_code", codes); o.sec("rq_so", so_); o.sec("rq_sn", sn_); o.sec("rq_out", exp)
    # append sequences: prefill block(s), then single tokens (the decode tail rewrite)
    seqs = [(64, 96, 32, 45), (32, 64, 0, 40), (96, 64, 16, 50), (64, 128, 64, 70)]
    o.sec("seqs", [len(seqs)])
    for i, (dh, cap, pre, total) in enumerate(seqs):
        vb = g2lib.f32_bits((rng.normal(size=(total, dh)) * np.exp(rng.normal(size=(total, 1)) * 3)).astype(np.float32))
        vb[rng.integers(0, total, size=6), rng.integers(0, dh, size=6)] = 0
        cache = g3lib.VtCache(dh, cap)
        if pre:
            cache.append_block(0, vb[:pre])
        for t in range(pre, total):
            cache.append_token(t, vb[t])
        o.sec(f"s{i}.shape", [dh, cap, pre, total])
        o.sec(f"s{i}.v", vb)
        o.sec(f"s{i}.codes", cache.codes); o.sec(f"s{i}.scales", cache.scales)
    o.write()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "out" / "kernels"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=["quant", "gemm", "epi", "fold", "attn", "vt"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for name in args.only:
        globals()[f"gen_{name}"](rng, out)


if __name__ == "__main__":
    main()
