"""Keep single-token decode on regular CPU attention while offloading prefill.

Apply to the pinned AICAS llama.cpp checkout before rebuilding libllama.
Requires GGML_GEN3_ATTN=1 and GGML_GEN3_ATTN_PREFILL_ONLY=1 at runtime.
Without both flags the existing graph construction is unchanged.
"""
from pathlib import Path
import argparse

ANCHOR = "    if (cparams.flash_attn && kq_b == nullptr) {"
REPLACEMENT = """    const char * gen3_attn = std::getenv("GGML_GEN3_ATTN");
    const char * gen3_prefill = std::getenv("GGML_GEN3_ATTN_PREFILL_ONLY");
    const bool gen3_cpu_decode = gen3_attn && std::strcmp(gen3_attn, "1") == 0 &&
        gen3_prefill && std::strcmp(gen3_prefill, "1") == 0 &&
        (q->ne[1] == 1 || q->ne[0] < 32 || q->ne[0] > 128 || q->ne[0] % 32 ||
         q->ne[2] > 16 || k->ne[2] > 4 || k->ne[1] > 2048 || q->ne[3] != 1 ||
         sinks || hparams.f_max_alibi_bias != 0 || hparams.attn_soft_cap ||
         kq_scale != 1.0f/std::sqrt(float(q->ne[0])));
    if (cparams.flash_attn && kq_b == nullptr && !gen3_cpu_decode) {"""


KV_HELPER = """#include <cstdlib>
#include <cstring>

static bool gen3_prefill_cpu_decode() {
    const char * a = std::getenv("GGML_GEN3_ATTN");
    const char * p = std::getenv("GGML_GEN3_ATTN_PREFILL_ONLY");
    return a && p && std::strcmp(a, "1") == 0 && std::strcmp(p, "1") == 0;
}

"""


def apply(root: Path):
    graph = root / "src/llama-graph.cpp"
    kv = root / "src/llama-kv-cache.cpp"
    source = graph.read_text(encoding="utf-8")
    cache = kv.read_text(encoding="utf-8")
    if REPLACEMENT not in source:
        if source.count(ANCHOR) != 1:
            raise ValueError("Unsupported llama-graph attention anchor")
        source = source.replace(ANCHOR, REPLACEMENT, 1)
    if KV_HELPER not in cache:
        ctor = "    GGML_ASSERT(kv_size % n_pad == 0);"
        pad = "    return cparams.flash_attn ? 256u : 32u;"
        if cache.count(ctor) != 1 or cache.count(pad) != 1:
            raise ValueError("Unsupported KV cache policy anchors")
        cache = KV_HELPER + cache
        cache = cache.replace(ctor, """    if (gen3_prefill_cpu_decode()) {
        GGML_ASSERT(type_k == GGML_TYPE_F16 && type_v == GGML_TYPE_F16);
        v_trans = this->v_trans = true;
    }
""" + ctor, 1)
        cache = cache.replace(pad, "    return cparams.flash_attn && !gen3_prefill_cpu_decode() ? 256u : 32u;", 1)
    # All anchors are checked before either source is changed.
    graph.write_text(source, encoding="utf-8")
    kv.write_text(cache, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llama_root", type=Path)
    apply(parser.parse_args().llama_root)
