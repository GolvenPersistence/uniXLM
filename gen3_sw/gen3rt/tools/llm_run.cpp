// gen3rt_llm: text decoding (llama: SmolVLM2's text model, SmolLM; qwen2 with zero q / k / v biases; qwen3, its q / k
// norm on the Node, or on the host with G3RT_QK_NODE=0 for images before R56) on the Gen3 Node v3 (the bit-exact emulator, or the board), next to llama.cpp on the CPU.
// Every prompt token is fed through the Node one decode step at a time (the KV cache grows in device memory), then
// N tokens are generated greedily by the Node; at each step the Node's logits are compared with llama.cpp's
// (cosine, top-1, the rank of llama.cpp's top-1 among the Node's).
//
//   gen3rt_llm -m model.gguf [-p prompt] [-n 16] [--ctx 512] [--layers L]
//   ... --board --benchmark --repeat 5 --warmup 1 --metrics latency.json
//   ... --benchmark --prompt-tokens 128  # length-controlled cyclic token input
//   ... --benchmark --token-prefill        # the prompt through the decode path, one token at a time
//   ... --perplexity wiki.test.raw --ctx 512 [--chunks K]   llama-perplexity's measure on the Node's logits
//   ... --no-reference   no CPU llama.cpp model (a large model on the board: --dump, compared on the PC)
//   ... --weights-cache W --model-sha256 S   W made from the model whose GGUF has SHA256 S (then -m may be a GGUF
//                                            with its metadata and token embeddings only)
// Program splitting checks (the logits must equal the plain run's: --compare with its dump):
//   --max-group G --max-kv-heads H --max-ffn-columns N   split attention / gate-up below the Node's limits
//   --test-neox   W_q / W_k rows inverse-permuted, then the NEOX program (which permutes them back)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iterator>
#include "gen3rt/sha256.h"
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "gen3rt/device.h"
#include "gen3rt/llm.h"
#ifdef G3RT_BOARD
#include "gen3rt/board.h"
#endif

using namespace g3rt;

namespace {

// The GGUF's tensors, read from the file when asked for (the context holds their shapes only, so a model larger
// than the host's memory loads one tensor at a time)
struct Weights {
    ggml_context* ctx = nullptr;
    gguf_context* g = nullptr;
    std::string path;
    std::vector<uint8_t> raw(const std::string& name, const ggml_tensor** tensor = nullptr) const {
        const ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
        const int64_t id = gguf_find_tensor(g, name.c_str());
        if (!t || id < 0) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
        std::vector<uint8_t> b(ggml_nbytes(t));
        std::ifstream f(path, std::ios::binary);
        f.seekg(std::streamoff(gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id)));
        if (!f.read(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()))) {
            std::fprintf(stderr, "reading tensor %s failed\n", name.c_str()); std::exit(1);
        }
        if (tensor) *tensor = t;
        return b;
    }
    std::vector<float> get(const std::string& name, int64_t* ne0 = nullptr, int64_t* ne1 = nullptr) const {
        const ggml_tensor* t = nullptr;
        const std::vector<uint8_t> b = raw(name, &t);
        const int64_t n = ggml_nelements(t);
        std::vector<float> out(static_cast<size_t>(n));
        if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), b.data(), size_t(n) * 4);
        else {
            const ggml_type_traits* tr = ggml_get_type_traits(t->type);
            if (!tr->to_float) { std::fprintf(stderr, "no conversion for %s\n", name.c_str()); std::exit(1); }
            tr->to_float(b.data(), out.data(), n);
        }
        if (ne0) *ne0 = t->ne[0];
        if (ne1) *ne1 = t->ne[1];
        return out;
    }
    bool has(const std::string& name) const { return ggml_get_tensor(ctx, name.c_str()) != nullptr; }
};

// Token embeddings as the GGUF stores them (F32, F16, BF16, Q8_0 rows); a row converted when used
struct Embeddings {
    std::vector<uint8_t> data;
    ggml_type type = GGML_TYPE_F32;
    size_t row_bytes = 0;
    int n = 0;
    void row(int token, float* out) const {
        const uint8_t* p = data.data() + size_t(token) * row_bytes;
        if (type == GGML_TYPE_F32) std::memcpy(out, p, size_t(n) * 4);
        else ggml_get_type_traits(type)->to_float(p, out, n);
    }
};

double cosine(const float* a, const float* b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
    return ab / std::sqrt(aa * bb + 1e-30);
}

int argmax(const float* v, int n) { return int(std::max_element(v, v + n) - v); }

int rank_of(const float* v, int n, int tok) {
    int r = 0;
    for (int i = 0; i < n; ++i) r += v[i] > v[tok];
    return r;
}

std::string piece(const llama_vocab* vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, size_t(n)) : std::string();
}

// Measure resident-model inference from tokenized input to sampled token IDs.
// No CPU-reference inference, file I/O, or console output runs inside a request.
// token_prefill: the prompt goes through the decode path one token at a time (lm_head for the last only), as a
// design with only a decode path would run it
int benchmark(Llm& llm, const Embeddings& embeddings, const std::vector<llama_token>& prompt,
              int n_gen, int warmup, int repeats, double core_mhz, const std::string& metrics_path,
              std::chrono::steady_clock::time_point process_start, double upload_s, bool token_prefill) {
    using Clock = std::chrono::steady_clock;
    const int nt = int(prompt.size()), E = llm.cfg().n_embd, V = llm.cfg().vocab;
    std::vector<float> xs(size_t(nt) * E), x(static_cast<size_t>(E)), logits(static_cast<size_t>(V));
    std::vector<int> tokens(static_cast<size_t>(n_gen));
    std::vector<double> ready(static_cast<size_t>(n_gen));
    std::FILE* metrics = metrics_path.empty() ? nullptr : std::fopen(metrics_path.c_str(), "w");
    if (!metrics_path.empty() && !metrics) { std::perror("open metrics"); return 1; }
    if (metrics) std::fprintf(metrics,
        "{\"schema\":1,\"core_mhz\":%.6f,\"prompt_tokens\":%d,\"generated_tokens\":%d,"
        "\"warmup\":%d,\"repetitions\":%d,\"weight_upload_s\":%.6f,\"prefill\":\"%s\","
        "\"scope\":\"tokenized input through host preparation, FPGA execution, logits readback and greedy argmax; weights resident; no CPU reference, profiling or timed I/O\",\"runs\":[\n",
        core_mhz, nt, n_gen, warmup, repeats, upload_s, token_prefill ? "token by token" : "tiles");
    std::printf("benchmark: %d prompt tokens%s, %d generated tokens, %d warmup, %d measured requests\n",
                nt, token_prefill ? " (prefill token by token)" : "", n_gen, warmup, repeats);
    for (int run = 0; run < warmup + repeats; ++run) {
        llm.reset_host_timing();
        const auto start = Clock::now();
        for (int k = 0; k < nt; ++k) embeddings.row(prompt[size_t(k)], &xs[size_t(k) * E]);
        // pos=0 starts a fresh logical KV sequence; subsequent positions only see its prefix.
        if (token_prefill)
            for (int k = 0; k < nt; ++k) llm.decode(&xs[size_t(k) * E], k, k + 1 == nt ? logits.data() : nullptr);
        else llm.prefill(xs.data(), nt, 0, logits.data());
        tokens[0] = argmax(logits.data(), V);
        const auto first = Clock::now();
        ready[0] = std::chrono::duration<double, std::milli>(first - start).count();
        for (int k = 1; k < n_gen; ++k) {
            embeddings.row(tokens[size_t(k - 1)], x.data());
            llm.decode(x.data(), nt + k - 1, logits.data());
            tokens[size_t(k)] = argmax(logits.data(), V);
            ready[size_t(k)] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        const double tpot = n_gen > 1 ? (ready.back() - ready.front()) / (n_gen - 1) : 0.0;
        std::printf("benchmark %s %d: TTFT %.3f ms, ", run < warmup ? "warmup" : "run", run < warmup ? run : run - warmup, ready[0]);
        if (n_gen > 1) std::printf("TPOT %.3f ms/token (%.2f token/s), ", tpot, 1000.0 / tpot);
        else std::printf("TPOT n/a (one generated token), ");
        std::printf("request %.3f ms\n", ready.back());
        if (metrics) {
            // start_s: the steady clock (CLOCK_MONOTONIC on Linux), to place the request in a power trace
            std::fprintf(metrics, "%s{\"run\":%d,\"warmup\":%s,\"start_s\":%.6f,\"ttft_ms\":%.6f,\"tpot_ms\":",
                         run ? ",\n" : "", run, run < warmup ? "true" : "false",
                         std::chrono::duration<double>(start.time_since_epoch()).count(), ready[0]);
            if (n_gen > 1) std::fprintf(metrics, "%.6f", tpot); else std::fputs("null", metrics);
            std::fprintf(metrics, ",\"request_ms\":%.6f,\"cold_start_to_first_token_ms\":", ready.back());
            if (run == 0) std::fprintf(metrics, "%.6f", std::chrono::duration<double, std::milli>(first - process_start).count());
            else std::fputs("null", metrics);
            std::fputs(",\"token_ready_ms\":[", metrics);
            for (int k = 0; k < n_gen; ++k) std::fprintf(metrics, "%s%.6f", k ? "," : "", ready[size_t(k)]);
            std::fputs("],\"token_ids\":[", metrics);
            for (int k = 0; k < n_gen; ++k) std::fprintf(metrics, "%s%d", k ? "," : "", tokens[size_t(k)]);
            const auto& ht = llm.host_timing();
            std::fprintf(metrics, "],\"host_stage_ms\":{\"input\":%.6f,\"build\":%.6f,\"submit\":%.6f,\"sync\":%.6f,\"read\":%.6f},\"tiles\":%llu,\"commands\":%llu}",
                         ht.input_ms, ht.build_ms, ht.submit_ms, ht.sync_ms, ht.read_ms,
                         (unsigned long long)ht.tiles, (unsigned long long)ht.commands);
        }
    }
    if (metrics) { std::fputs("\n]}\n", metrics); if (std::fclose(metrics) != 0) return 1; }
    return 0;
}

// llama-perplexity's measure with the Node's logits: the text tokenized whole (special tokens added, none parsed), in
// chunks of n_ctx tokens (the first a BOS when the vocabulary adds one), each a fresh sequence whose second half
// [n_ctx/2, n_ctx - 1) is scored on the next token. The prefill runs a 32-token band at a time with the logits of
// every scored row (prefill_rows).
int perplexity(Llm& llm, const Embeddings& embeddings, const llama_vocab* vocab, const std::string& path, int n_ctx,
               int max_chunks) {
    std::ifstream f(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!f.good() && !f.eof()) { std::fprintf(stderr, "reading %s failed\n", path.c_str()); return 1; }
    const int n_tok = -llama_tokenize(vocab, text.c_str(), int32_t(text.size()), nullptr, 0, true, false);
    std::vector<llama_token> toks(static_cast<size_t>(std::max(n_tok, 0)));
    if (n_tok <= 0 || llama_tokenize(vocab, text.c_str(), int32_t(text.size()), toks.data(), n_tok, true, false) != n_tok) {
        std::fprintf(stderr, "tokenizing %s failed\n", path.c_str()); return 1;
    }
    uint64_t digest = 1469598103934665603ull;            // FNV-1a of the token ids, to match runs' tokenizations
    for (llama_token t : toks) for (int b = 0; b < 4; ++b) digest = (digest ^ ((uint32_t(t) >> (8 * b)) & 0xFF)) * 1099511628211ull;
    const int n_all = n_tok / n_ctx, n_chunk = max_chunks > 0 ? std::min(max_chunks, n_all) : n_all;
    const int first = n_ctx / 2, E = llm.cfg().n_embd, V = llm.cfg().vocab;
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::printf("perplexity: %d tokens (FNV-1a %016llx), %d of %d chunks of %d, scored [%d, %d), BOS %s\n", n_tok,
                (unsigned long long)digest, n_chunk, n_all, n_ctx, first, n_ctx - 1, add_bos ? "yes" : "no");
    if (n_chunk < 1) { std::fprintf(stderr, "fewer than n_ctx tokens\n"); return 2; }
    std::vector<float> xs(size_t(32) * E), logits(size_t(32) * V);
    double nll = 0, nll2 = 0;
    long count = 0, top1 = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < n_chunk; ++c) {
        const size_t start = size_t(c) * n_ctx;
        for (int i = 0; i < n_ctx; i += 32) {
            const int n = std::min(32, n_ctx - i);
            for (int k = 0; k < n; ++k) {
                const llama_token t = add_bos && i + k == 0 ? llama_vocab_bos(vocab) : toks[start + i + k];
                embeddings.row(t, &xs[size_t(k) * E]);
            }
            std::vector<int> rows;
            for (int k = 0; k < n; ++k) if (i + k >= first && i + k < n_ctx - 1) rows.push_back(k);
            llm.prefill_rows(xs.data(), n, i, rows, logits.data());
            for (size_t r = 0; r < rows.size(); ++r) {
                const float* l = &logits[r * V];
                const int target = toks[start + i + rows[r] + 1];
                const float mx = *std::max_element(l, l + V);
                double sum = 0;
                for (int v = 0; v < V; ++v) sum += std::exp(l[v] - mx);
                const double e = std::log(sum) - (double(l[target]) - mx);
                nll += e; nll2 += e * e; ++count;
                top1 += argmax(l, V) == target;
            }
        }
        std::printf("[%d]%.4f%s", c + 1, std::exp(nll / count), (c + 1) % 8 ? "," : "\n");
        std::fflush(stdout);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double mean = nll / count, var = nll2 / count - mean * mean, ppl = std::exp(mean);
    std::printf("\nFinal estimate: PPL = %.4f +/- %.5f (mean NLL %.6f over %ld tokens, top-1 %.4f), %.1f s, %.2f s/chunk\n",
                ppl, var > 0 ? std::sqrt(var / (count - 1)) * ppl : 0.0, mean, count, double(top1) / count, s, s / n_chunk);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto process_start = std::chrono::steady_clock::now();
    std::string model_path, prompt = "The capital of France is";
    int n_gen = 16, ctx = 512, layers = -1;
    bool trace = false, use_prefill = false, board = false, profile = false;
    double core_mhz = 0;
    bool bench = false, host_timing = false, test_neox = false, no_reference = false, token_prefill = false;
    int max_group = 4, max_kv_heads = 4, max_ffn_columns = 4064;
    int repeats = 5, warmup = 1, prompt_tokens = 0;
    std::string metrics_path, weights_cache, model_sha;
    std::string dump_path, compare_path, profile_dump, ppl_path;
    int ppl_chunks = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (a == "-n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) layers = std::atoi(argv[++i]);
        else if (a == "--benchmark") bench = true;
        else if (a == "--token-prefill") token_prefill = true;     // benchmark: the prompt through the decode path
        else if (a == "--host-timing") host_timing = true;
        else if (a == "--weights-cache" && i + 1 < argc) weights_cache = argv[++i];
        else if (a == "--repeat" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--prompt-tokens" && i + 1 < argc) prompt_tokens = std::atoi(argv[++i]);
        else if (a == "--metrics" && i + 1 < argc) metrics_path = argv[++i];
        else if (a == "--perplexity" && i + 1 < argc) ppl_path = argv[++i];
        else if (a == "--chunks" && i + 1 < argc) ppl_chunks = std::atoi(argv[++i]);
        else if (a == "--trace") trace = true;
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--board") board = true;
        else if (a == "--profile") profile = true;                             // Node cycles per command role
        else if (a == "--profile-dump" && i + 1 < argc) { profile = true; profile_dump = argv[++i]; }  // per command
        else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];        // per step: pos, token, logits
        else if (a == "--compare" && i + 1 < argc) compare_path = argv[++i];  // follow a dump's tokens, compare
        else if (a == "--max-group" && i + 1 < argc) max_group = std::atoi(argv[++i]);
        else if (a == "--max-kv-heads" && i + 1 < argc) max_kv_heads = std::atoi(argv[++i]);
        else if (a == "--max-ffn-columns" && i + 1 < argc) max_ffn_columns = std::atoi(argv[++i]);
        else if (a == "--test-neox") test_neox = true;
        else if (a == "--no-reference") no_reference = true;
        else if (a == "--model-sha256" && i + 1 < argc) model_sha = argv[++i];
        else { std::fprintf(stderr, "usage: %s -m model.gguf [-p prompt] [-n N] [--ctx C] [--layers L]\n", argv[0]); return 2; }
    }
    if (bench && (n_gen < 1 || repeats < 1 || warmup < 0 || prompt_tokens < 0 ||
                  trace || profile || !dump_path.empty() || !compare_path.empty())) {
        std::fprintf(stderr, "benchmark requires N>=1, repeat>=1, warmup>=0 and excludes trace/profile/dump/compare\n"); return 2;
    }
    if (!bench && (prompt_tokens || !metrics_path.empty())) {
        std::fprintf(stderr, "--prompt-tokens and --metrics require --benchmark\n"); return 2;
    }
    if (model_path.empty()) { std::fprintf(stderr, "-m model.gguf required\n"); return 2; }
    if (!ppl_path.empty() && (bench || trace || profile || !dump_path.empty() || !compare_path.empty() || ctx % 32)) {
        std::fprintf(stderr, "--perplexity excludes benchmark/trace/profile/dump/compare; --ctx a multiple of 32\n"); return 2;
    }
    if (!ppl_path.empty()) no_reference = true;

    // llama.cpp on the CPU: tokenizer and reference logits
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
    }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = bench || no_reference;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(ctx);
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* lctx = bench || no_reference ? nullptr : llama_init_from_model(model, cp);
    std::vector<llama_token> toks(prompt.size() + 8);
    int nt = llama_tokenize(vocab, prompt.c_str(), int32_t(prompt.size()), toks.data(), int32_t(toks.size()), true, true);
    if (nt <= 0) { std::fprintf(stderr, "prompt tokenization failed\n"); return 2; }
    toks.resize(static_cast<size_t>(nt));
    if (bench && prompt_tokens) {
        const auto seed = toks;
        nt = prompt_tokens; toks.resize(static_cast<size_t>(nt));
        for (int k = 0; k < nt; ++k) toks[size_t(k)] = seed[size_t(k) % seed.size()];
        std::printf("benchmark input: cyclic repetition of tokenized prompt, %d tokens\n", nt);
    }
    if (bench && (ctx < 1 || int64_t(nt) + n_gen - 1 > ctx)) {
        std::fprintf(stderr, "benchmark prompt + generated tokens - 1 exceeds context\n"); return 2;
    }

    // weights from the gguf file
    Weights W;
    gguf_init_params gp{true, &W.ctx};
    W.g = gguf_init_from_file(model_path.c_str(), gp);
    if (!W.g) return 1;
    W.path = model_path;
    LlmConfig cfg;
    cfg.ctx = (ctx + 31) / 32 * 32;
    cfg.max_group = max_group; cfg.max_kv_heads = max_kv_heads; cfg.max_ffn_columns = max_ffn_columns;
    int64_t ne0 = 0, ne1 = 0;
    Embeddings emb;
    {
        const ggml_tensor* t = nullptr;
        emb.data = W.raw("token_embd.weight", &t);
        if (t->type != GGML_TYPE_F32 && !ggml_get_type_traits(t->type)->to_float)
            throw std::runtime_error("token embeddings of an unconvertible type");
        emb.type = t->type; emb.row_bytes = ggml_row_size(t->type, t->ne[0]); emb.n = int(t->ne[0]);
        ne0 = t->ne[0]; ne1 = t->ne[1];
    }
    cfg.n_embd = int(ne0); cfg.vocab = int(ne1);
    auto integer = [&](const std::string& key) {
        const int64_t index=gguf_find_key(W.g,key.c_str());
        if (index<0 || gguf_get_kv_type(W.g,index)!=GGUF_TYPE_UINT32)
            throw std::runtime_error("missing/invalid GGUF metadata: "+key);
        return int(gguf_get_val_u32(W.g,index));
    };
    // llama (SmolLM, SmolVLM2's text model), qwen2 (NEOX RoPE; its q / k / v biases must be zero until the Node
    // has a bias epilogue) or qwen3 (NEOX RoPE, q / k norm)
    const int64_t arch_key=gguf_find_key(W.g,"general.architecture");
    const std::string arch=arch_key<0 ? "" : gguf_get_val_str(W.g,arch_key);
    if (arch!="llama" && arch!="qwen2" && arch!="qwen3")
        throw std::runtime_error("this runtime supports the llama, qwen2 and qwen3 architectures only");
    cfg.rope_neox=arch=="qwen2" || arch=="qwen3";
    cfg.qk_norm=arch=="qwen3";
    cfg.n_head=integer(arch+".attention.head_count");
    cfg.n_kv=integer(arch+".attention.head_count_kv");
    cfg.n_ff=integer(arch+".feed_forward_length");
    // dh from W_q, or (a GGUF with the embeddings only) from the metadata
    const ggml_tensor* q=ggml_get_tensor(W.ctx,"blk.0.attn_q.weight");
    const int64_t key_length=gguf_find_key(W.g,(arch+".attention.key_length").c_str());
    if (cfg.n_head<1 || (q && q->ne[1]%cfg.n_head)) throw std::runtime_error("invalid Q/head dimensions");
    cfg.dh=q ? int(q->ne[1])/cfg.n_head : key_length>=0 ? integer(arch+".attention.key_length") : cfg.n_embd/cfg.n_head;
    const int64_t rope=gguf_find_key(W.g,(arch+".rope.freq_base").c_str());
    const int64_t eps=gguf_find_key(W.g,(arch+".attention.layer_norm_rms_epsilon").c_str());
    if (rope>=0) cfg.rope_base=gguf_get_val_f32(W.g,rope);
    if (eps>=0) { const float e=gguf_get_val_f32(W.g,eps); cfg.eps_profile=e<=0?0:e<3e-6f?1:e<3e-5f?2:3; cfg.qk_eps=e; }
    if (test_neox && cfg.rope_neox) throw std::runtime_error("--test-neox needs a model with adjacent RoPE pairs");
    if (test_neox) cfg.rope_neox = true;

    cfg.n_layer = 0;
    while (W.has("blk." + std::to_string(cfg.n_layer) + ".attn_q.weight")) ++cfg.n_layer;
    if (!cfg.n_layer) cfg.n_layer = integer(arch + ".block_count");
    if (layers > 0) cfg.n_layer = std::min(cfg.n_layer, layers);
    for (int l = 0; l < cfg.n_layer; ++l)
        for (const char* t : {"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias"}) {
            const std::string name = "blk." + std::to_string(l) + "." + t;
            if (!W.has(name)) continue;
            const std::vector<float> b = W.get(name);
            if (std::any_of(b.begin(), b.end(), [](float v) { return v != 0.0f; }))
                throw std::runtime_error(name + " is not zero: the Node program has no bias yet");
        }
    std::printf("model: %d layers, n_embd %d, heads %d/%d x %d, ff %d, vocab %d, RoPE %s; prompt %d tokens\n",
                cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv, cfg.dh, cfg.n_ff, cfg.vocab,
                cfg.rope_neox ? (test_neox ? "NEOX (test)" : "NEOX") : "adjacent", nt);

    g3::NodeOptions nopt;
    nopt.trace = trace;
    std::unique_ptr<Device> devp;
#ifdef G3RT_BOARD
    if (board) {
        auto* bd = new BoardDevice();
        core_mhz = bd->core_mhz();
        std::printf("board: core clock %.1f MHz\n", core_mhz);
        devp.reset(bd);
    }
#else
    if (board) { std::fprintf(stderr, "built without BOARD=1\n"); return 2; }
#endif
    if (!devp) devp.reset(new EmuDevice(nopt));
    Device& dev = *devp;
    std::printf("device: %s\n", dev.name().c_str());
    Llm llm(dev, cfg);
    std::printf("UB map: %d-token prefill tiles\n", llm.tile());
    llm.set_host_timing(host_timing);
    Profile prof;
    if (profile) llm.set_profile(&prof);
    if (!profile_dump.empty() && !prof.open_dump(profile_dump)) { std::perror("open profile dump"); return 1; }
    // a dump (on the board) and its comparison (on the PC, with the emulator): records {int32 pos, int32 token,
    // float logits[vocab]} of every step the Node ran
    std::FILE* dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");
    std::FILE* cmpf = compare_path.empty() ? nullptr : std::fopen(compare_path.c_str(), "rb");
    std::vector<float> ref_logits;
    int cmp_equal = 0, cmp_steps = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::string model_digest;
    const bool cache_hit = !weights_cache.empty() && std::filesystem::exists(weights_cache);
    try {
    if (!weights_cache.empty()) model_digest = model_sha.empty() ? sha256_file(model_path) : model_sha;
    if (cache_hit) llm.load_weights(weights_cache, model_digest);
    else {
    for (int l = 0; l < cfg.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        std::vector<float> wq = W.get(b + "attn_q.weight"), wk = W.get(b + "attn_k.weight"), wv = W.get(b + "attn_v.weight");
        std::vector<float> wo = W.get(b + "attn_output.weight"), wg = W.get(b + "ffn_gate.weight");
        std::vector<float> wu = W.get(b + "ffn_up.weight"), wd = W.get(b + "ffn_down.weight");
        std::vector<float> ga = W.get(b + "attn_norm.weight"), gf = W.get(b + "ffn_norm.weight");
        std::vector<float> gq, gk;
        if (cfg.qk_norm) { gq = W.get(b + "attn_q_norm.weight"); gk = W.get(b + "attn_k_norm.weight"); }
        if (test_neox)          // row d of a head -> the row the NEOX arrangement takes it back from
            for (std::vector<float>* m : {&wq, &wk}) {
                const std::vector<float> w0 = *m;
                const int D = cfg.dh, E = cfg.n_embd;
                for (int r = 0; r < int(w0.size()) / E; ++r) {
                    const int d = r % D, to = r / D * D + (d & 1 ? d / 2 + D / 2 : d / 2);
                    std::memcpy(&(*m)[size_t(to) * E], &w0[size_t(r) * E], size_t(E) * 4);
                }
            }
        LayerWeights lw{wq.data(), wk.data(), wv.data(), wo.data(), wg.data(), wu.data(), wd.data(), ga.data(), gf.data(),
                        gq.data(), gk.data()};
        llm.upload_layer(l, lw);
    }
    std::vector<float> g_out = W.get("output_norm.weight");
    // a tied output (Qwen2.5-0.5B) is the embedding matrix itself
    const std::vector<float> lm = W.get(W.has("output.weight") ? "output.weight" : "token_embd.weight");
    llm.upload_output(g_out.data(), lm.data());
    if (!weights_cache.empty()) llm.save_weights(weights_cache, model_digest);
    }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "weight preparation failed: %s\n", e.what()); return 1;
    }
    if (!weights_cache.empty()) std::printf("weight cache: %s\n", cache_hit ? "hit (verified)" : "created");
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("weights converted and uploaded: %.1f MB of device memory, %.1f s\n", double(llm.device_bytes()) / 1e6,
                std::chrono::duration<double>(t1 - t0).count());

    if (!ppl_path.empty()) {
        const int status = perplexity(llm, emb, vocab, ppl_path, ctx, ppl_chunks);
        llama_model_free(model); gguf_free(W.g); ggml_free(W.ctx);
        return status;
    }
    if (bench) {
        const int status = benchmark(llm, emb, toks, n_gen, warmup, repeats, core_mhz, metrics_path,
                                     process_start, std::chrono::duration<double>(t1 - t0).count(), token_prefill);
        llama_model_free(model); gguf_free(W.g); ggml_free(W.ctx);
        return status;
    }

    const int V = cfg.vocab, E = cfg.n_embd;
    std::vector<float> node_logits(static_cast<size_t>(V)), x(static_cast<size_t>(E));
    std::string text_node;
    int pos = 0, agree = 0, steps = 0;
    double cos_min = 1.0, node_s = 0;
    llama_token next = 0;
    const int total = nt + n_gen;
    for (int i = 0; i < total; ++i) {
        const llama_token t = i < nt ? toks[size_t(i)] : next;
        // CPU reference
        const float* ref = nullptr;
        if (lctx) {
            llama_token tt = t;
            if (llama_decode(lctx, llama_batch_get_one(&tt, 1)) != 0) { std::fprintf(stderr, "llama_decode failed\n"); return 1; }
            ref = llama_get_logits_ith(lctx, -1);
        }
        // Node: the prompt in one prefill (tiles of up to 8 tokens), or every token as a decode step
        if (use_prefill && i < nt - 1) { ++pos; continue; }
        const auto a = std::chrono::steady_clock::now();
        if (use_prefill && i == nt - 1) {
            std::vector<float> xs(static_cast<size_t>(nt) * E);
            for (int k = 0; k < nt; ++k) emb.row(toks[size_t(k)], &xs[size_t(k) * E]);
            llm.prefill(xs.data(), nt, 0, node_logits.data());
        } else {
            emb.row(t, x.data());
            llm.decode(x.data(), pos, node_logits.data());
        }
        if (dump) {
            const int32_t hdr[2] = {pos, t};
            std::fwrite(hdr, 4, 2, dump);
            std::fwrite(node_logits.data(), 4, size_t(V), dump);
        }
        int32_t cmp_next = -1;
        if (cmpf) {
            int32_t hdr[2];
            ref_logits.resize(size_t(V));
            if (std::fread(hdr, 4, 2, cmpf) == 2 && std::fread(ref_logits.data(), 4, size_t(V), cmpf) == size_t(V)) {
                if (hdr[0] != pos || hdr[1] != t) std::printf("  dump step (%d, %d) for (%d, %d)\n", hdr[0], hdr[1], pos, t);
                int nd = 0;
                for (int k = 0; k < V; ++k) nd += std::memcmp(&ref_logits[size_t(k)], &node_logits[size_t(k)], 4) != 0;
                cmp_equal += nd == 0;
                ++cmp_steps;
                if (nd) std::printf("  dump != this run at pos %d: %d of %d logits differ\n", pos, nd, V);
                cmp_next = argmax(ref_logits.data(), V);
            }
        }
        node_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        const int top_node = argmax(node_logits.data(), V);
        ++steps;
        if (ref) {
            const double c = layers > 0 ? 0.0 : cosine(ref, node_logits.data(), V);
            const int top_ref = argmax(ref, V);
            cos_min = std::min(cos_min, c);
            agree += top_ref == top_node;
            std::printf("pos %3d token %6d %-12s  cos %.5f  top-1 cpu %6d node %6d %s  rank of cpu top-1 %d\n", pos, t,
                        ("'" + piece(vocab, t) + "'").c_str(), c, top_ref, top_node, top_ref == top_node ? "=" : "x",
                        rank_of(node_logits.data(), V, top_ref));
        } else std::printf("pos %3d token %6d %-12s  top-1 node %6d\n", pos, t, ("'" + piece(vocab, t) + "'").c_str(), top_node);
        ++pos;
        if (i >= nt - 1) {
            next = cmp_next >= 0 ? cmp_next : top_node;       // a comparison follows the dump's tokens
            if (i >= nt - 1 && i < total - 1) text_node += piece(vocab, next);
        }
    }
    std::printf("\nprompt: %s\nnode continuation: %s\n", prompt.c_str(), text_node.c_str());
    if (lctx)
        std::printf("steps %d, top-1 agreement %d/%d, min cosine %.5f, %s %.4f s/step\n", steps, agree, steps, cos_min,
                    dev.name().c_str(), node_s / steps);
    else std::printf("steps %d (no CPU reference), %s %.4f s/step\n", steps, dev.name().c_str(), node_s / steps);
    if (profile) prof.print(stdout, core_mhz);
    if (cmpf) std::printf("logits bit-identical to the dump's: %d/%d steps\n", cmp_equal, cmp_steps);
    if (dump) std::fclose(dump);
    if (cmpf) std::fclose(cmpf);
    if (lctx) llama_free(lctx);
    llama_model_free(model);
    gguf_free(W.g);
    ggml_free(W.ctx);
    return 0;
}
