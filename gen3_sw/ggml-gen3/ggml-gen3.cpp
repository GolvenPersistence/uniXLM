// ggml-gen3: GEN3 matmul and opt-in attention offload with CPU fallback; GGML_GEN3_MODE=fused selects the legacy whole-model path.
//
// Device and buffers
//   One device ("GEN3", GPU type, so llama.cpp offloads layers to it). Buffers are host memory for the tensors the
//   graph exchanges with the CPU (its inputs: inp_embd, positions, KV indices, mask; its outputs: logits); weights
//   never land there: set_tensor converts a weight as it arrives (all of it, or chunk by chunk into a staging copy)
//   into an MXFP8 P32 image in device memory (gen3rt), recorded by name. The KV cache tensors stand for the Node's
//   K / V^T images (gen3rt::Llm); their contents are the backend's, get_tensor of them is not supported.
// Graphs
//   graph_compute recognises the llama graph of SmolVLM2's text model (RMS_NORM attn_norm-0 .. MUL_MAT
//   result_output, the names llama.cpp's callbacks give), reads inp_embd, the positions (ROPE), the KV row indices
//   (SET_ROWS: they must equal the positions, one sequence), the output rows (inp_out_ids), freq_base and the
//   pairing (adjacent or NEOX: Qwen2) from the ROPE op params, and eps (RMS_NORM), and runs the whole model through
//   gen3rt::Llm (tiles of up to 8 tokens; decode steps append the KV cache on the Node). q / k / v biases (Qwen2)
//   must be zero. The logits go into result_output. Other graphs are refused.
// Device selection: GGML_GEN3_DEVICE=emu (default; the bit-exact emulator) or board (PYNQ-ZU, built with G3RT_BOARD).
#include "ggml-gen3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "gen3rt/device.h"
#include "gen3rt/llm.h"
#include "gen3rt/matmul.h"
#include "gen3rt/attention.h"
#include "ggml-cpu.h"
#include "gen3rt/profile.h"
#include "gen3rt/vit.h"
#ifdef G3RT_BOARD
#include "gen3rt/board.h"
#endif

namespace {

// The legacy fused path is opt-in; normal scheduling uses explicit operator support.
bool hybrid_mode() {
    const char* mode = std::getenv("GGML_GEN3_MODE");
    return !mode || std::string(mode) != "fused";
}

bool attention_supports(const ggml_tensor* op) {
    const char* flag=std::getenv("GGML_GEN3_ATTN");
    if(!flag || std::string(flag)!="1" || op->op!=GGML_OP_FLASH_ATTN_EXT)return false;
    const auto* q=op->src[0];const auto* k=op->src[1];const auto* v=op->src[2];const auto* mask=op->src[3];
    if(!q || !k || !v || !mask || op->src[4])return false;
    float params[3];std::memcpy(params,op->op_params,sizeof(params));
    return q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F16 && v->type==GGML_TYPE_F16 &&
        mask->type==GGML_TYPE_F16 && q->nb[0]==4 && k->nb[0]==2 && (v->nb[0]==2 || v->nb[1]==2) &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) && op->type==GGML_TYPE_F32 &&
        q->ne[3]==1 && k->ne[3]==1 && v->ne[3]==1 && mask->ne[2]==1 && mask->ne[3]==1 &&
        q->ne[0]>=32 && q->ne[0]<=128 && q->ne[0]%32==0 && q->ne[0]==k->ne[0] && q->ne[0]==v->ne[0] &&
        q->ne[2]>=1 && q->ne[2]<=16 && k->ne[2]>=1 && k->ne[2]<=4 && q->ne[2]%k->ne[2]==0 && k->ne[2]==v->ne[2] &&
        k->ne[1]>=1 && k->ne[1]<=2048 && k->ne[1]==v->ne[1] && mask->ne[0]==k->ne[1] && mask->ne[1]>=q->ne[1] &&
        q->ne[1]>=1 && q->ne[1]<=2048 && params[0]==1.0f/std::sqrt(float(q->ne[0])) && params[1]==0 && params[2]==0;
}

bool ffn_glu(const ggml_tensor* t) {
    const char* flag=std::getenv("GGML_GEN3_FFN");
    return flag && (std::string(flag)=="1" || std::string(flag)=="exact" || std::string(flag)=="quant") && t->op==GGML_OP_GLU && ggml_get_glu_op(t)==GGML_GLU_OP_SWIGLU && t->op_params[1]==0 &&
        t->src[0] && t->src[1] && t->type==GGML_TYPE_F32 && t->src[0]->type==GGML_TYPE_F32 && t->src[1]->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(t) && ggml_is_contiguous(t->src[0]) && ggml_is_contiguous(t->src[1]);
}

bool hybrid_supports(const ggml_tensor* op) {
    if(ffn_glu(op)) return true;
    if (attention_supports(op)) return true;
    switch (op->op) {
    case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: return true;
    case GGML_OP_MUL_MAT: break;
    default: return false;
    }
    const auto* w = op->src[0]; const auto* x = op->src[1];
    if (!w || !x) return false;
    const char* head_flag=std::getenv("GGML_GEN3_VOCAB");
    const bool head=head_flag && std::string(head_flag)=="1" &&
        (!std::strcmp(w->name,"output.weight") || !std::strcmp(w->name,"token_embd.weight"));
    if (!head && (std::strncmp(w->name,"blk.",4) || !std::strstr(w->name,".weight"))) return false;
    const auto* traits = ggml_get_type_traits(w->type);
    return (w->type == GGML_TYPE_F32 || traits->to_float) && x->type == GGML_TYPE_F32 &&
        op->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_is_contiguous(x) &&
        ggml_is_contiguous(op) && w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
        w->ne[0] >= 32 && w->ne[0] <= g3rt::Matmul::max_k && w->ne[0] % 32 == 0 &&
        w->ne[1] >= 64 && w->ne[1] <= (head ? g3rt::Matmul::max_n : 16384) && w->ne[1] % 64 == 0 && x->ne[0] == w->ne[0];
}

struct TensorMeta {
    int layer = -2;            // -1: output tensors, >= 0: blk.N, -2: not a weight we take
    std::string role;
    int rows = 0, cols = 0;
};

// parse "blk.N.role.weight", "output_norm.weight", "output.weight"
TensorMeta parse_name(const ggml_tensor* t) {
    TensorMeta m;
    const std::string n = t->name;
    m.cols = int(t->ne[0]);
    m.rows = int(t->ne[1]);
    if (n == "output_norm.weight") { m.layer = -1; m.role = "output_norm"; return m; }
    // A tied output is duplicated into GEN3 under the embedding tensor name.
    if (n == "output.weight" || n == "token_embd.weight") { m.layer = -1; m.role = "output"; return m; }
    if (n.rfind("blk.", 0) == 0) {
        const size_t d1 = n.find('.', 4), d2 = n.rfind(".weight");
        if (d1 != std::string::npos && d2 != std::string::npos && d2 > d1) {
            m.layer = std::atoi(n.substr(4, d1 - 4).c_str());
            m.role = n.substr(d1 + 1, d2 - d1 - 1);
        }
    }
    return m;
}

bool is_kv_cache(const ggml_tensor* t) {
    return std::strncmp(t->name, "cache_k_l", 9) == 0 || std::strncmp(t->name, "cache_v_l", 9) == 0;
}

struct State {
    std::mutex mu;
    std::unique_ptr<g3rt::Device> dev;
    std::unique_ptr<g3rt::Llm> llm;
    std::unique_ptr<g3rt::Matmul> matmul;
    std::unique_ptr<g3rt::Attention> attention;
    ggml_backend_t attention_cpu=nullptr;
    uint64_t attention_mask_fallbacks=0;
    std::map<const ggml_tensor*, g3rt::Matmul::Weight> hybrid_weights;
    std::map<std::tuple<const ggml_tensor*,const ggml_tensor*,const ggml_tensor*>, g3rt::Matmul::FfnWeight> ffn_weights;
    std::map<std::string, g3rt::WeightImage> images;            // name -> image
    std::map<std::string, TensorMeta> metas;
    std::map<const ggml_tensor*, std::vector<uint8_t>> partial;  // chunked set_tensor
    std::map<const ggml_tensor*, size_t> received;
    struct GateUp { int rows=0, cols=0; std::vector<float> gate, up; };
    std::map<int, GateUp> pending_gate_up;
    struct ExactFfn {g3rt::Matmul::Weight gu,down;};
    std::map<std::tuple<const ggml_tensor*,const ggml_tensor*,const ggml_tensor*>,ExactFfn> exact_ffn_weights;
    uint64_t exact_ffn_operations=0;
    ggml_backend_t exact_cpu=nullptr;
    struct ggml_threadpool* exact_pool=nullptr;
    int max_layer = -1;
    int kv_ctx = 0;
    bool verbose = false;
    std::unique_ptr<g3rt::Profile> profile;                     // GGML_GEN3_PROFILE: Node cycles per command role
    double mhz = 0;
    ~State() {
        if(attention) GGML_LOG_INFO("GEN3_ATTN_SUMMARY ops=%llu commands=%llu submissions=%llu uploaded_bands=%llu reused_bands=%llu mask_fallbacks=%llu\n",
            (unsigned long long)attention->operations,(unsigned long long)attention->commands,(unsigned long long)attention->submissions,
            (unsigned long long)attention->uploaded_bands,(unsigned long long)attention->reused_bands,(unsigned long long)attention_mask_fallbacks);
        if(attention)GGML_LOG_INFO("GEN3_ATTN_F16_SUMMARY checked_values=%llu converted_values=%llu prefix_hits=%llu k_quantized_values=%llu\n",
            (unsigned long long)attention->f16_checked_values,(unsigned long long)attention->f16_converted_values,(unsigned long long)attention->f16_prefix_hits,(unsigned long long)attention->k_quantized_values);
        if(attention_cpu)ggml_backend_free(attention_cpu);
        if(exact_cpu)ggml_backend_free(exact_cpu);
        if(exact_pool)ggml_threadpool_free(exact_pool);
        if (matmul) GGML_LOG_INFO("GEN3_HYBRID_SUMMARY ops=%llu tiles=%llu streamed_bytes=%llu cache_bytes=%zu submissions=%llu input_reuse_hits=%llu\n",
            (unsigned long long)matmul->operations, (unsigned long long)matmul->tiles,
            (unsigned long long)matmul->streamed_bytes, matmul->peak_cache_bytes,
            (unsigned long long)matmul->submissions, (unsigned long long)matmul->input_reuse_hits);
        if(matmul) GGML_LOG_INFO("GEN3_FFN_EXACT_SUMMARY ops=%llu\n",(unsigned long long)exact_ffn_operations);
        if(matmul) GGML_LOG_INFO("GEN3_ONLINE_QUANT_SUMMARY ops=%llu\n",(unsigned long long)matmul->online_quant_operations);
        if(matmul) GGML_LOG_INFO("GEN3_FFN_SUMMARY ops=%llu\n",(unsigned long long)matmul->ffn_operations);
        if (profile) profile->print(stderr, mhz);
    }

    g3rt::Device& device() {
        if (!dev) {
            const char* e = std::getenv("GGML_GEN3_DEVICE");
            verbose = std::getenv("GGML_GEN3_VERBOSE") != nullptr;
            if (e && std::string(e) == "board") {
#ifdef G3RT_BOARD
                auto* bd = new g3rt::BoardDevice();
                dev.reset(bd);
                if (std::getenv("GGML_GEN3_PROFILE")) { profile.reset(new g3rt::Profile()); mhz = bd->core_mhz(); }
#else
                GGML_ABORT("ggml-gen3: built without G3RT_BOARD");
#endif
            } else {
                g3::NodeOptions opt;
                const char* th = std::getenv("GGML_GEN3_THREADS");
                opt.threads = th ? std::max(1, std::atoi(th)) : 8;
                dev.reset(new g3rt::EmuDevice(opt));
            }
            GGML_LOG_INFO("ggml-gen3: device %s\n", dev->name().c_str());
        }
        return *dev;
    }
};

State& state() {
    static State s;
    return s;
}

// a weight tensor's FP32 values (any type with a to_float)
std::vector<float> to_f32(const ggml_tensor* t, const void* data) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), data, size_t(n) * 4);
    else {
        const ggml_type_traits* tr = ggml_get_type_traits(t->type);
        GGML_ASSERT(tr->to_float && "ggml-gen3: weight type without to_float");
        tr->to_float(data, out.data(), n);
    }
    return out;
}

void convert_weight(State& s, const ggml_tensor* t, const void* data) {
    const TensorMeta m = parse_name(t);
    std::vector<float> f = to_f32(t, data);
    if (m.role=="ffn_gate" || m.role=="ffn_up") {
        auto& pair=s.pending_gate_up[m.layer];
        if (pair.rows && (pair.rows!=m.rows || pair.cols!=m.cols))
            GGML_ABORT("ggml-gen3: gate/up dimensions differ");
        pair.rows=m.rows; pair.cols=m.cols;
        (m.role=="ffn_gate" ? pair.gate : pair.up)=std::move(f);
        if (!pair.gate.empty() && !pair.up.empty()) {
            const auto image=g3rt::upload_gate_up_image(s.device(),pair.gate.data(),pair.up.data(),m.rows,m.cols);
            const std::string name="blk."+std::to_string(m.layer)+".ffn_gate_up.weight";
            TensorMeta combined=m; combined.role="ffn_gate_up"; combined.rows=2*m.rows;
            s.metas[name]=combined; s.images[name]=image;
            if (s.llm) s.llm->attach(m.layer,combined.role,image);
            s.pending_gate_up.erase(m.layer);
        }
        return;
    }
    g3rt::WeightImage img;
    if (m.role == "attn_norm" || m.role == "ffn_norm" || m.role == "output_norm")
        img = g3rt::upload_f32_row_image(s.device(), f.data(), m.cols);
    else img = g3rt::upload_matrix_image(s.device(), f.data(), m.rows, m.cols);
    s.images[t->name] = img;
    if (s.llm) s.llm->attach(m.layer, m.role, img);
}

// ---------------------------------------------------------------------------------------------------------------
// buffer
// ---------------------------------------------------------------------------------------------------------------
struct Buffer {
    uint8_t* raw = nullptr;     // calloc'd
    uint8_t* base = nullptr;    // 64-byte aligned (ggml-alloc aligns tensors from the base)
    size_t size = 0;
};

void buf_free(ggml_backend_buffer_t b) {
    Buffer* ctx = static_cast<Buffer*>(b->context);
    if (hybrid_mode() && ggml_backend_buffer_get_usage(b) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        State& s = state(); std::lock_guard<std::mutex> lk(s.mu);
        s.hybrid_weights.clear();s.ffn_weights.clear();s.exact_ffn_weights.clear();
        if (s.matmul) s.matmul->reset_cache();
    }
    std::free(ctx->raw);
    delete ctx;
}

void* buf_base(ggml_backend_buffer_t b) { return static_cast<Buffer*>(b->context)->base; }

ggml_status buf_init_tensor(ggml_backend_buffer_t, ggml_tensor* t) {
    if (hybrid_mode() || t->view_src != nullptr) return GGML_STATUS_SUCCESS;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    const TensorMeta m = parse_name(t);
    if (m.layer != -2) {
        s.metas[t->name] = m;
        s.max_layer = std::max(s.max_layer, m.layer);
    }
    if (std::strcmp(t->name, "cache_k_l0") == 0) s.kv_ctx = int(t->ne[1]);
    return GGML_STATUS_SUCCESS;
}

void buf_memset_tensor(ggml_backend_buffer_t b, ggml_tensor* t, uint8_t value, size_t offset, size_t size) {
    if (hybrid_mode() && ggml_backend_buffer_get_usage(b) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        State& s = state(); std::lock_guard<std::mutex> lk(s.mu);
        s.hybrid_weights.clear();s.ffn_weights.clear();s.exact_ffn_weights.clear();
        if (s.matmul) s.matmul->reset_cache();
    }
    std::memset(static_cast<uint8_t*>(t->data) + offset, value, size);
}

void buf_set_tensor(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t offset, size_t size) {
    if (hybrid_mode()) {
        if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            State& s = state(); std::lock_guard<std::mutex> lk(s.mu);
            s.hybrid_weights.clear();s.ffn_weights.clear();s.exact_ffn_weights.clear();
            if (s.matmul) s.matmul->reset_cache();
        }
        std::memcpy(static_cast<uint8_t*>(t->data) + offset, data, size);
        return;
    }
    State& s = state();
    const TensorMeta m = parse_name(t);
    if (m.layer == -2 || t->view_src != nullptr) {
        std::memcpy(static_cast<uint8_t*>(t->data) + offset, data, size);
        return;
    }
    std::lock_guard<std::mutex> lk(s.mu);
    const size_t nb = ggml_nbytes(t);
    if (offset == 0 && size == nb) {
        convert_weight(s, t, data);
        return;
    }
    std::vector<uint8_t>& st = s.partial[t];
    if (st.empty()) st.resize(nb);
    std::memcpy(st.data() + offset, data, size);
    s.received[t] += size;
    if (s.received[t] >= nb) {
        convert_weight(s, t, st.data());
        s.partial.erase(t);
        s.received.erase(t);
    }
}

void buf_get_tensor(ggml_backend_buffer_t, const ggml_tensor* t, void* data, size_t offset, size_t size) {
    if (!hybrid_mode() && parse_name(t).layer != -2 && t->view_src == nullptr)
        GGML_ABORT("ggml-gen3: reading a weight back is not supported (%s)", t->name);
    std::memcpy(data, static_cast<const uint8_t*>(t->data) + offset, size);
}

void buf_clear(ggml_backend_buffer_t b, uint8_t value) {
    Buffer* ctx = static_cast<Buffer*>(b->context);
    if (hybrid_mode()) {
        std::memset(ctx->base, value, ctx->size);
        if (ggml_backend_buffer_get_usage(b) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            State& s = state(); std::lock_guard<std::mutex> lk(s.mu);
            s.hybrid_weights.clear();s.ffn_weights.clear();s.exact_ffn_weights.clear();
            if (s.matmul) s.matmul->reset_cache();
        }
    }
    // In fused mode the KV buffers stand for Node images, overwritten by a new sequence.
}

const ggml_backend_buffer_i buffer_iface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ buf_init_tensor,
    /* .memset_tensor = */ buf_memset_tensor,
    /* .set_tensor    = */ buf_set_tensor,
    /* .get_tensor    = */ buf_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ buf_clear,
    /* .reset         = */ nullptr,
};

const char* buft_name(ggml_backend_buffer_type_t) { return "GEN3"; }

ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    Buffer* ctx = new Buffer;
    ctx->size = size;
    // Hybrid mode retains original weights here so CPU fallback can read them.
    ctx->raw = static_cast<uint8_t*>(std::calloc(1, std::max<size_t>(size, 64) + 64));
    GGML_ASSERT(ctx->raw);
    ctx->base = ctx->raw + ((64 - reinterpret_cast<uintptr_t>(ctx->raw) % 64) % 64);
    return ggml_backend_buffer_init(buft, buffer_iface, ctx, size);
}

size_t buft_alignment(ggml_backend_buffer_type_t) { return 64; }
// Hybrid buffers contain the original host-readable tensors, including CPU fallbacks.
bool buft_is_host(ggml_backend_buffer_type_t) { return hybrid_mode(); }

ggml_backend_buffer_type_i buft_iface = {
    /* .get_name       = */ buft_name,
    /* .alloc_buffer   = */ buft_alloc,
    /* .get_alignment  = */ buft_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ buft_is_host,
};

// ---------------------------------------------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------------------------------------------
const ggml_tensor* find_named(const ggml_cgraph* g, const char* name) {
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor* n = g->nodes[i];
        if (std::strcmp(n->name, name) == 0) return n;
        for (int k = 0; k < GGML_MAX_SRC; ++k)
            if (n->src[k] && std::strcmp(n->src[k]->name, name) == 0) return n->src[k];
    }
    return nullptr;
}

int eps_profile(float eps) {
    if (eps <= 0.0f) return 0;
    if (eps < 3e-6f) return 1;
    if (eps < 3e-5f) return 2;
    return 3;
}

ggml_status llm_graph(State& s, ggml_cgraph* g) {
    const ggml_tensor* norm0 = find_named(g, "norm-0");
    const ggml_tensor* q0 = find_named(g, "Qcur-0");
    const ggml_tensor* out = find_named(g, "result_output");
    if (!norm0 || norm0->op != GGML_OP_RMS_NORM || !out || out->op != GGML_OP_MUL_MAT) return GGML_STATUS_FAILED;
    // the RoPE node of layer 0 (q): positions and freq_base
    const ggml_tensor* rope = nullptr;
    for (int i = 0; i < g->n_nodes && !rope; ++i)
        if (g->nodes[i]->op == GGML_OP_ROPE) rope = g->nodes[i];
    if (!rope || !q0) return GGML_STATUS_FAILED;
    const ggml_tensor* x = norm0->src[0];
    const int n = int(x->ne[1]);
    const ggml_tensor* pos_t = rope->src[1];
    std::vector<int> pos(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) pos[size_t(i)] = static_cast<const int32_t*>(pos_t->data)[i];
    for (int i = 1; i < n; ++i)
        if (pos[size_t(i)] != pos[0] + i) { GGML_LOG_ERROR("ggml-gen3: positions are not consecutive\n"); return GGML_STATUS_FAILED; }
    // KV rows (SET_ROWS indices of layer 0's K) equal the positions
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor* t = g->nodes[i];
        if (t->op == GGML_OP_SET_ROWS && t->src[1] && std::strstr(t->name, "cache_k_l0")) {
            const ggml_tensor* idx = t->src[1];
            for (int k = 0; k < n; ++k) {
                const int64_t r = idx->type == GGML_TYPE_I64 ? static_cast<const int64_t*>(idx->data)[k]
                                                              : static_cast<const int32_t*>(idx->data)[k];
                if (r != pos[size_t(k)]) {
                    GGML_LOG_ERROR("ggml-gen3: KV row %lld for position %d (one sequence from 0 only)\n", (long long)r,
                                   pos[size_t(k)]);
                    return GGML_STATUS_FAILED;
                }
            }
            break;
        }
    }
    // output rows
    std::vector<int> rows;
    if (const ggml_tensor* ids = find_named(g, "inp_out_ids")) {
        for (int64_t k = 0; k < ids->ne[0]; ++k) rows.push_back(static_cast<const int32_t*>(ids->data)[k]);
    } else
        for (int64_t k = n - out->ne[1]; k < n; ++k) rows.push_back(int(k));      // all rows, or the last ones
    if (int64_t(rows.size()) != out->ne[1]) {
        GGML_LOG_ERROR("ggml-gen3: %zu output rows for an output of %lld\n", rows.size(), (long long)out->ne[1]);
        return GGML_STATUS_FAILED;
    }
    float freq_base;
    std::memcpy(&freq_base, (const int32_t*)rope->op_params + 5, sizeof(float));
    const int rope_mode = ((const int32_t*)rope->op_params)[2];
    if (rope_mode != 0 && rope_mode != GGML_ROPE_TYPE_NEOX) {
        GGML_LOG_ERROR("ggml-gen3: RoPE mode %d (adjacent or NEOX pairs only)\n", rope_mode);
        return GGML_STATUS_FAILED;
    }
    // Qwen2's q / k / v biases (ADD after the projection): the Node program has none yet, so they must be zero
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor* b = g->nodes[i]->op == GGML_OP_ADD ? g->nodes[i]->src[1] : nullptr;
        if (!b || std::strncmp(b->name, "blk.", 4) != 0 || !std::strstr(b->name, ".bias")) continue;
        bool zero = b->type == GGML_TYPE_F32 && b->data;
        for (int64_t k = 0; zero && k < ggml_nelements(b); ++k) zero = static_cast<const float*>(b->data)[k] == 0.0f;
        if (!zero) {
            GGML_LOG_ERROR("ggml-gen3: %s is not zero (no bias in the Node program)\n", b->name);
            return GGML_STATUS_FAILED;
        }
    }
    float eps;
    std::memcpy(&eps, norm0->op_params, sizeof(float));
    // the model on first use
    if (!s.llm) {
        g3rt::LlmConfig cfg;
        const TensorMeta& mq = s.metas.at("blk.0.attn_q.weight");
        const TensorMeta& mk = s.metas.at("blk.0.attn_k.weight");
        const TensorMeta& mg = s.metas.at("blk.0.ffn_gate.weight");
        const auto output_meta = s.metas.find(out->src[0]->name);
        if (output_meta == s.metas.end() || output_meta->second.role != "output") {
            GGML_LOG_ERROR("ggml-gen3: missing output weight metadata: %s\n", out->src[0]->name);
            return GGML_STATUS_FAILED;
        }
        const TensorMeta& mo = output_meta->second;
        cfg.dh = int(rope->src[0]->ne[0]);
        cfg.n_embd = mq.cols;
        cfg.n_head = mq.rows / cfg.dh;
        cfg.n_kv = mk.rows / cfg.dh;
        cfg.n_ff = mg.rows;
        cfg.vocab = mo.rows;
        cfg.n_layer = s.max_layer + 1;
        cfg.rope_base = freq_base;
        cfg.rope_neox = rope_mode == GGML_ROPE_TYPE_NEOX;
        cfg.eps_profile = eps_profile(eps);
        cfg.ctx = std::min(2048, (s.kv_ctx + 31) / 32 * 32);
        s.llm.reset(new g3rt::Llm(s.device(), cfg));
        s.llm->set_profile(s.profile.get());
        for (const auto& kv : s.images) {
            const TensorMeta& m = s.metas.at(kv.first);
            s.llm->attach(m.layer, m.role, kv.second);
        }
        GGML_LOG_INFO("ggml-gen3: model %d layers, n_embd %d, heads %d/%d x %d, ff %d, vocab %d, ctx %d, rope base %g, "
                      "eps profile %d\n", cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv, cfg.dh, cfg.n_ff, cfg.vocab,
                      cfg.ctx, double(cfg.rope_base), cfg.eps_profile);
    }
    if (!s.llm->complete()) { GGML_LOG_ERROR("ggml-gen3: weights missing\n"); return GGML_STATUS_FAILED; }
    if (s.verbose) GGML_LOG_INFO("ggml-gen3: %d tokens at %d, %zu output rows\n", n, pos[0], rows.size());
    s.llm->prefill_rows(static_cast<const float*>(x->data), n, pos[0], rows, static_cast<float*>(out->data));
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------
// vision encoder graph (clip.cpp build_siglip, PROJECTOR_TYPE_IDEFICS3): inp_raw -> IM2COL .. MUL_MAT(mm.model.fc)
// ---------------------------------------------------------------------------------------------------------------
const ggml_tensor* find_weight(const ggml_cgraph* g, const std::string& name) {
    for (int i = 0; i < g->n_nodes; ++i)
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor* s = g->nodes[i]->src[k];
            if (s && name == s->name) return s;
            if (s && s->view_src && name == s->view_src->name) return s->view_src;
        }
    return nullptr;
}

struct VitState {
    std::unique_ptr<g3rt::Vit> vit;
};

VitState& vit_state() {
    static VitState v;
    return v;
}

ggml_status vit_graph(State& s, ggml_cgraph* g) {
    const ggml_tensor* im2col = g->n_nodes ? g->nodes[0] : nullptr;
    if (!im2col || im2col->op != GGML_OP_IM2COL || !find_named(g, "pos_embed")) return GGML_STATUS_FAILED;
    const ggml_tensor* inp = im2col->src[1];
    const ggml_tensor* out = g->nodes[g->n_nodes - 1];
    VitState& vs = vit_state();
    auto W = [&](const std::string& n) -> std::vector<float> {
        const ggml_tensor* t = find_weight(g, n);
        if (!t) GGML_ABORT("ggml-gen3: vision weight %s not in the graph", n.c_str());
        return to_f32(t, t->data);
    };
    if (!vs.vit) {
        g3rt::VitConfig cfg;
        const ggml_tensor* pos = find_weight(g, "v.position_embd.weight");
        const ggml_tensor* fc = find_weight(g, "mm.model.fc.weight");
        const ggml_tensor* q0 = find_named(g, "Qcur_0");
        const ggml_tensor* sh = find_named(g, "pixel_shuffle");
        const ggml_tensor* n0 = find_named(g, "norm_0");
        if (!pos || !fc || !q0 || !sh || !n0) return GGML_STATUS_FAILED;
        cfg.n_embd = int(pos->ne[0]);
        cfg.n_patch = int(pos->ne[1]);
        cfg.dh = int(q0->ne[0]);
        cfg.n_head = int(q0->ne[1]);
        cfg.patch_k = int(im2col->ne[0]);
        cfg.shuffle = int(std::lround(std::sqrt(double(sh->ne[0] / cfg.n_embd))));
        cfg.proj_out = int(fc->ne[1]);
        std::memcpy(&cfg.ln_eps, n0->op_params, sizeof(float));
        cfg.n_layer = 0;
        while (find_weight(g, "v.blk." + std::to_string(cfg.n_layer) + ".attn_q.weight")) ++cfg.n_layer;
        // fc1: the MLP weight with n_ff outputs (the gguf's names of the two are swapped)
        const ggml_tensor* d0 = find_weight(g, "v.blk.0.ffn_down.weight");
        const bool swapped = d0 && d0->ne[1] > d0->ne[0];
        cfg.n_ff = int(swapped ? d0->ne[1] : find_weight(g, "v.blk.0.ffn_up.weight")->ne[1]);
        GGML_LOG_INFO("ggml-gen3: vision %d layers, %d wide, %d heads x %d, ff %d, %d patches, shuffle %d -> %d\n",
                      cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.dh, cfg.n_ff, cfg.n_patch, cfg.shuffle, cfg.proj_out);
        std::vector<std::vector<float>> keep;
        auto K = [&](const std::string& n) { keep.push_back(W(n)); return keep.back().data(); };
        g3rt::VitWeights w;
        keep.reserve(size_t(16 * cfg.n_layer + 8));
        w.patch_w = K("v.patch_embd.weight");
        w.patch_b = K("v.patch_embd.bias");
        w.pos = K("v.position_embd.weight");
        w.post_w = K("v.post_ln.weight");
        w.post_b = K("v.post_ln.bias");
        w.proj_w = K("mm.model.fc.weight");
        const std::string f1 = swapped ? "ffn_down" : "ffn_up", f2 = swapped ? "ffn_up" : "ffn_down";
        for (int l = 0; l < cfg.n_layer; ++l) {
            const std::string b = "v.blk." + std::to_string(l) + ".";
            g3rt::VitLayerWeights lw;
            lw.ln1_w = K(b + "ln1.weight"); lw.ln1_b = K(b + "ln1.bias");
            lw.q_w = K(b + "attn_q.weight"); lw.q_b = K(b + "attn_q.bias");
            lw.k_w = K(b + "attn_k.weight"); lw.k_b = K(b + "attn_k.bias");
            lw.v_w = K(b + "attn_v.weight"); lw.v_b = K(b + "attn_v.bias");
            lw.o_w = K(b + "attn_out.weight"); lw.o_b = K(b + "attn_out.bias");
            lw.ln2_w = K(b + "ln2.weight"); lw.ln2_b = K(b + "ln2.bias");
            lw.fc1_w = K(b + f1 + ".weight"); lw.fc1_b = K(b + f1 + ".bias");
            lw.fc2_w = K(b + f2 + ".weight"); lw.fc2_b = K(b + f2 + ".bias");
            w.layers.push_back(lw);
        }
        vs.vit.reset(new g3rt::Vit(s.device(), cfg));
        const char* cache = std::getenv("GGML_GEN3_VIT_ENCODE_CACHE");
        if (cache && std::string(cache) == "0") vs.vit->set_encode_cache(false);
        vs.vit->set_profile(s.profile.get());
        vs.vit->upload(w);
    }
    const g3rt::VitConfig& c = vs.vit->cfg();
    // im2col (ggml_im2col, 2-D, stride = kernel): patch p = oy * OW + ox, k = (ic * KH + ky) * KW + kx
    const int IW = int(inp->ne[0]), IH = int(inp->ne[1]), IC = int(inp->ne[2]);
    const int OW = int(im2col->ne[1]), OH = int(im2col->ne[2]);
    const int KW = IW / OW, KH = IH / OH;
    if (IC * KH * KW != c.patch_k || OW * OH != c.n_patch) return GGML_STATUS_FAILED;
    std::vector<float> patches(static_cast<size_t>(c.n_patch) * c.patch_k);
    const float* src = static_cast<const float*>(inp->data);
    for (int oy = 0; oy < OH; ++oy)
        for (int ox = 0; ox < OW; ++ox)
            for (int ic = 0; ic < IC; ++ic)
                for (int ky = 0; ky < KH; ++ky)
                    for (int kx = 0; kx < KW; ++kx)
                        patches[size_t(oy * OW + ox) * c.patch_k + size_t((ic * KH + ky) * KW + kx)] =
                            src[size_t(ox * KW + kx) + size_t(IW) * size_t(oy * KH + ky) + size_t(IW) * IH * ic];
    // GGML_GEN3_VIT_CHECK: the same graph on ggml's CPU backend first (the vision weights are in host memory in
    // ggml's layout), then the relative error of the Node's embeddings against it
    std::vector<float> ref;
    if (std::getenv("GGML_GEN3_VIT_CHECK")) {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu && ggml_backend_graph_compute(cpu, g) == GGML_STATUS_SUCCESS) {
            const float* o = static_cast<const float*>(out->data);
            ref.assign(o, o + ggml_nelements(out));
        }
        if (cpu) ggml_backend_free(cpu);
    }
    vs.vit->encode(patches.data(), static_cast<float*>(out->data));
    if (!ref.empty()) {
        const float* o = static_cast<const float*>(out->data);
        double num = 0, den = 0, dot = 0, nn = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = double(o[i]) - ref[i];
            num += d * d; den += double(ref[i]) * ref[i]; dot += double(o[i]) * ref[i]; nn += double(o[i]) * o[i];
        }
        GGML_LOG_INFO("ggml-gen3: vision check: relative error %.4f, cosine %.5f (%zu values)\n", std::sqrt(num / den),
                      dot / std::sqrt(den * nn + 1e-30), ref.size());
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------
// backend
// ---------------------------------------------------------------------------------------------------------------
const char* be_name(ggml_backend_t) { return "GEN3"; }
void be_free(ggml_backend_t b) { delete b; }

void dump_graph(const ggml_cgraph* g) {
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor* t = g->nodes[i];
        std::fprintf(stderr, "gen3 node %4d %-12s %-36s [%lld %lld %lld %lld] %s", i, ggml_op_desc(t), t->name,
                     (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                     ggml_type_name(t->type));
        for (int k = 0; k < GGML_MAX_SRC && t->src[k]; ++k)
            std::fprintf(stderr, " | %s [%lld %lld %lld]", t->src[k]->name, (long long)t->src[k]->ne[0],
                         (long long)t->src[k]->ne[1], (long long)t->src[k]->ne[2]);
        std::fprintf(stderr, "\n");
    }
}

bool causal_range(const ggml_tensor* t,int& delta,int& keys) {
    const auto* mask=t->src[3];const int m=int(t->src[0]->ne[1]),n=int(mask->ne[0]);
    delta=-1;
    for(int row=0;row<m;++row) {
        const auto* p=(const uint16_t*)((const char*)mask->data+size_t(row)*mask->nb[1]);
        int visible=0;while(visible<n && (p[visible]&0x7FFF)==0)++visible;
        for(int j=visible;j<n;++j)if(p[j]!=0xFC00)return false;
        if(row==0)delta=visible-1;
        if(delta<0 || visible!=delta+row+1)return false;
    }
    keys=delta+m;return keys<=n;
}

ggml_status hybrid_attention(State& s,ggml_cgraph* graph,ggml_tensor* t) {
    int delta=0,keys=0;
    if(!causal_range(t,delta,keys)) {
        // Unsupported mask semantics retain ggml's CPU result. Never hide hardware failures.
        if(!s.attention_cpu){s.attention_cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(s.attention_cpu,4);}
        auto op=*t;auto value=*t->src[2];std::vector<ggml_fp16_t> contiguous;
        if(value.nb[0]!=sizeof(ggml_fp16_t)) {
            contiguous.resize(size_t(ggml_nelements(&value)));
            for(int64_t h=0;h<value.ne[2];++h)for(int64_t row=0;row<value.ne[1];++row)for(int64_t d=0;d<value.ne[0];++d)
                std::memcpy(contiguous.data()+(size_t(h)*value.ne[1]+row)*value.ne[0]+d,
                    (const char*)value.data+size_t(h)*value.nb[2]+size_t(row)*value.nb[1]+size_t(d)*value.nb[0],sizeof(ggml_fp16_t));
            value.data=contiguous.data();value.nb[0]=sizeof(ggml_fp16_t);
            for(int j=1;j<4;++j)value.nb[j]=value.nb[j-1]*value.ne[j-1];
            value.view_src=nullptr;value.view_offs=0;op.src[2]=&value;
        }
        auto* node=&op;auto one=*graph;one.n_nodes=1;one.nodes=&node;
        ++s.attention_mask_fallbacks;return ggml_backend_graph_compute(s.attention_cpu,&one);
    }
    const auto* q=t->src[0];const auto* k=t->src[1];const auto* v=t->src[2];
    const int m=int(q->ne[1]),qh=int(q->ne[2]),kh=int(k->ne[2]),d=int(q->ne[0]);
    const char* raw_flag=std::getenv("GGML_GEN3_ATTN_F16");
    if(!raw_flag || std::strcmp(raw_flag,"0")) {
        std::vector<float> qf(size_t(qh)*m*d);
        for(int h=0;h<qh;++h)for(int row=0;row<m;++row)
            std::memcpy(qf.data()+(size_t(h)*m+row)*d,(const char*)q->data+size_t(h)*q->nb[2]+size_t(row)*q->nb[1],size_t(d)*4);
        if(!s.attention)s.attention.reset(new g3rt::Attention(s.device()));
        const g3rt::Attention::F16View kv{k->data,k->nb[0],k->nb[1],k->nb[2]},vv{v->data,v->nb[0],v->nb[1],v->nb[2]};
        s.attention->run_f16(t->name,qf.data(),kv,vv,m,keys,qh,kh,d,delta,(float*)t->data,
            [](const uint16_t* in,float* out,int n){ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(in),out,n);});
        return GGML_STATUS_SUCCESS;
    }
    std::vector<float> qf(size_t(qh)*m*d),kf(size_t(kh)*keys*d),vf(kf.size());
    for(int h=0;h<qh;++h)for(int row=0;row<m;++row)
        std::memcpy(qf.data()+(size_t(h)*m+row)*d,(const char*)q->data+size_t(h)*q->nb[2]+size_t(row)*q->nb[1],size_t(d)*4);
    for(int h=0;h<kh;++h)for(int row=0;row<keys;++row) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)((const char*)k->data+size_t(h)*k->nb[2]+size_t(row)*k->nb[1]),kf.data()+(size_t(h)*keys+row)*d,d);
    }
    if(v->nb[0]==2) {
        for(int h=0;h<kh;++h)for(int row=0;row<keys;++row)
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)((const char*)v->data+size_t(h)*v->nb[2]+size_t(row)*v->nb[1]),vf.data()+(size_t(h)*keys+row)*d,d);
    } else {
        std::vector<float> column(keys);
        for(int h=0;h<kh;++h)for(int dim=0;dim<d;++dim) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)((const char*)v->data+size_t(h)*v->nb[2]+size_t(dim)*v->nb[0]),column.data(),keys);
            for(int row=0;row<keys;++row)vf[(size_t(h)*keys+row)*d+dim]=column[row];
        }
    }
    if(!s.attention)s.attention.reset(new g3rt::Attention(s.device()));
    s.attention->run(t->name,qf.data(),kf.data(),vf.data(),m,keys,qh,kh,d,delta,(float*)t->data);
    return GGML_STATUS_SUCCESS;
}

// Only fuse the closed four-node FFN pattern. Explicit intermediate outputs
// and additional consumers retain ordinary materialized execution.
bool ffn_pattern(const ggml_cgraph* g,int i) {
    if(i+3>=g->n_nodes)return false;
    const auto* act=g->nodes[i+2];const auto* down=g->nodes[i+3];
    if(!ffn_glu(act))return false;
    const auto* gate=act->src[0];const auto* up=act->src[1];
    if(!((up==g->nodes[i] && gate==g->nodes[i+1]) || (gate==g->nodes[i] && up==g->nodes[i+1])))return false;
    if(!ffn_glu(act) || up->op!=GGML_OP_MUL_MAT || gate->op!=GGML_OP_MUL_MAT || down->op!=GGML_OP_MUL_MAT ||
       act->src[0]!=gate || act->src[1]!=up || down->src[1]!=act || up->src[1]!=gate->src[1] ||
       !hybrid_supports(up) || !hybrid_supports(gate) || !hybrid_supports(down))return false;
    const auto* x=up->src[1];int e=int(x->ne[0]),f=int(up->ne[0]);
    if(e<64 || e>1024 || e%64 || f<64 || f>8192 || f%64 || gate->ne[0]!=f || down->ne[0]!=e ||
       !ggml_are_same_shape(up,gate) || !ggml_are_same_shape(up,act) ||
       up->flags&GGML_TENSOR_FLAG_OUTPUT || gate->flags&GGML_TENSOR_FLAG_OUTPUT || act->flags&GGML_TENSOR_FLAG_OUTPUT)return false;
    // Graph views retain the full graph's use counts, including consumers in
    // other backend splits. A local scan alone would miss those consumers.
    const int nodes[]={i,i+1,i+2,i+3},outputs[]={i+3};
    const ggml_op ops[]={GGML_OP_MUL_MAT,GGML_OP_MUL_MAT,GGML_OP_GLU,GGML_OP_MUL_MAT};
    return ggml_can_fuse_subgraph_ext(g,nodes,4,ops,outputs,1);
}

ggml_status hybrid_graph(State& s, ggml_cgraph* g) {
    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_tensor* t = g->nodes[i];
        if (!hybrid_supports(t)) {
            GGML_LOG_ERROR("GEN3 hybrid received unsupported op %s %s\n", ggml_op_name(t->op), t->name);
            return GGML_STATUS_FAILED;
        }
        if(t->op==GGML_OP_FLASH_ATTN_EXT) {
            const auto status=hybrid_attention(s,g,t);if(status!=GGML_STATUS_SUCCESS)return status;continue;
        }
        if(t->op==GGML_OP_GLU) {
            if(!s.attention_cpu){s.attention_cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(s.attention_cpu,4);}
            auto* node=t;auto one=*g;one.n_nodes=1;one.nodes=&node;
            const auto status=ggml_backend_graph_compute(s.attention_cpu,&one);if(status!=GGML_STATUS_SUCCESS)return status;
            continue;
        }
        // A non-final llama.cpp ubatch can prune all output rows from the last FFN.
        // Empty tensors are valid no-ops; they must not allocate or submit hardware work.
        if (ggml_nelements(t) == 0 || t->op != GGML_OP_MUL_MAT) continue;
        if (!s.matmul) {
            size_t cache_mib = 256;
            if (const char* value = std::getenv("GGML_GEN3_CACHE_MIB")) {
                char* end = nullptr; const unsigned long n = std::strtoul(value, &end, 10);
                if (end == value || *end || n > 4096) throw std::invalid_argument("GGML_GEN3_CACHE_MIB: integer 0..4096");
                cache_mib = n;
            }
            size_t cache_bytes=cache_mib << 20;
            if (const char* value=std::getenv("GGML_GEN3_CACHE_KIB")) {
                char* end=nullptr; const unsigned long n=std::strtoul(value,&end,10);
                if(end==value || *end || n>4096u*1024) throw std::invalid_argument("GGML_GEN3_CACHE_KIB: integer 0..4194304");
                cache_bytes=size_t(n)<<10;
            }
            const char* ffn_mode=std::getenv("GGML_GEN3_FFN");
            const size_t scratch=ffn_mode && (std::string(ffn_mode)=="exact" || std::string(ffn_mode)=="quant") ? 393216 : g3rt::Matmul::output_elements;
            s.matmul.reset(new g3rt::Matmul(s.device(), cache_bytes, scratch));
            GGML_LOG_INFO("GEN3_HYBRID initialized cache_bytes=%zu; unsupported ops use CPU\n", cache_bytes);
        }
        if(ffn_pattern(g,i)) {
            const auto* gate=g->nodes[i+2]->src[0];const auto* up=g->nodes[i+2]->src[1];auto* down=g->nodes[i+3];
            const auto key=std::make_tuple(gate->src[0],up->src[0],down->src[0]);
            if(std::string(std::getenv("GGML_GEN3_FFN"))=="exact" || std::string(std::getenv("GGML_GEN3_FFN"))=="quant") {
                auto it=s.exact_ffn_weights.find(key);
                if(it==s.exact_ffn_weights.end()) {
                    auto gu=to_f32(gate->src[0],gate->src[0]->data);
                    const auto uw=to_f32(up->src[0],up->src[0]->data),dw=to_f32(down->src[0],down->src[0]->data);
                    gu.insert(gu.end(),uw.begin(),uw.end());
                    State::ExactFfn weights;
                    weights.gu=s.matmul->prepare(gu.data(),int(up->ne[0])*2,int(up->src[1]->ne[0]));
                    weights.down=s.matmul->prepare(dw.data(),int(down->ne[0]),int(down->src[0]->ne[0]));
                    it=s.exact_ffn_weights.emplace(key,std::move(weights)).first;
                }
                s.matmul->run(it->second.gu,(const float*)up->src[1]->data,int(up->ne[1]),(float*)gate->data,(float*)up->data,int(up->ne[0]));
                if(!s.exact_cpu) {
                    auto params=ggml_threadpool_params_default(4);params.poll=0;
                    s.exact_pool=ggml_threadpool_new(&params);
                    if(!s.exact_pool)throw std::runtime_error("FFN CPU threadpool allocation failed");
                    s.exact_cpu=ggml_backend_cpu_init();
                    if(!s.exact_cpu)throw std::runtime_error("FFN CPU backend allocation failed");
                    ggml_backend_cpu_set_n_threads(s.exact_cpu,4);ggml_backend_cpu_set_threadpool(s.exact_cpu,s.exact_pool);
                }
                auto* node=g->nodes[i+2];
                const bool online=std::string(std::getenv("GGML_GEN3_FFN"))=="quant" && node->ne[1]>=32 && node->ne[0]%128==0 && it->second.down.address;
                if(online) {
                    auto prepare_rows=[&](int row,int rows) {
                        // Row slicing preserves the original CPU SwiGLU kernel,
                        // op params and element order. No graph storage is mutated.
                        auto gt=*gate,ut=*up,act=*node;
                        for(auto* t:{&gt,&ut,&act}) {
                            t->data=static_cast<char*>(t->data)+size_t(row)*t->nb[1];
                            t->ne[1]=rows;t->nb[2]=t->nb[1]*size_t(rows);t->nb[3]=t->nb[2];
                        }
                        act.src[0]=&gt;act.src[1]=&ut;auto* selected=&act;
                        auto one=*g;one.n_nodes=1;one.nodes=&selected;
                        ggml_backend_cpu_set_n_threads(s.exact_cpu,std::min(4,rows));
                        if(ggml_backend_graph_compute(s.exact_cpu,&one)!=GGML_STATUS_SUCCESS)
                            throw std::runtime_error("online FFN CPU SwiGLU failed");
                    };
                    s.matmul->run_online_quantized(it->second.down,(const float*)node->data,int(node->ne[1]),(float*)down->data,prepare_rows);
                } else {
                    auto one=*g;one.n_nodes=1;one.nodes=&node;
                    ggml_backend_cpu_set_n_threads(s.exact_cpu,int(std::min<int64_t>(4,ggml_nrows(node))));
                    const auto status=ggml_backend_graph_compute(s.exact_cpu,&one);if(status!=GGML_STATUS_SUCCESS)return status;
                    s.matmul->run(it->second.down,(const float*)node->data,int(node->ne[1]),(float*)down->data);
                }
                ++s.exact_ffn_operations;i+=3;continue;
            }
            auto it=s.ffn_weights.find(key);
            if(it==s.ffn_weights.end()) {
                const auto gw=to_f32(gate->src[0],gate->src[0]->data),uw=to_f32(up->src[0],up->src[0]->data),dw=to_f32(down->src[0],down->src[0]->data);
                it=s.ffn_weights.emplace(key,s.matmul->prepare_ffn(gw.data(),uw.data(),dw.data(),int(up->src[1]->ne[0]),int(up->ne[0]))).first;
            }
            s.matmul->run_ffn(it->second,(const float*)up->src[1]->data,int(up->ne[1]),(float*)down->data);
            i+=3;continue;
        }
        auto it = s.hybrid_weights.find(t->src[0]);
        if (it == s.hybrid_weights.end()) {
            const auto* w = t->src[0]; const auto floats = to_f32(w, w->data);
            it = s.hybrid_weights.emplace(w, s.matmul->prepare(floats.data(), int(w->ne[1]), int(w->ne[0]))).first;
        }
        s.matmul->run(it->second, static_cast<const float*>(t->src[1]->data), int(t->src[1]->ne[1]), static_cast<float*>(t->data));
        if (s.matmul->operations <= 7 || std::getenv("GGML_GEN3_HYBRID_TRACE"))
            GGML_LOG_INFO("GEN3_HYBRID op=%llu name=%s m=%lld n=%lld k=%lld cached=%d\n",
                (unsigned long long)s.matmul->operations, t->name, (long long)t->ne[1],
                (long long)t->ne[0], (long long)t->src[0]->ne[0], it->second.address != 0);
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status be_graph_compute(ggml_backend_t, ggml_cgraph* g) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (std::getenv("GGML_GEN3_DUMP")) dump_graph(g);
    if (hybrid_mode()) {
        try { return hybrid_graph(s, g); }
        catch (const std::exception& e) { GGML_LOG_ERROR("GEN3 hybrid failed: %s\n", e.what()); return GGML_STATUS_FAILED; }
    }
    ggml_status st = llm_graph(s, g);
    if (st != GGML_STATUS_SUCCESS) st = vit_graph(s, g);
    if (st != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ggml-gen3: graph of %d nodes not recognised (first %s %s)\n", g->n_nodes,
                       g->n_nodes ? ggml_op_name(g->nodes[0]->op) : "-", g->n_nodes ? g->nodes[0]->name : "-");
    }
    return st;
}

ggml_backend_i backend_iface = {
    /* .get_name           = */ be_name,
    /* .free               = */ be_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ be_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

ggml_guid_t guid() {
    static ggml_guid g = {0x47, 0x33, 0x4e, 0x03, 0x67, 0x67, 0x6d, 0x6c, 0x2d, 0x67, 0x65, 0x6e, 0x33, 0x00, 0x00, 0x01};
    return &g;
}

// ---------------------------------------------------------------------------------------------------------------
// device, registry
// ---------------------------------------------------------------------------------------------------------------
ggml_backend_buffer_type buft_obj;
ggml_backend_device dev_obj;
ggml_backend_reg reg_obj;

const char* dev_name(ggml_backend_dev_t) { return "GEN3"; }
const char* dev_desc(ggml_backend_dev_t) { return "Gen3 Node v3 (PYNQ-ZU) / emulator"; }
void dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) { *free = size_t(384) << 20; *total = size_t(384) << 20; }
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
void dev_props(ggml_backend_dev_t d, ggml_backend_dev_props* p) {
    p->name = dev_name(d);
    p->description = dev_desc(d);
    dev_memory(d, &p->memory_free, &p->memory_total);
    p->type = dev_type(d);
    p->device_id = nullptr;
    p->caps = {false, false, false, false};
}
ggml_backend_t dev_init(ggml_backend_dev_t d, const char*) {
    return new ggml_backend{guid(), backend_iface, d, nullptr};
}
ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t) { return &buft_obj; }

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    // every op of the graphs it takes (the text model, the vision encoder): one GEN3 split per graph; ops whose
    // weights stay on the CPU (token_embd's GET_ROWS) go to the CPU by the scheduler's weight rule
    if (hybrid_mode()) return hybrid_supports(op);
    // Explicit legacy fused mode accepts the whole known SmolLM/SmolVLM graph.
    return true;

}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) {
    const char* host=std::getenv("GGML_GEN3_HOST_BUFFER");
    return b==&buft_obj || (hybrid_mode() && host && !std::strcmp(host,"1") && b==ggml_backend_cpu_buffer_type());
}

ggml_backend_device_i dev_iface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_desc,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init,
    /* .get_buffer_type      = */ dev_buft,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

const char* reg_name(ggml_backend_reg_t) { return "GEN3"; }
size_t reg_count(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t reg_dev(ggml_backend_reg_t, size_t) { return &dev_obj; }

ggml_backend_reg_i reg_iface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_count,
    /* .get_device       = */ reg_dev,
    /* .get_proc_address = */ nullptr,
};

}  // namespace

ggml_backend_reg_t ggml_backend_gen3_reg(void) {
    static bool init = false;
    if (!init) {
        reg_obj = ggml_backend_reg{GGML_BACKEND_API_VERSION, reg_iface, nullptr};
        dev_obj = ggml_backend_device{dev_iface, &reg_obj, nullptr};
        buft_obj = ggml_backend_buffer_type{buft_iface, &dev_obj, nullptr};
        init = true;
    }
    return &reg_obj;
}

ggml_backend_t ggml_backend_gen3_init(void) {
    ggml_backend_gen3_reg();
    return dev_init(&dev_obj, nullptr);
}
