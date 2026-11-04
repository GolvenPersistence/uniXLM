// Included inside ggml-cpu.c after its type traits and barrier definitions.
// A53 lacks SDOT/I8MM. This Q8 GEMV preserves the existing ARM accumulation
// order while reusing activation scales and reducing widened integer products.
#pragma once
static bool g3_a53_q8_gemv(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
#if defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_FEATURE_SVE)
    const char * enabled = getenv("GGML_GEN3_CPU_A53");
    if (enabled && strcmp(enabled,"0")==0) return false;
    const struct ggml_tensor * w=dst->src[0], * x=dst->src[1];
    if (w->type!=GGML_TYPE_Q8_0 || x->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(dst) ||
        w->ne[2]!=1 || w->ne[3]!=1 || x->ne[1]!=1 || x->ne[2]!=1 || x->ne[3]!=1 ||
        w->ne[0]!=x->ne[0] || w->ne[0]<64 || w->ne[0]>8192 || w->ne[0]%64 ||
        w->ne[1]<256) return false;
    const int k=(int)w->ne[0], nb=k/32;
    block_q8_0 y[8192/32]; float yd[8192/32];
    type_traits_cpu[GGML_TYPE_Q8_0].from_float((const float*)x->data,y,k);
    // Two signed products fit int16 when y excludes -128. Check explicitly
    // rather than assuming externally supplied/nonfinite activations obey it.
    for(int b=0;b<nb;++b) {
        if(vminvq_s8(vld1q_s8(y[b].qs))==-128 || vminvq_s8(vld1q_s8(y[b].qs+16))==-128) return false;
        yd[b]=GGML_CPU_FP16_TO_FP32(y[b].d);
        if(!isfinite(yd[b])) return false;
    }
    const int64_t begin=w->ne[1]*params->ith/params->nth, end=w->ne[1]*(params->ith+1)/params->nth;
    for(int64_t row=begin;row<end;++row) {
        const block_q8_0 * a=(const block_q8_0*)((const char*)w->data+row*w->nb[1]);
        float32x4_t sum0=vdupq_n_f32(0),sum1=vdupq_n_f32(0);
        for(int b=0;b<nb;b+=2) {
            const int8x16_t y00=vld1q_s8(y[b].qs), y01=vld1q_s8(y[b].qs+16);
            const int8x16_t y10=vld1q_s8(y[b+1].qs), y11=vld1q_s8(y[b+1].qs+16);
            const int8x16_t x00=vld1q_s8(a[b].qs), x01=vld1q_s8(a[b].qs+16);
            const int8x16_t x10=vld1q_s8(a[b+1].qs), x11=vld1q_s8(a[b+1].qs+16);
            const int16x8_t p00=vmlal_s8(vmull_s8(vget_low_s8(x00),vget_low_s8(y00)),vget_low_s8(x01),vget_low_s8(y01));
            const int16x8_t p01=vmlal_s8(vmull_s8(vget_high_s8(x00),vget_high_s8(y00)),vget_high_s8(x01),vget_high_s8(y01));
            const int16x8_t p10=vmlal_s8(vmull_s8(vget_low_s8(x10),vget_low_s8(y10)),vget_low_s8(x11),vget_low_s8(y11));
            const int16x8_t p11=vmlal_s8(vmull_s8(vget_high_s8(x10),vget_high_s8(y10)),vget_high_s8(x11),vget_high_s8(y11));
            const int32x4_t dot0=vaddq_s32(vpaddlq_s16(p00),vpaddlq_s16(p01));
            const int32x4_t dot1=vaddq_s32(vpaddlq_s16(p10),vpaddlq_s16(p11));
            sum0=vmlaq_n_f32(sum0,vcvtq_f32_s32(dot0),GGML_CPU_FP16_TO_FP32(a[b].d)*yd[b]);
            sum1=vmlaq_n_f32(sum1,vcvtq_f32_s32(dot1),GGML_CPU_FP16_TO_FP32(a[b+1].d)*yd[b+1]);
        }
        ((float*)dst->data)[row]=vaddvq_f32(sum0)+vaddvq_f32(sum1);
    }
    return true;
#else
    (void)params;(void)dst;return false;
#endif
}
