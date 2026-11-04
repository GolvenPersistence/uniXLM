// Exact MXFP8 input/weight conversion. AArch64 uses integer NEON lanes;
// exceptional FP32 blocks retain the scalar reference's NaN/subnormal semantics.
#pragma once
#include "gen3rt/layout.h"
#include "g3emu_kernels.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
namespace g3rt {
inline bool conversion_fast_enabled() {
    static const bool enabled=[] {const char* v=std::getenv("GGML_GEN3_CONVERT_FAST");return !v || std::strcmp(v,"0");}();
    return enabled;
}
inline int conversion_threads() {
    static const int count=[] {const char* v=std::getenv("GGML_GEN3_QUANT_THREADS");
        if(!v)return 1;
        char* end=nullptr;long n=std::strtol(v,&end,10);
        if(end==v || *end || n<1 || n>4)throw std::invalid_argument("GGML_GEN3_QUANT_THREADS: integer 1..4");
        return int(n);}();return count;
}
inline void quantize_fast(const uint32_t* bits, int rows, int cols,
                          std::vector<uint8_t>& codes, std::vector<uint8_t>& scales) {
#ifndef __aarch64__
    quantize(bits, rows, cols, codes, scales);
#else
    codes.resize(size_t(rows)*cols);
    scales.resize(size_t(rows)*(cols/32));
    const bool fast=conversion_fast_enabled();
    auto convert_block=[&](size_t block) {
        const auto* in=bits+block*32;
        uint32x4_t values[8], maximum=vdupq_n_u32(0), exceptional=vdupq_n_u32(0), minimum=vdupq_n_u32(255);
        for (int j=0;j<8;++j) {
            values[j]=vld1q_u32(in+j*4);
            auto abs=vandq_u32(values[j],vdupq_n_u32(0x7fffffff));
            auto exponent=vshrq_n_u32(abs,23);
            maximum=vmaxq_u32(maximum,abs);
            minimum=vminq_u32(minimum,vbslq_u32(vceqq_u32(abs,vdupq_n_u32(0)),vdupq_n_u32(255),exponent));
            exceptional=vorrq_u32(exceptional,vorrq_u32(vceqq_u32(exponent,vdupq_n_u32(255)),
                vandq_u32(vceqq_u32(exponent,vdupq_n_u32(0)),vcgtq_u32(abs,vdupq_n_u32(0)))));
        }
        if (vmaxvq_u32(exceptional)) {
            g3::quantize_block32(in,codes.data()+block*32,scales.data()+block);
            return;
        }
        // Exceptional values already took the scalar path. Inline the finite
        // scale rule so eight live NEON vectors do not cross an external call.
        const uint32_t max_bits=vmaxvq_u32(maximum);
        const uint8_t scale=max_bits==0?127:uint8_t(std::max(0,int(max_bits>>23)-8+int((max_bits&0x7fffff)>0x600000)));
        scales[block]=scale;
        const bool normal_only=fast && int(vminvq_u32(minimum))-int(scale)>=-6;
        uint16x4_t packed[8];
        for (int j=0;j<8;++j) {
            auto value=values[j];
            auto abs=vandq_u32(value,vdupq_n_u32(0x7fffffff));
            auto exponent=vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(abs,23)),vdupq_n_s32(scale));
            auto significand=vorrq_u32(vandq_u32(value,vdupq_n_u32(0x7fffff)),vdupq_n_u32(0x800000));
            auto rounded=vshrq_n_u32(vaddq_u32(vaddq_u32(significand,vdupq_n_u32(0x7ffff)),
                vandq_u32(vshrq_n_u32(significand,20),vdupq_n_u32(1))),20);
            auto normal=vreinterpretq_u32_s32(vaddq_s32(vshlq_n_s32(vaddq_s32(exponent,vdupq_n_s32(6)),3),vreinterpretq_s32_u32(rounded)));
            normal=vminq_u32(normal,vdupq_n_u32(126));
            // For subnormal E4M3: integer RNE at a variable shift. Shifts >31
            // round a 24-bit significand to zero; clamping avoids undefined shifts.
            auto magnitude=normal;
            if(!normal_only) {
            auto shift=vminq_s32(vmaxq_s32(vsubq_s32(vdupq_n_s32(14),exponent),vdupq_n_s32(1)),vdupq_n_s32(31));
            auto half=vshlq_u32(vdupq_n_u32(1),vsubq_s32(shift,vdupq_n_s32(1)));
            auto quotient=vshlq_u32(significand,vnegq_s32(shift));
            auto subnormal=vshlq_u32(vaddq_u32(vaddq_u32(significand,vsubq_u32(half,vdupq_n_u32(1))),
                vandq_u32(quotient,vdupq_n_u32(1))),vnegq_s32(shift));
            magnitude=vbslq_u32(vcgeq_s32(exponent,vdupq_n_s32(-6)),normal,subnormal);
            }
            magnitude=vbslq_u32(vceqq_u32(abs,vdupq_n_u32(0)),vdupq_n_u32(0),magnitude);
            auto sign=vandq_u32(vshrq_n_u32(value,24),vdupq_n_u32(128));
            packed[j]=vmovn_u32(vorrq_u32(magnitude,sign));
        }
        for (int j=0;j<8;j+=4) {
            auto lo=vmovn_u16(vcombine_u16(packed[j],packed[j+1]));
            auto hi=vmovn_u16(vcombine_u16(packed[j+2],packed[j+3]));
            vst1q_u8(codes.data()+block*32+j*4,vcombine_u8(lo,hi));
        }

    };
#ifdef _OPENMP
    const int threads=conversion_threads();
    if(threads>1 && size_t(rows)*(cols/32)>=1024) {
#pragma omp parallel for num_threads(threads)
        for(size_t block=0;block<size_t(rows)*(cols/32);++block)convert_block(block);
    } else
#endif
    {
        // Do not enter a serialized OpenMP region for decode: doing so changes
        // worker-team state and adds wakeup overhead to subsequent CPU graphs.
        for(size_t block=0;block<size_t(rows)*(cols/32);++block)convert_block(block);
    }
#endif
}

// Transpose eight rows of 16-bit code pairs into the Node's 256-bit words.
// This is a byte permutation only, including zero-filled padding lanes.
inline std::vector<Word> pack_fast(int rows, int cols, bool p32,
                                   const uint8_t* codes, const uint8_t* scales, int heads=1) {
    const Layout layout(rows,cols,p32,heads);
#ifndef __aarch64__
    return layout.pack(codes,scales);
#else
    std::vector<Word> out(size_t(layout.words()));
    for(int head=0;head<heads;++head) for(int panel=0;panel<(rows+7)/8;++panel) for(int group=0;group<cols/32;++group) {
        const unsigned block=unsigned(layout.block(head*layout.head_panels()+panel,group));
        const unsigned word=33*(block/4)+1+8*(block%4);
        for(int lane=0;lane<8 && panel*8+lane<rows;++lane)
            out[33*(block/4)].b[8*(block%4)+lane]=scales[(size_t(head)*rows+panel*8+lane)*(cols/32)+group];
        for(int half=0;half<2;++half) {
            uint16x8_t a[8],b[8];uint32x4_t c[8];
            for(int lane=0;lane<8;++lane) {
                a[lane]=panel*8+lane<rows ? vreinterpretq_u16_u8(vld1q_u8(codes+(size_t(head)*rows+panel*8+lane)*cols+group*32+half*16)) : vdupq_n_u16(0);
            }
            for(int j=0;j<8;j+=2) {b[j]=vtrn1q_u16(a[j],a[j+1]);b[j+1]=vtrn2q_u16(a[j],a[j+1]);}
            for(int j=0;j<8;j+=4) {
                c[j]=vtrn1q_u32(vreinterpretq_u32_u16(b[j]),vreinterpretq_u32_u16(b[j+2]));
                c[j+1]=vtrn1q_u32(vreinterpretq_u32_u16(b[j+1]),vreinterpretq_u32_u16(b[j+3]));
                c[j+2]=vtrn2q_u32(vreinterpretq_u32_u16(b[j]),vreinterpretq_u32_u16(b[j+2]));
                c[j+3]=vtrn2q_u32(vreinterpretq_u32_u16(b[j+1]),vreinterpretq_u32_u16(b[j+3]));
            }
            for(int j=0;j<4;++j) {
                auto lo=vtrn1q_u64(vreinterpretq_u64_u32(c[j]),vreinterpretq_u64_u32(c[j+4]));
                auto hi=vtrn2q_u64(vreinterpretq_u64_u32(c[j]),vreinterpretq_u64_u32(c[j+4]));
                vst1q_u8(out[word+half*4+j/2].b+(j%2)*16,vreinterpretq_u8_u64(lo));
                vst1q_u8(out[word+half*4+2+j/2].b+(j%2)*16,vreinterpretq_u8_u64(hi));
            }
        }
    }
    return out;
#endif
}

// All 65,536 code/scale pairs retain the scalar emulator's exact bit pattern,
// including signed zero, subnormals, overflow and canonical NaNs. The table is
// process-local, initialized once; no floating-point approximation is added.
inline const std::array<uint32_t,65536>& decode_table() {
    static const auto table=[] {std::array<uint32_t,65536> t{};
        for(int s=0;s<256;++s)for(int c=0;c<256;++c)t[s*256+c]=g3::decode_e4m3(uint8_t(c),uint8_t(s));
        return t;}();return table;
}

// Fuse layout unpacking and exact dequantization, avoiding codes/scales and
// double/ldexp conversions for every output element. Output remains head-major.
inline void unpack_dequantize_fast(const Layout& layout,const Word* words,uint32_t* out) {
    const auto& table=decode_table();const int groups=layout.groups();
    for(int h=0;h<layout.heads;++h)for(int row=0;row<layout.rows;++row) {
        const int panel=h*layout.head_panels()+row/8,lane=row%8;
        for(int group=0;group<groups;++group) {
            const int block=layout.block(panel,group),base=33*(block/4)+1+8*(block%4);
            const auto* lut=table.data()+256*words[33*(block/4)].b[8*(block%4)+lane];
            for(int c=0;c<32 && group*32+c<layout.cols;++c)
                out[(size_t(h)*layout.rows+row)*layout.cols+group*32+c]=lut[words[base+c/4].b[(c%4)/2*16+lane*2+c%2]];
        }
    }
}

} // namespace g3rt
