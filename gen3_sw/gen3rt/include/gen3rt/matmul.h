// General dense matmul for CPU/GEN3 split graphs. No Llama topology assumptions.
#pragma once
#include "gen3rt/device.h"
#include "gen3rt/layout.h"
#include "gen3rt/quantize_fast.h"
#include "gen3rt/program.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

namespace g3rt {
class Matmul {
public:
    struct Weight {
        int n = 0, k = 0;
        uint64_t address = 0;
        std::vector<Word> host;
    };
    static constexpr int tile_m = 32, tile_n = 1024, max_k = 8192, max_n = 262144;
    static constexpr size_t output_elements = 262144;
    explicit Matmul(Device& dev, size_t cache_bytes, size_t output_capacity=output_elements) : dev_(dev), cache_capacity_(cache_bytes), output_capacity_(output_capacity) {
        if(output_capacity<output_elements || output_capacity>2*output_elements)throw std::invalid_argument("matmul output capacity");
        io_ = dev_.alloc(288 * 1024 + output_capacity_ * sizeof(float));
        if (cache_bytes) cache_ = dev_.alloc(cache_bytes);
    }
    ~Matmul() {
        if (!profiling_) return;
        for (int i=0;i<2;++i) {
            const auto& p=profile_[i];
            std::fprintf(stderr,"GEN3_MATMUL_PROFILE mode=%s calls=%llu quant_pack_ms=%.3f write_ms=%.3f commands_ms=%.3f wait_ms=%.3f read_ms=%.3f scatter_ms=%.3f\n",
                i?"prefill":"decode",(unsigned long long)p.calls,p.ms[0],p.ms[1],p.ms[2],p.ms[3],p.ms[4],p.ms[5]);
        }
    }
    void reset_cache() { cache_used_ = 0; quant_zero_ = {}; }
    Weight prepare(const float* data, int n, int k) {
        if (n < 64 || n > max_n || n % 64 || k < 32 || k % 32 || k > max_k)
            throw std::invalid_argument("GEN3 matmul: unsupported matrix shape");
        Weight w; w.n = n; w.k = k;
        std::vector<uint32_t> bits(size_t(n) * k);
        std::memcpy(bits.data(), data, bits.size() * 4);
        std::vector<uint8_t> codes, scales;
        quantize_fast(bits.data(), n, k, codes, scales);
        w.host = pack_fast(n, k, true, codes.data(), scales.data());
        const size_t bytes = w.host.size() * 32, aligned = (bytes + 4095) / 4096 * 4096;
        if (aligned <= cache_capacity_ - cache_used_) {
            w.address = cache_ + cache_used_; cache_used_ += aligned; peak_cache_bytes = std::max(peak_cache_bytes, cache_used_);
            dev_.write(w.address, w.host.data(), bytes);
            w.host.clear(); w.host.shrink_to_fit();
        }
        return w;
    }
    void run(const Weight& w, const float* input, int m, float* output, float* second_output=nullptr, int split_n=0) {
        if (m < 1) throw std::invalid_argument("GEN3 matmul: empty input");
        if(second_output && (split_n<1 || split_n>=w.n))throw std::invalid_argument("matmul output split");
        auto copy_output=[&](int row,int col,const float* src,int cols) {
            if(!second_output){std::memcpy(output+size_t(row)*w.n+col,src,size_t(cols)*4);return;}
            const int first=std::max(0,std::min(cols,split_n-col));
            if(first)std::memcpy(output+size_t(row)*split_n+col,src,size_t(first)*4);
            if(first<cols)std::memcpy(second_output+size_t(row)*(w.n-split_n)+std::max(0,col-split_n),src+first,size_t(cols-first)*4);
        };
        auto& profile = profile_[m > 1];
        ++profile.calls;
        auto mark = profiling_ ? Clock::now() : Clock::time_point{};
        auto tick = [&](int phase) {
            if (!profiling_) return;
            const auto now = Clock::now();
            profile.ms[phase] += std::chrono::duration<double, std::milli>(now-mark).count();
            mark = now;
        };
        const int step = std::min(tile_m, int(output_capacity_ / w.n));
        // A and D occupy disjoint UB ranges even at the maximum K and tile sizes.
        constexpr uint32_t d_base = 8448, stream_base = 16368;
        constexpr uint64_t output_offset = 288 * 1024;
        // Q/K/V and gate/up often share their input. Verify the complete bytes:
        // pointers alone are unsafe because ggml recycles buffers between graphs.
        const size_t elements = size_t(m) * w.k;
        const bool reuse = input_valid_ && input_rows_ == m && input_cols_ == w.k && input_tile_rows_ == step &&
            input_bits_.size() == elements && !std::memcmp(input_bits_.data(), input, elements * sizeof(float));
        if (reuse) ++input_reuse_hits;
        else {
            input_valid_ = false;
            input_rows_ = m; input_cols_ = w.k; input_tile_rows_ = step;
            input_bits_.resize(elements);
            std::memcpy(input_bits_.data(), input, elements * sizeof(float));
            input_words_.resize(size_t(m / step) * Layout(step,w.k).words() +
                (m % step ? Layout(m % step,w.k).words() : 0));
        }
        for (int row = 0; row < m; row += step) {
            const int rows = std::min(step, m - row);
            const size_t a_count = size_t(Layout(rows,w.k).words());
            auto* a_words = input_words_.data() + size_t(row/step)*Layout(step,w.k).words();
            if (!reuse) {
                quantize_fast(input_bits_.data()+size_t(row)*w.k, rows, w.k, codes_, scales_);
                const auto packed = pack_fast(rows, w.k, false, codes_.data(), scales_.data());
                std::copy(packed.begin(), packed.end(), a_words);
            }
            tick(0);
            dev_.write(io_, a_words, a_count * 32);
            tick(1);
            // Resident weights can be consumed in one submission. Each tile's
            // MOVE_OUT retires at a fence before D or the stream binding is reused.
            // DDR output regions remain disjoint until the single host readback.
            const bool batch = w.address != 0;
            prog_.clear();
            auto a = prog_.bind({0x01, 0, 0, round8(uint32_t(a_count)), Order::P8, Kind::Static, true});
            prog_.move_in(a.b(), 0, uint32_t(a_count), io_, true);
            prog_.fence();
            result_.resize(size_t(rows) * w.n);
            for (int col = 0; col < w.n; col += tile_n) {
                const int cols = std::min(tile_n, w.n - col);
                const uint32_t nw = uint32_t(Layout(cols, w.k, true).words());
                const uint64_t offset = uint64_t(Layout(32, w.k, true).words()) * (col / 32) * 32;
                uint64_t weights = w.address + offset;
                if (!w.address) {
                    if (!stream_) stream_ = dev_.alloc(size_t(Layout(tile_n, max_k, true).words()) * 32);
                    dev_.write(stream_, w.host.data() + offset / 32, size_t(nw) * 32);
                    weights = stream_; streamed_bytes += size_t(nw) * 32;
                }
                tick(1);
                BindArgs wb{0xF8, 0, stream_base, 8, Order::P32, Kind::Stream};
                wb.src = weights; wb.length = nw;
                auto b = prog_.bind(wb);
                auto d = prog_.bind({0x02, 0, d_base, f32_extent(rows, cols), Order::F32});
                GemmArgs g; g.a = a.b(); g.b = b.b(); g.d = d.b(); g.m = rows; g.n = cols; g.k = w.k;
                prog_.gemm(g);
                prog_.move_out(d.b(), 0, uint32_t(rows * cols / 8), io_ + output_offset + (batch ? size_t(rows) * col * 4 : 0));
                prog_.fence();
                tick(2);
                if (!batch) {
                    finish(); tick(3);
                    dev_.read(io_ + output_offset, result_.data(), size_t(rows) * cols * 4); tick(4);
                    for (int r = 0; r < rows; ++r)
                        copy_output(row+r,col,result_.data()+size_t(r)*cols,cols);
                    prog_.clear(); tick(5);
                }
                ++tiles;
            }
            if (batch) {
                finish(); tick(3);
                dev_.read(io_ + output_offset, result_.data(), size_t(rows)*w.n*4); tick(4);
                for (int col=0; col<w.n; col+=tile_n) {
                    const int cols=std::min(tile_n,w.n-col);
                    for (int r=0; r<rows; ++r)
                        copy_output(row+r,col,result_.data()+size_t(rows)*col+size_t(r)*cols,cols);
                }
                tick(5);
            }
        }
        input_valid_ = true;
        ++operations;
    }
    struct FfnWeight {
        int e=0, f=0;
        std::vector<Weight> gu, down;
    };
    FfnWeight prepare_ffn(const float* gate, const float* up, const float* down, int e, int f) {
        if(e<64 || e>1024 || e%64 || f<64 || f>8192 || f%64)
            throw std::invalid_argument("GEN3 fused FFN shape");
        FfnWeight w;w.e=e;w.f=f;
        size_t need=0;
        for(int col=0;col<f;col+=3072) {
            int n=std::min(3072,f-col);
            for(size_t bytes:{size_t(Layout(2*n,e,true).words())*32,size_t(Layout(e,n,true).words())*32})need+=(bytes+4095)/4096*4096;
        }
        if(need>cache_capacity_-cache_used_)throw std::runtime_error("fused FFN requires resident weights; increase cache or disable GGML_GEN3_FFN");
        for(int col=0;col<f;col+=3072) {
            const int n=std::min(3072,f-col);
            std::vector<float> gu(size_t(2*n)*e),d(size_t(e)*n);
            for(int j=0;j<n;j+=32) {
                std::memcpy(gu.data()+size_t(2*j)*e,gate+size_t(col+j)*e,size_t(32)*e*4);
                std::memcpy(gu.data()+size_t(2*j+32)*e,up+size_t(col+j)*e,size_t(32)*e*4);
            }
            for(int row=0;row<e;++row)std::memcpy(d.data()+size_t(row)*n,down+size_t(row)*f+col,size_t(n)*4);
            w.gu.push_back(prepare(gu.data(),2*n,e));w.down.push_back(prepare(d.data(),e,n));
        }
        return w;
    }
    // SILU_MUL emits MXFP8 directly in UB. The down projection consumes it
    // without a host round trip. Split K at 3072: R51 stores decoded N in 12 bits; N=4096 wraps.
    void run_ffn(const FfnWeight& w,const float* input,int m,float* output) {
        if(m<1 || w.gu.empty() || w.gu.size()!=w.down.size())throw std::invalid_argument("fused FFN input");
        input_valid_=false;
        for(int row=0;row<m;row+=tile_m) {
            const int rows=std::min(tile_m,m-row);
            std::vector<uint32_t> bits(size_t(rows)*w.e);
            std::memcpy(bits.data(),input+size_t(row)*w.e,bits.size()*4);
            quantize_fast(bits.data(),rows,w.e,codes_,scales_);
            const auto packed=pack_fast(rows,w.e,false,codes_.data(),scales_.data());
            dev_.write(io_,packed.data(),packed.size()*32);
            prog_.clear();
            auto a=prog_.bind({0x10,0,0,round8(uint32_t(packed.size())),Order::P8,Kind::Static,true});
            prog_.move_in(a.b(),0,uint32_t(packed.size()),io_,true);prog_.fence();
            Bound previous;
            for(size_t part=0;part<w.gu.size();++part) {
                const auto& gu=w.gu[part];const auto& down=w.down[part];const int n=down.k;
                if(!gu.address || !down.address)throw std::runtime_error("fused FFN nonresident weight");
                BindArgs wb{0xF8,0,16368,8,Order::P32,Kind::Stream};wb.src=gu.address;wb.length=Layout(2*n,w.e,true).words();
                auto b=prog_.bind(wb);auto h=prog_.bind({0x12,0,2048,round8(uint32_t(Layout(rows,n).words()))});
                GemmArgs g;g.a=a.b();g.b=b.b();g.d=h.b();g.m=rows;g.n=n;g.k=w.e;g.epi=Epi::SiluMul;prog_.gemm(g);
                BindArgs db{0xF9,0,16376,8,Order::P32,Kind::Stream};db.src=down.address;db.length=Layout(w.e,n,true).words();
                auto dw=prog_.bind(db);auto out=prog_.bind({uint8_t(0x20+(part&1)),0,uint32_t((part&1)?11264:6656),f32_extent(rows,w.e),Order::F32});
                GemmArgs d;d.a=h.b();d.b=dw.b();d.d=out.b();d.m=rows;d.n=w.e;d.k=n;
                if(part){d.c=previous.b();d.epi=Epi::Add;}
                prog_.gemm(d);prog_.fence();previous=out;tiles+=2;
            }
            prog_.move_out(previous.b(),0,uint32_t(rows*w.e/8),io_+288*1024);prog_.fence();finish();
            dev_.read(io_+288*1024,output+size_t(row)*w.e,size_t(rows)*w.e*4);
        }
        ++ffn_operations;
    }
    // Quantize an FP32 activation with the existing ADD epilogue (zero GEMM + C),
    // compact its P8 chunks by device DMA, then consume it with one full-K GEMM.
    // No host quantization, P8 packing, or intermediate device readback.
    void run_online_quantized(const Weight& w,const float* input,int m,float* output,
                              const std::function<void(int,int)>& prepare_rows = {}) {
        if(m<1 || w.k%128 || w.n>1024 || !w.address)
            throw std::invalid_argument("online quantized matmul requires resident N<=1024, K multiple of 128");
        if(!quant_zero_.address) {
            std::vector<float> zeros(2048*32,0);quant_zero_=prepare(zeros.data(),2048,32);
            if(!quant_zero_.address)throw std::runtime_error("online quantization requires resident zero weights");
        }
        input_valid_=false;
        if(prepare_rows)prepare_rows(0,std::min(tile_m,m));
        auto& profile=profile_[m>1];++profile.calls;
        auto mark=profiling_?Clock::now():Clock::time_point{};
        auto tick=[&](int phase) {
            if(!profiling_)return;
            const auto now=Clock::now();profile.ms[phase]+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;
        };
        constexpr uint64_t hidden_offset=size_t(tile_m)*max_k*4;
        constexpr uint64_t max_p8_bytes=33u*((tile_m/8)*(max_k/32)/4)*32;
        constexpr uint64_t zero_offset=hidden_offset+max_p8_bytes;
        std::vector<uint8_t> zc(32*32,0),zs(32,127);
        const auto zero=pack_fast(32,32,false,zc.data(),zs.data());dev_.write(io_+zero_offset,zero.data(),zero.size()*32);tick(1);
        Bound zero_b;
        const uint32_t zero_b_words=uint32_t(Layout(std::min(2048,w.k),32,true).words());
        const bool hidden_overwrites_zero=4096+Layout(tile_m,w.k).words()>11264;
        for(int row=0;row<m;row+=tile_m) {
            const int rows=std::min(tile_m,m-row),panels=(rows+7)/8;
            // Upload original FP32 rows once. Strided MOVE_IN commands gather
            // each C tile on the device, avoiding a second host layout copy.
            dev_.write(io_,input+size_t(row)*w.k,size_t(rows)*w.k*4);tick(1);prog_.clear();
            auto a=prog_.bind({0x10,0,8192,40,Order::P8,Kind::Static,true});
            prog_.move_in(a.b(),0,uint32_t(zero.size()),io_+zero_offset,true);prog_.fence();
            // Static zero B avoids streaming the same zero matrix for every
            // quantization chunk. Reuse across row tiles when full H is disjoint.
            if(row==0 || hidden_overwrites_zero) {
                zero_b=prog_.bind({0xF8,0,11264,round8(zero_b_words),Order::P32,Kind::Static,true});
                prog_.move_in(zero_b.b(),0,zero_b_words,quant_zero_.address,true);prog_.fence();
            }
            for(int col=0;col<w.k;col+=2048) {
                const int cols=std::min(2048,w.k-col);
                auto cbind=prog_.bind({0x11,0,0,f32_extent(rows,cols),Order::F32,Kind::Static,true});
                for(int r=0;r<rows;++r)prog_.move_in(cbind.b(),uint32_t(r*cols/8),uint32_t(cols/8),io_+(size_t(r)*w.k+col)*4,r+1==rows);
                prog_.fence();

                auto h=prog_.bind({0x12,0,8448,round8(uint32_t(Layout(rows,cols).words()))});
                GemmArgs g;g.a=a.b();g.b=zero_b.b();g.c=cbind.b();g.d=h.b();g.m=rows;g.n=cols;g.k=32;g.epi=Epi::Add;prog_.gemm(g);
                // K/chunks are multiples of 128: each panel ends at a complete
                // four-block scale record. DMA can concatenate without repacking.
                const uint32_t panel_words=uint32_t(cols/128*33),full_panel_words=uint32_t(w.k/128*33);
                for(int p=0;p<panels;++p)prog_.move_out(h.b(),p*panel_words,panel_words,io_+hidden_offset+uint64_t(p*full_panel_words+col/128*33)*32);
                prog_.fence();++tiles;
            }
            const uint32_t words=uint32_t(Layout(rows,w.k).words());
            auto h=prog_.bind({0x18,0,4096,round8(words),Order::P8,Kind::Static,true});
            prog_.move_in(h.b(),0,words,io_+hidden_offset,true);prog_.fence();
            BindArgs db{0xF9,0,16376,8,Order::P32,Kind::Stream};db.src=w.address;db.length=Layout(w.n,w.k,true).words();auto b=prog_.bind(db);
            auto d=prog_.bind({0x20,0,0,f32_extent(rows,w.n),Order::F32});
            GemmArgs g;g.a=h.b();g.b=b.b();g.d=d.b();g.m=rows;g.n=w.n;g.k=w.k;prog_.gemm(g);
            prog_.move_out(d.b(),0,uint32_t(rows*w.n/8),io_+288*1024);prog_.fence();tick(2);
            dev_.submit(prog_.cmds);
            // The current tile is already in device memory. Compute the next
            // CPU activation tile while FPGA quantization/down are in flight.
            try {
                if(prepare_rows && row+rows<m)prepare_rows(row+rows,std::min(tile_m,m-row-rows));
            } catch(...) {
                complete(); // drain device work before propagating a CPU error
                throw;
            }
            complete();tick(3);
            dev_.read(io_+288*1024,output+size_t(row)*w.n,size_t(rows)*w.n*4);tick(4);++tiles;
        }
        ++operations;++online_quant_operations;
    }
    uint64_t online_quant_operations=0;

    uint64_t ffn_operations=0;

    size_t cache_used() const { return cache_used_; }
    uint64_t operations = 0, tiles = 0, streamed_bytes = 0, submissions = 0, input_reuse_hits = 0;
    size_t peak_cache_bytes = 0;
private:
    Weight quant_zero_;
    void finish() {
        dev_.submit(prog_.cmds);complete();
    }
    void complete() {
        dev_.sync();
        for (const auto& c : dev_.take_completions())
            if (c.status != g3::ST_OK) throw std::runtime_error("GEN3 matmul command failed: " + std::to_string(c.status));
        ++submissions;
    }
    std::vector<float> result_;
    std::vector<uint32_t> input_bits_;
    std::vector<Word> input_words_;
    std::vector<uint8_t> codes_, scales_;
    int input_rows_ = 0, input_cols_ = 0, input_tile_rows_ = 0;
    bool input_valid_ = false;
    using Clock = std::chrono::steady_clock;
    struct PhaseProfile { uint64_t calls=0; double ms[6]{}; };
    bool profiling_ = std::getenv("GGML_GEN3_MATMUL_PROFILE") != nullptr;
    PhaseProfile profile_[2];
    Device& dev_;
    Program prog_;
    size_t cache_capacity_ = 0, cache_used_ = 0, output_capacity_ = output_elements;
    uint64_t io_ = 0, stream_ = 0, cache_ = 0;
};
} // namespace g3rt
