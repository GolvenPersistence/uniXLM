// Standalone causal attention for split ggml graphs; existing R51 ATTN commands.
#pragma once
#include "gen3rt/device.h"
#include "gen3rt/layout.h"
#include "gen3rt/quantize_fast.h"
#include "gen3rt/program.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace g3rt {
class Attention {
public:
    static constexpr int capacity=2048, tile=128;
    explicit Attention(Device& dev):dev_(dev),io_(dev.alloc(256*1024)) {
        if (const char* value=std::getenv("GGML_GEN3_ATTN_CAPACITY")) {
            char* end=nullptr; const long n=std::strtol(value,&end,10);
            if(end==value || *end || n<32 || n>2048 || n%32) throw std::invalid_argument("attention capacity: multiple of 32 in 32..2048");
            capacity_=int(n);
        }
    }
    struct F16View {
        const void* data=nullptr;
        size_t dim_stride=2,row_stride=0,head_stride=0;
    };
    using ConvertF16 = void (*)(const uint16_t*,float*,int);
    void run_f16(const std::string& id,const float* q,const F16View& k,const F16View& v,
                 int m,int keys,int qh,int kvh,int d,int delta,float* out,ConvertF16 convert) {
        if(!convert || !k.data || !v.data)throw std::invalid_argument("attention F16 source");
        const F16View views[]={k,v};
        run(id,q,nullptr,nullptr,m,keys,qh,kvh,d,delta,out,views,convert);
    }
    // Q [qh][m][d], K/V [kvh][keys][d]; output [m][qh][d].
    // All input values are FP32. The caller validates scale and causal masking.
    void run(const std::string& id,const float* q,const float* k,const float* v,
             int m,int keys,int qh,int kvh,int d,int delta,float* out,const F16View* f16=nullptr,ConvertF16 convert=nullptr) {
        if(m<1 || keys<1 || keys>capacity_ || delta<0 || delta+m!=keys ||
           qh<1 || qh>16 || kvh<1 || kvh>4 || qh%kvh || d<32 || d>128 || d%32)
            throw std::invalid_argument("GEN3 attention shape/causal range");
        auto it=cache_.find(id);
        if(it==cache_.end()) {
            if(cache_.size()>=64)throw std::runtime_error("GEN3 attention cache limit");
            Cache c;c.d=d;c.h=kvh;c.words=Layout(capacity_,d,true).words();
            c.base=dev_.alloc(size_t(c.words)*32*kvh*2);
            c.snapshot.resize(size_t(kvh)*2*capacity_*d);
            c.valid.resize(size_t(kvh)*2*(capacity_/32),false);
            c.raw.resize(size_t(kvh)*2*capacity_*d);
            c.raw_valid.resize(c.valid.size(),false);
            c.rows_raw.resize(size_t(kvh)*2*capacity_*d);
            c.k_codes.resize(size_t(kvh)*capacity_*d);
            c.k_scales.resize(size_t(kvh)*capacity_*(d/32));
            it=cache_.emplace(id,std::move(c)).first;
        }
        auto& c=it->second;
        if(c.d!=d || c.h!=kvh)throw std::invalid_argument("GEN3 attention cache shape changed");
        std::vector<uint32_t> block(size_t(32)*d),vt(size_t(32)*d);
        std::vector<uint8_t> codes,scales;
        std::vector<uint16_t> raw(size_t(32)*d);
        std::vector<float> converted(size_t(32)*d);
        int skip[2]={};
        // Common llama FA KV layout is [token][head][dim]. One bulk check
        // validates the unchanged full-band prefix, avoiding thousands of
        // per-row copies/checks. No pointer-only or append-only assumption.
        for(int kind=0;kind<2;++kind) {
            if(!f16){c.rows_valid[kind]=0;continue;}
            const auto& view=f16[kind];
            if(view.dim_stride==2 && view.head_stride==size_t(d)*2 && view.row_stride==size_t(kvh*d)*2) {
                const int prefix=std::min(keys,c.rows_valid[kind])/32*32;
                f16_checked_values+=size_t(prefix)*kvh*d;
                if(prefix && !std::memcmp(view.data,c.rows_raw.data()+size_t(kind)*capacity_*kvh*d,size_t(prefix)*kvh*d*2)) {
                    skip[kind]=prefix/32;reused_bands+=uint64_t(skip[kind])*kvh;++f16_prefix_hits;
                }
            }
            // A later upload may fail after earlier bands have changed DDR.
            // Only publish this prefix again after every band succeeds.
            c.rows_valid[kind]=0;
        }
        for(int kind=0;kind<2;++kind)for(int h=0;h<kvh;++h)for(int b=skip[kind];b<(keys+31)/32;++b) {
            const int rows=std::min(32,keys-b*32),slot=kind*kvh+h;
            std::fill(block.begin(),block.end(),0);
            auto* snapshot=c.snapshot.data()+(size_t(slot)*capacity_+b*32)*d;
            auto* raw_snapshot=c.raw.data()+(size_t(slot)*capacity_+b*32)*d;
            const size_t index=size_t(slot)*(capacity_/32)+b;
            bool changed[32];std::fill(changed,changed+32,true);int dirty=32;
            const bool previous=c.valid[index] && c.raw_valid[index];
            if(f16) {
                // Validate original F16 bytes before conversion. This remains safe
                // for cache restore, rewinds, sequence reuse and in-place edits.
                const auto& view=f16[kind];std::fill(raw.begin(),raw.end(),uint16_t(0));
                const auto* src=static_cast<const char*>(view.data)+size_t(h)*view.head_stride+size_t(b*32)*view.row_stride;
                for(int row=0;row<rows;++row) {
                    if(view.dim_stride==2)std::memcpy(raw.data()+size_t(row)*d,src+size_t(row)*view.row_stride,size_t(d)*2);
                    else for(int dim=0;dim<d;++dim)std::memcpy(raw.data()+size_t(row)*d+dim,src+size_t(row)*view.row_stride+size_t(dim)*view.dim_stride,2);
                }
                f16_checked_values+=size_t(rows)*d;
                if(c.valid[index] && c.raw_valid[index] && !std::memcmp(raw_snapshot,raw.data(),raw.size()*2)){++reused_bands;continue;}
                c.raw_valid[index]=false;
                if(previous) {
                    dirty=0;
                    for(int row=0;row<32;++row){changed[row]=std::memcmp(raw.data()+size_t(row)*d,raw_snapshot+size_t(row)*d,size_t(d)*2)!=0;dirty+=changed[row];}
                }
                if(dirty<=4) {
                    std::memcpy(block.data(),snapshot,block.size()*4);
                    for(int row=0;row<32;++row)if(changed[row]){
                        convert(raw.data()+size_t(row)*d,converted.data(),d);
                        std::memcpy(block.data()+size_t(row)*d,converted.data(),size_t(d)*4);
                        f16_converted_values+=d;
                    }
                }else {
                    convert(raw.data(),converted.data(),int(block.size()));
                    std::memcpy(block.data(),converted.data(),block.size()*4);
                    f16_converted_values+=block.size();
                }
            } else {
                const float* src=(kind?v:k)+(size_t(h)*keys+b*32)*d;
                std::memcpy(block.data(),src,size_t(rows)*d*4);
                if(c.valid[index] && !std::memcmp(snapshot,block.data(),block.size()*4)) {++reused_bands;continue;}
                c.raw_valid[index]=false;
            }
            // Mark invalid until the complete DDR write succeeds.
            c.valid[index]=false;
            std::vector<Word> packed;
            if(kind==0) {
                auto* kc=c.k_codes.data()+(size_t(h)*capacity_+b*32)*d;
                auto* ks=c.k_scales.data()+(size_t(h)*capacity_+b*32)*(d/32);
                // K scales are per row: changing one token cannot change old
                // rows' codes/scales. V scales span 32 tokens and still need
                // the complete changed tail band below.
                if(f16 && previous && dirty<=4) {
                    for(int row=0;row<32;++row)if(changed[row]){
                        quantize_fast(block.data()+size_t(row)*d,1,d,codes,scales);
                        std::memcpy(kc+size_t(row)*d,codes.data(),size_t(d));
                        std::memcpy(ks+size_t(row)*(d/32),scales.data(),size_t(d/32));
                        k_quantized_values+=d;
                    }
                } else {
                    quantize_fast(block.data(),32,d,codes,scales);
                    std::memcpy(kc,codes.data(),codes.size());std::memcpy(ks,scales.data(),scales.size());
                    k_quantized_values+=block.size();
                }
                packed=pack_fast(32,d,true,kc,ks);
            } else {
                for(int x=0;x<d;++x)for(int t=0;t<32;++t)vt[size_t(x)*32+t]=block[size_t(t)*d+x];
                quantize_fast(vt.data(),d,32,codes,scales);
                packed=pack_fast(d,32,true,codes.data(),scales.data());
            }
            dev_.write(c.base+(size_t(slot)*c.words+b*Layout(32,d,true).words())*32,packed.data(),packed.size()*32);
            std::memcpy(snapshot,block.data(),block.size()*4);
            if(f16){std::memcpy(raw_snapshot,raw.data(),raw.size()*2);c.raw_valid[index]=true;}
            c.valid[index]=true;++uploaded_bands;
        }
        if(f16)for(int kind=0;kind<2;++kind) {
            const auto& view=f16[kind];
            if(view.dim_stride==2 && view.head_stride==size_t(d)*2 && view.row_stride==size_t(kvh*d)*2) {
                const size_t start=size_t(skip[kind])*32*kvh*d;
                std::memcpy(c.rows_raw.data()+size_t(kind)*capacity_*kvh*d+start,
                    static_cast<const char*>(view.data)+start*2,(size_t(keys)*kvh*d-start)*2);
                c.rows_valid[kind]=keys;
            }
        }
        const int group=qh/kvh;
        struct Part {int h,nh;size_t offset;int words;};
        int batch=tile;
        auto bytes_for=[&](int rows) {size_t bytes=0;for(int h=0;h<kvh;++h)for(int g=0;g<group;g+=4)bytes+=size_t(Layout(rows,d,false,std::min(4,group-g)).words())*32;return bytes;};
        while(bytes_for(batch)>128*1024)batch/=2;
        for(int row=0;row<m;row+=batch) {
            const int rows=std::min(batch,m-row);
            std::vector<Part> parts;std::vector<Word> input,output;
            prog_.clear();
            for(int kh=0;kh<kvh;++kh)for(int gh=0;gh<group;gh+=4) {
                const int nh=std::min(4,group-gh),h=kh*group+gh;
                std::vector<uint32_t> qb(size_t(nh)*rows*d);
                for(int j=0;j<nh;++j)std::memcpy(qb.data()+size_t(j)*rows*d,q+(size_t(h+j)*m+row)*d,size_t(rows)*d*4);
                quantize_fast(qb.data(),nh*rows,d,codes,scales);
                auto packed=conversion_fast_enabled()?pack_fast(rows,d,false,codes.data(),scales.data(),nh):Layout(rows,d,false,nh).pack(codes.data(),scales.data());
                const size_t offset=input.size();input.insert(input.end(),packed.begin(),packed.end());
                parts.push_back({h,nh,offset,int(packed.size())});
                auto Q=prog_.bind({0x60,0,0,round8(uint32_t(packed.size())),Order::P8,Kind::Static,true});
                prog_.move_in(Q.b(),0,uint32_t(packed.size()),io_+offset*32,true);
                prog_.fence(); // R51 requires Q commit to retire before ATTN start.
                BindArgs ks{0xF8,0,16368,8,Order::P32,Kind::Stream};ks.src=c.base+size_t(kh)*c.words*32;ks.length=c.words;
                BindArgs vs{0xF9,0,16376,8,Order::VT,Kind::Stream};vs.src=c.base+size_t(kvh+kh)*c.words*32;vs.length=c.words;
                auto K=prog_.bind(ks),V=prog_.bind(vs);
                auto O=prog_.bind({0x61,0,4096,round8(uint32_t(packed.size()))});
                AttnArgs a;a.q=Q.b();a.k=K.b();a.v=V.b();a.o=O.b();a.tokens=rows;a.q_heads=nh;a.group=nh;a.keys=keys;a.delta=delta+row;a.dh=d;
                prog_.attn(a);prog_.move_out(O.b(),0,uint32_t(packed.size()),io_+128*1024+offset*32);prog_.fence();++commands;
            }
            if(input.size()*32>128*1024)throw std::runtime_error("GEN3 attention I/O overflow");
            dev_.write(io_,input.data(),input.size()*32);
            dev_.submit(prog_.cmds);dev_.sync();
            for(const auto& done:dev_.take_completions())if(done.status!=g3::ST_OK)
                throw std::runtime_error("GEN3 attention command failed: "+std::to_string(done.status));
            ++submissions;output.resize(input.size());dev_.read(io_+128*1024,output.data(),output.size()*32);
            for(const auto& part:parts) {
                std::vector<uint32_t> bits(size_t(part.nh)*rows*d);
                if(conversion_fast_enabled())unpack_dequantize_fast(Layout(rows,d,false,part.nh),output.data()+part.offset,bits.data());
                else {
                    codes.resize(size_t(part.nh)*rows*d);scales.resize(size_t(part.nh)*rows*(d/32));
                    Layout(rows,d,false,part.nh).unpack(output.data()+part.offset,codes.data(),scales.data());
                    dequantize(codes.data(),scales.data(),part.nh*rows,d,bits.data());
                }
                for(int h=0;h<part.nh;++h)for(int t=0;t<rows;++t)
                    std::memcpy(out+(size_t(row+t)*qh+part.h+h)*d,bits.data()+(size_t(h)*rows+t)*d,size_t(d)*4);
            }
        }
        ++operations;
    }
    uint64_t operations=0,commands=0,submissions=0,uploaded_bands=0,reused_bands=0;
    uint64_t f16_checked_values=0,f16_converted_values=0,f16_prefix_hits=0,k_quantized_values=0;
private:
    struct Cache {int d=0,h=0,words=0;uint64_t base=0;std::vector<uint32_t> snapshot;std::vector<bool> valid;std::vector<uint16_t> raw;std::vector<bool> raw_valid;std::vector<uint16_t> rows_raw;std::vector<uint8_t> k_codes,k_scales;int rows_valid[2]={};};
    int capacity_=capacity;
    Device& dev_;uint64_t io_;Program prog_;std::map<std::string,Cache> cache_;
};
} // namespace g3rt
