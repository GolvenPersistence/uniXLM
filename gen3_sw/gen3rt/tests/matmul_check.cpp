#include "gen3rt/matmul.h"
#include "gen3rt/board.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
// Inject a completed-transfer read failure to ensure partial input caches
// cannot be reused on a subsequent call. Real command errors remain fatal.
struct CheckedDevice : g3rt::Device {
    g3rt::Device& inner;
    bool fail_read=false;
    explicit CheckedDevice(g3rt::Device& d):inner(d) {}
    uint64_t alloc(size_t n) override {return inner.alloc(n);}
    void write(uint64_t a,const void* p,size_t n) override {inner.write(a,p,n);}
    void read(uint64_t a,void* p,size_t n) override {
        if(fail_read){fail_read=false;throw std::runtime_error("injected read failure");}
        inner.read(a,p,n);
    }
    void submit(const std::vector<g3rt::Cmd128>& c) override {inner.submit(c);}
    void sync() override {inner.sync();auto c=inner.take_completions();cpl_.insert(cpl_.end(),c.begin(),c.end());}
    std::string name() const override {return inner.name();}
};
int main(int argc,char** argv) {
    if(argc!=5) {std::fprintf(stderr,"usage: matmul_check emu|board small|qwen cache_mib output_prefix\n");return 2;}
    try {
        std::unique_ptr<g3rt::Device> dev;
        if(std::string(argv[1])=="board") dev.reset(new g3rt::BoardDevice()); else dev.reset(new g3rt::EmuDevice());
        CheckedDevice checked(*dev);
        g3rt::Matmul mm(checked,size_t(std::stoul(argv[3]))<<20);
        struct Shape {int m,n,k;};
        const std::vector<Shape> small={{1,64,64},{9,1088,32},{33,128,96},{1,64,8192},{33,16384,32}};
        const std::vector<Shape> qwen={{1,4864,896},{1,896,4864},{32,128,896}};
        const auto& cases=std::string(argv[2])=="qwen"?qwen:small;
        std::mt19937 rng(1234);
        for(size_t ci=0;ci<cases.size();++ci) {
            auto s=cases[ci];std::vector<float> w(size_t(s.n)*s.k),x(size_t(s.m)*s.k),y(size_t(s.m)*s.n);
            for(auto& v:w)v=(int(rng()%2001)-1000)/1000.0f/std::sqrt(float(s.k));
            for(auto& v:x)v=(int(rng()%2001)-1000)/1000.0f;
            auto weights=mm.prepare(w.data(),s.n,s.k);mm.run(weights,x.data(),s.m,y.data());
            double err=0,ref2=0,y2=0,dot=0;bool finite=true;
            for(int r=0;r<s.m;++r)for(int c=0;c<s.n;++c) {
                double ref=0;for(int k=0;k<s.k;++k)ref+=double(x[size_t(r)*s.k+k])*w[size_t(c)*s.k+k];
                double o=y[size_t(r)*s.n+c];finite &= std::isfinite(o);err+=(o-ref)*(o-ref);ref2+=ref*ref;y2+=o*o;dot+=ref*o;
            }
            double rel=std::sqrt(err/ref2),cos=dot/std::sqrt(ref2*y2);
            std::printf("{\"case\":%zu,\"m\":%d,\"n\":%d,\"k\":%d,\"finite\":%s,\"relative_l2\":%.9g,\"cosine\":%.9g,\"cached\":%s}\n",ci,s.m,s.n,s.k,finite?"true":"false",rel,cos,weights.address?"true":"false");std::fflush(stdout);
            std::ofstream out(std::string(argv[4])+"_"+std::to_string(ci)+".bin",std::ios::binary);out.write((char*)y.data(),y.size()*4);
            if(!finite || rel>0.08 || cos<0.99) return 1;
            const auto original=y;
            auto hits=mm.input_reuse_hits;
            mm.run(weights,x.data(),s.m,y.data());
            if(y!=original || mm.input_reuse_hits!=hits+1) return 3;
            // Same allocation, different contents must not reuse stale codes.
            for(auto& v:x)v=-v;
            mm.run(weights,x.data(),s.m,y.data());
            if(mm.input_reuse_hits!=hits+1) return 4;
            for(size_t j=0;j<y.size();++j) if(y[j]!=-original[j]) return 5;
            for(auto& v:x)v=-v;
            mm.run(weights,x.data(),s.m,y.data());
            if(y!=original) return 6;
        }
        if(std::string(argv[2])=="small") {
            std::vector<float> w(64*32,0.03125f), x(33*32,0.5f), y(33*64);
            auto weight=mm.prepare(w.data(),64,32); checked.fail_read=true;
            bool threw=false;
            try {mm.run(weight,x.data(),33,y.data());} catch(const std::runtime_error& e) {
                threw=std::string(e.what())=="injected read failure";
            }
            if(!threw)return 7;
            const auto hits=mm.input_reuse_hits;
            mm.run(weight,x.data(),33,y.data());
            if(mm.input_reuse_hits!=hits)return 8;
            for(float v:y)if(v!=0.5f)return 9;
            std::printf("PASS partial_input_cache_recovery\n");
        }
        std::printf("PASS ops=%llu tiles=%llu streamed_bytes=%llu\n",(unsigned long long)mm.operations,(unsigned long long)mm.tiles,(unsigned long long)mm.streamed_bytes);
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
