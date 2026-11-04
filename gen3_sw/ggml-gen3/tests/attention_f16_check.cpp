#include "gen3rt/attention.h"
#include "gen3rt/board.h"
#include "ggml.h"
#include <memory>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
static void convert(const uint16_t* p,float* q,int n){ggml_fp16_to_fp32_row((const ggml_fp16_t*)p,q,n);}
// Simulate a partial DDR upload, then restore the original host KV content.
class FailWriteDevice : public g3rt::Device {
 public:
  explicit FailWriteDevice(g3rt::Device& dev):dev_(dev){}
  int fail_after=0;
  uint64_t alloc(size_t n) override{return dev_.alloc(n);}
  void write(uint64_t a,const void* p,size_t n) override{dev_.write(a,p,n);if(fail_after && !--fail_after)throw std::runtime_error("injected upload failure");}
  void read(uint64_t a,void* p,size_t n) override{dev_.read(a,p,n);}
  void submit(const std::vector<g3::Cmd128>& c) override{dev_.submit(c);}
  void sync() override{dev_.sync();auto c=dev_.take_completions();cpl_.insert(cpl_.end(),c.begin(),c.end());}
  std::string name() const override{return dev_.name();}
 private:g3rt::Device& dev_;
};
int main(int argc,char**argv){
 if(argc!=3)return 2;
 try {
  std::unique_ptr<g3rt::Device> dev;
  if(std::string(argv[1])=="board")dev.reset(new g3rt::BoardDevice());else dev.reset(new g3rt::EmuDevice());
  FailWriteDevice guarded(*dev);g3rt::Attention a(guarded);int checks=0;
  const int cap=160,d=64,kh=2,qh=14;
  std::vector<uint16_t> k(cap*kh*d),v(k.size()),vt(k.size());
  std::vector<float> seed(k.size());for(size_t i=0;i<seed.size();++i)seed[i]=(int(i%31)-15)*.013f;
  ggml_fp32_to_fp16_row(seed.data(),(ggml_fp16_t*)k.data(),int(k.size()));
  for(size_t i=0;i<seed.size();++i)seed[i]=(int(i%37)-18)*.019f;
  ggml_fp32_to_fp16_row(seed.data(),(ggml_fp16_t*)v.data(),int(v.size()));
  // Repeated snapshots, tail growth, 32-token boundary, rewind, regrowth,
  // same-address history edits, F32/F16 alternation and transposed V.
  const int lengths[]={31,31,32,33,65,17,66,66,129};
  std::vector<uint16_t> ki(k.size()),vi(v.size());
  for(int layout=0;layout<3;++layout)for(int ci=0;ci<9;++ci){
   int keys=lengths[ci],m=ci==0?9:1;
   if(ci==7){k[0]^=0x8000;v[7]^=0x8000;}
   for(int h=0;h<kh;++h)for(int row=0;row<cap;++row)for(int j=0;j<d;++j)vt[(h*d+j)*cap+row]=v[(h*cap+row)*d+j];
   std::vector<float> q(qh*m*d),kf(kh*keys*d),vf(kf.size()),ref(q.size()),got(q.size());
   for(size_t i=0;i<q.size();++i)q[i]=(int(i%17)-8)*.02f;
   for(int h=0;h<kh;++h)for(int row=0;row<keys;++row){convert(k.data()+(h*cap+row)*d,kf.data()+(h*keys+row)*d,d);convert(v.data()+(h*cap+row)*d,vf.data()+(h*keys+row)*d,d);}
   a.run("reference",q.data(),kf.data(),vf.data(),m,keys,qh,kh,d,keys-m,ref.data());
   g3rt::Attention::F16View kv{k.data(),2,size_t(d*2),size_t(cap*d*2)};
   g3rt::Attention::F16View vv=layout==1?g3rt::Attention::F16View{vt.data(),size_t(cap*2),2,size_t(cap*d*2)}:g3rt::Attention::F16View{v.data(),2,size_t(d*2),size_t(cap*d*2)};
   if(layout==2){
    for(int row=0;row<cap;++row)for(int h=0;h<kh;++h)for(int j=0;j<d;++j){ki[(row*kh+h)*d+j]=k[(h*cap+row)*d+j];vi[(row*kh+h)*d+j]=v[(h*cap+row)*d+j];}
    kv={ki.data(),2,size_t(kh*d*2),size_t(d*2)};vv={vi.data(),2,size_t(kh*d*2),size_t(d*2)};
   }
   const auto cv=a.f16_converted_values,up=a.uploaded_bands;
   a.run_f16("typed",q.data(),kv,vv,m,keys,qh,kh,d,keys-m,got.data(),convert);
   if(std::memcmp(ref.data(),got.data(),ref.size()*4))return 3;
   for(float x:got)if(!std::isfinite(x))return 4;
   if(ci==1 && (cv!=a.f16_converted_values || up!=a.uploaded_bands))return 5;
   const auto cv2=a.f16_converted_values;
   a.run_f16("typed",q.data(),kv,vv,m,keys,qh,kh,d,keys-m,got.data(),convert);
   if(cv2!=a.f16_converted_values || std::memcmp(ref.data(),got.data(),ref.size()*4))return 6;
   std::ofstream f(std::string(argv[2])+"_"+std::to_string(checks)+".bin",std::ios::binary);f.write((const char*)got.data(),got.size()*4);
   if(ci==8){a.run("typed",q.data(),kf.data(),vf.data(),m,keys,qh,kh,d,keys-m,got.data());a.run_f16("typed",q.data(),kv,vv,m,keys,qh,kh,d,keys-m,got.data(),convert);if(std::memcmp(ref.data(),got.data(),ref.size()*4))return 7;}
   ++checks;
   if(layout==2 && ci==8){
    ki[0]^=0x8000;ki[d]^=0x8000;guarded.fail_after=2;bool failed=false;
    try{a.run_f16("typed",q.data(),kv,vv,m,keys,qh,kh,d,keys-m,got.data(),convert);}catch(const std::runtime_error&){failed=true;}
    if(!failed)return 9;
    ki[0]^=0x8000;ki[d]^=0x8000;
    a.run_f16("typed",q.data(),kv,vv,m,keys,qh,kh,d,keys-m,got.data(),convert);
    if(std::memcmp(ref.data(),got.data(),ref.size()*4))return 10;
    std::ofstream recovered(std::string(argv[2])+"_"+std::to_string(checks)+".bin",std::ios::binary);recovered.write((const char*)got.data(),got.size()*4);++checks;
   }
  }
  if(!a.f16_prefix_hits)return 8;
  printf("PASS %d cases; checked=%llu converted=%llu prefix_hits=%llu\n",checks,(unsigned long long)a.f16_checked_values,(unsigned long long)a.f16_converted_values,(unsigned long long)a.f16_prefix_hits);
 }catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
