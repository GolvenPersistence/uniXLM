#include "gen3rt/matmul.h"
#include "gen3rt/board.h"
#include <memory>
#include <random>
#include <fstream>
#include <cmath>
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 try {
  std::unique_ptr<g3rt::Device> dev;
  if(std::string(argv[1])=="board")dev.reset(new g3rt::BoardDevice());else dev.reset(new g3rt::EmuDevice());
  g3rt::Matmul mm(*dev,64u<<20);std::mt19937 rng(4913);
  struct Shape{int m,n,k;};std::vector<Shape> shapes={{1,64,128},{9,128,4224},{33,64,8192},{32,896,4864},{32,1024,8192}};
  for(size_t ci=0;ci<shapes.size();++ci) {
   mm.reset_cache();auto s=shapes[ci];std::vector<float> w(size_t(s.n)*s.k),x(size_t(s.m)*s.k),y(size_t(s.m)*s.n),ref(y.size());
   for(auto& a:w)a=(int(rng()%2001)-1000)/1000.f/std::sqrt(float(s.k));
   for(auto& a:x)a=(int(rng()%2001)-1000)/250.f;
   for(size_t j=0;j<x.size();j+=97)x[j]=-0.0f;
   auto weight=mm.prepare(w.data(),s.n,s.k);
   for(int repeat=0;repeat<2;++repeat) {
    if(repeat)x[0]+=.25f;
    mm.run(weight,x.data(),s.m,ref.data());mm.run_online_quantized(weight,x.data(),s.m,y.data());
    if(std::memcmp(y.data(),ref.data(),y.size()*4)) {int n=0;double mx=0;for(size_t j=0;j<y.size();++j){n+=y[j]!=ref[j];mx=std::max(mx,double(std::abs(y[j]-ref[j])));}std::fprintf(stderr,"DIFF case=%zu n=%d max=%g\n",ci,n,mx);return 3;}
   }
   std::ofstream out(std::string(argv[2])+"_"+std::to_string(ci)+".bin",std::ios::binary);out.write((char*)y.data(),y.size()*4);
   std::printf("PASS case=%zu m=%d n=%d k=%d online/CPU quant bitexact; changed input exact\n",ci,s.m,s.n,s.k);std::fflush(stdout);
  }
 }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
