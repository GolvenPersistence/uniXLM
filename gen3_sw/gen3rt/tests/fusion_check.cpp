#include "gen3rt/matmul.h"
#include "gen3rt/board.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
int main(int argc,char** argv) {
 if(argc!=3)return 2;
 try {
  std::unique_ptr<g3rt::Device> dev;
  if(std::string(argv[1])=="board")dev.reset(new g3rt::BoardDevice());else dev.reset(new g3rt::EmuDevice());
  g3rt::Matmul mm(*dev,64u<<20);std::mt19937 rng(7341);
  struct Shape{int m,e,f;};std::vector<Shape> shapes={{1,64,64},{9,64,4160},{33,128,192},{1,896,4864},{32,896,4864}};
  for(size_t ci=0;ci<shapes.size();++ci) {
   auto s=shapes[ci];std::vector<float> g(size_t(s.e)*s.f),u(g.size()),d(g.size()),x(size_t(s.m)*s.e),y(x.size()),ref(x.size());
   for(auto* v:{&g,&u})for(auto& a:*v)a=(int(rng()%2001)-1000)/1000.f/std::sqrt(float(s.e));
   for(auto& a:d)a=(int(rng()%2001)-1000)/1000.f/std::sqrt(float(s.f));
   for(auto& a:x)a=(int(rng()%2001)-1000)/1000.f;
   auto weights=mm.prepare_ffn(g.data(),u.data(),d.data(),s.e,s.f);mm.run_ffn(weights,x.data(),s.m,y.data());
   // FP32 FFN reference, including the nonlinear gate, independent of command packing.
   std::vector<float> h(size_t(s.m)*s.f);
   for(int row=0;row<s.m;++row)for(int j=0;j<s.f;++j) {
    double a=0,b=0;for(int k=0;k<s.e;++k){a+=double(x[size_t(row)*s.e+k])*g[size_t(j)*s.e+k];b+=double(x[size_t(row)*s.e+k])*u[size_t(j)*s.e+k];}
    h[size_t(row)*s.f+j]=float(a/(1+std::exp(-a))*b);
   }
   double err=0,rr=0,yy=0,dot=0;
   for(int row=0;row<s.m;++row)for(int j=0;j<s.e;++j) {
    double a=0;for(int k=0;k<s.f;++k)a+=double(h[size_t(row)*s.f+k])*d[size_t(j)*s.f+k];
    double b=y[size_t(row)*s.e+j];if(!std::isfinite(b))return 3;err+=(a-b)*(a-b);rr+=a*a;yy+=b*b;dot+=a*b;
   }
   double rel=std::sqrt(err/rr),cos=dot/std::sqrt(rr*yy);
   std::printf("case=%zu m=%d e=%d f=%d rel=%.9g cos=%.9g\n",ci,s.m,s.e,s.f,rel,cos);std::fflush(stdout);
   if(rel>.12 || cos<.99)return 4;
   std::ofstream out(std::string(argv[2])+"_"+std::to_string(ci)+".bin",std::ios::binary);out.write((char*)y.data(),y.size()*4);
   auto old=y;mm.run_ffn(weights,x.data(),s.m,y.data());if(old!=y)return 5;
  }
  // Wide vocabulary output exercises one submission, adaptive M tiling, and scatter.
  for(int m:{1,3}) {
   int n=151936,k=32;std::vector<float>w(size_t(n)*k,0.03125f),x(size_t(m)*k,.5f),y(size_t(m)*n);
   auto weight=mm.prepare(w.data(),n,k);mm.run(weight,x.data(),m,y.data());for(float a:y)if(a!=.5f)return 6;
  }
  std::puts("PASS fused FFN repeat + wide vocabulary");
 }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
