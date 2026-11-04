#include "gen3rt/attention.h"
#include "gen3rt/board.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
int main(int argc,char** argv) {
 if(argc!=3)return 2;
 try {
  std::unique_ptr<g3rt::Device> dev;
  if(std::string(argv[1])=="board")dev.reset(new g3rt::BoardDevice());else dev.reset(new g3rt::EmuDevice());
  g3rt::Attention attn(*dev);std::mt19937 rng(987);
  struct Shape{int m,k,qh,kh,d;};int ci=0;
  for(auto s:std::vector<Shape>{{1,1,14,2,64},{9,33,14,2,64},{33,97,14,2,64},{1,129,14,2,64},{9,65,16,4,128},{3,35,3,1,32},{129,257,14,2,64}}) {
   std::vector<float> q(size_t(s.qh)*s.m*s.d),k(size_t(s.kh)*s.k*s.d),v(k.size()),out(q.size());
   for(auto* a:{&q,&k,&v})for(float& x:*a)x=(int(rng()%2001)-1000)/1000.f;
   const auto id=std::to_string(s.kh)+"_"+std::to_string(s.d);
   attn.run(id,q.data(),k.data(),v.data(),s.m,s.k,s.qh,s.kh,s.d,s.k-s.m,out.data());
   double err=0,ref2=0,dot=0,out2=0;
   for(int h=0;h<s.qh;++h)for(int row=0;row<s.m;++row) {
    const int kh=h/(s.qh/s.kh),visible=s.k-s.m+row+1;std::vector<double> p(visible);double total=0;
    for(int j=0;j<visible;++j){double sum=0;for(int d=0;d<s.d;++d)sum+=double(q[(size_t(h)*s.m+row)*s.d+d])*k[(size_t(kh)*s.k+j)*s.d+d];total+=(p[j]=std::exp(sum/std::sqrt(double(s.d))));}
    for(int d=0;d<s.d;++d){double ref=0;for(int j=0;j<visible;++j)ref+=p[j]/total*v[(size_t(kh)*s.k+j)*s.d+d];double y=out[(size_t(row)*s.qh+h)*s.d+d];if(!std::isfinite(y))return 3;err+=(y-ref)*(y-ref);ref2+=ref*ref;dot+=y*ref;out2+=y*y;}
   }
   const double rel=std::sqrt(err/ref2),cos=dot/std::sqrt(ref2*out2);
   printf("case=%d m=%d keys=%d qh=%d kvh=%d d=%d relative_l2=%.9g cosine=%.9g\n",ci,s.m,s.k,s.qh,s.kh,s.d,rel,cos);fflush(stdout);
   if(rel>0.15 || cos<0.99)return 4;
   std::ofstream file(std::string(argv[2])+"_"+std::to_string(ci)+".bin",std::ios::binary);file.write((char*)out.data(),out.size()*4);
   const auto expected=out;auto uploaded=attn.uploaded_bands;
   attn.run(id,q.data(),k.data(),v.data(),s.m,s.k,s.qh,s.kh,s.d,s.k-s.m,out.data());
   if(out!=expected || attn.uploaded_bands!=uploaded)return 5;
   for(float& x:v)x=-x;
   attn.run(id,q.data(),k.data(),v.data(),s.m,s.k,s.qh,s.kh,s.d,s.k-s.m,out.data());
   for(size_t j=0;j<out.size();++j)if(out[j]!=-expected[j])return 6;
   ++ci;
  }
  printf("PASS cases=%d ops=%llu commands=%llu\n",ci,(unsigned long long)attn.operations,(unsigned long long)attn.commands);
 }catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
