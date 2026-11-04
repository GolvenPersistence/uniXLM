#include "gen3rt/matmul.h"
#include "gen3rt/board.h"
#include <memory>
#include <random>
#include <limits>
#include <fstream>
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 try {
  std::unique_ptr<g3rt::Device> dev;
  if(std::string(argv[1])=="board")dev.reset(new g3rt::BoardDevice());else dev.reset(new g3rt::EmuDevice());
  g3rt::Matmul mm(*dev,8u<<20);std::mt19937 rng(631);const int m=65,n=64,k=4864;
  std::vector<float> w(n*k),x(m*k),prepared(x.size(),std::numeric_limits<float>::quiet_NaN()),ref(m*n),out(ref.size());
  for(auto* v:{&w,&x})for(auto& a:*v)a=(int(rng()%2001)-1000)/1000.f;
  auto weight=mm.prepare(w.data(),n,k);mm.run(weight,x.data(),m,ref.data());
  int calls=0;
  auto prepare=[&](int row,int rows){if(row!=calls*32)throw std::runtime_error("callback row order");++calls;std::memcpy(prepared.data()+size_t(row)*k,x.data()+size_t(row)*k,size_t(rows)*k*4);};
  mm.run_online_quantized(weight,prepared.data(),m,out.data(),prepare);
  if(calls!=3 || std::memcmp(out.data(),ref.data(),out.size()*4))return 3;
  bool caught=false;
  try {mm.run_online_quantized(weight,prepared.data(),m,out.data(),[](int row,int){if(row)throw std::runtime_error("injected CPU failure");});}
  catch(const std::runtime_error& e){caught=std::string(e.what())=="injected CPU failure";}
  if(!caught)return 4;
  // The previous tile must have retired before another command stream starts.
  calls=0;std::fill(prepared.begin(),prepared.end(),std::numeric_limits<float>::quiet_NaN());
  mm.run_online_quantized(weight,prepared.data(),m,out.data(),prepare);
  if(calls!=3 || std::memcmp(out.data(),ref.data(),out.size()*4))return 5;
  std::ofstream f(argv[2],std::ios::binary);f.write((char*)out.data(),out.size()*4);
  std::puts("PASS ordered CPU preparation, 65-row tail, static zero reuse, in-flight exception drain and recovery; bitexact");
 }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
