#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
int main() {
 auto be=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(be,4);
 std::mt19937 rng(1234);
 struct Shape{int k,n,m;};
 for(int threads: {1,3,4}) for(auto shape:std::vector<Shape>{{64,256,1},{896,1024,1},{8192,260,1},{96,256,1},{64,259,1},{128,256,2},{896,151936,1}}) {
  if(threads!=4 && shape.n==151936)continue;
  ggml_backend_cpu_set_n_threads(be,threads);
  auto cp=ggml_init_params{ggml_tensor_overhead()*8+ggml_graph_overhead_custom(16,false),nullptr,true};auto ctx=ggml_init(cp);
  auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,shape.k,shape.n),x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,shape.k,shape.m);
  auto y=ggml_mul_mat(ctx,w,x);auto graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);
  auto buffer=ggml_backend_alloc_ctx_tensors(ctx,be);if(!buffer)return 2;
  std::vector<float> floats(size_t(shape.k)*shape.m);
  for(int row=0;row<shape.n;++row) {
   for(int j=0;j<shape.k;++j)floats[j]=(int(rng()%20001)-10000)/10000.0f;
   ggml_quantize_chunk(GGML_TYPE_Q8_0,floats.data(),(char*)w->data+row*w->nb[1],0,1,shape.k,nullptr);
   // Exercise -128 weights, still safe because quantized activations exclude -128.
   if(row%7==0)std::memset((char*)w->data+row*w->nb[1]+2,0x80,32);
  }
  for(auto& v:floats)v=(int(rng()%20001)-10000)/10000.0f;
  std::memcpy(x->data,floats.data(),floats.size()*4);
  std::vector<float> baseline(size_t(shape.n)*shape.m);
  for(int repeat=0;repeat<6;++repeat) for(int mode=0;mode<2;++mode) {
   if(repeat==5)std::memset(x->data,0,floats.size()*4);
   setenv("GGML_GEN3_CPU_A53",mode?"1":"0",1);
   auto start=std::chrono::steady_clock::now();if(ggml_backend_graph_compute(be,graph)!=GGML_STATUS_SUCCESS)return 3;
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   if(mode==0)std::memcpy(baseline.data(),y->data,baseline.size()*4);
   else if(std::memcmp(baseline.data(),y->data,baseline.size()*4)) {
    double err=0;for(size_t j=0;j<baseline.size();++j)err=std::max(err,std::abs(double(baseline[j])-((float*)y->data)[j]));
    std::printf("FAIL k=%d n=%d m=%d max_error=%.9g\n",shape.k,shape.n,shape.m,err);return 4;
   }
   std::printf("GEMV threads=%d k=%d n=%d m=%d repeat=%d mode=%d ms=%.6f\n",threads,shape.k,shape.n,shape.m,repeat,mode,ms);std::fflush(stdout);
  }
  ggml_backend_buffer_free(buffer);ggml_free(ctx);
 }

 ggml_backend_free(be);puts("PASS all Q8 outputs bitexact");
}
