#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-gen3.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
int main(int argc,char**) {
 setenv("GGML_GEN3_MODE","hybrid",1);setenv("GGML_GEN3_ATTN","1",1);
 auto be=ggml_backend_gen3_init();auto cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,4);auto dev=ggml_backend_get_device(be);
 auto ctx=ggml_init({1<<20,nullptr,true});
 auto qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,64,14,9);auto q=ggml_permute(ctx,qb,0,2,1,3);
 auto kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,64,2,32);auto k=ggml_permute(ctx,kb,0,2,1,3);
 auto vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,argc>1?32:64,argc>1?64:2,argc>1?2:32);auto v=argc>1?ggml_permute(ctx,vb,1,0,2,3):ggml_permute(ctx,vb,0,2,1,3);
 auto mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,32,64);
 auto y=ggml_flash_attn_ext(ctx,q,k,v,mask,0.125f,0,0);ggml_set_name(y,"contract-attn");
 auto vc=ggml_cont(ctx,v);auto reference=ggml_flash_attn_ext(ctx,q,k,vc,mask,0.125f,0,0);
 auto bad=ggml_flash_attn_ext(ctx,q,k,v,mask,0.5f,0,0);
 if(!ggml_backend_dev_supports_op(dev,y)||ggml_backend_dev_supports_op(dev,bad))return 1;
 setenv("GGML_GEN3_ATTN","0",1);if(ggml_backend_dev_supports_op(dev,y))return 2;setenv("GGML_GEN3_ATTN","1",1);
 auto buf=ggml_backend_alloc_ctx_tensors(ctx,be);
 for(int i=0;i<64*14*9;++i)((float*)qb->data)[i]=(i%23-11)*0.03f;
 std::vector<float> f(64*2*32);for(size_t i=0;i<f.size();++i)f[i]=(int(i%19)-9)*0.02f;
 ggml_fp32_to_fp16_row(f.data(),(ggml_fp16_t*)kb->data,f.size());ggml_fp32_to_fp16_row(f.data(),(ggml_fp16_t*)vb->data,f.size());
 auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
 auto refgraph=ggml_new_graph(ctx);ggml_build_forward_expand(refgraph,reference);
 for(int row=0;row<64;++row)for(int j=0;j<32;++j)((uint16_t*)mask->data)[row*32+j]=j<=row?0:0xFC00;
 if(ggml_backend_graph_compute(be,graph)!=GGML_STATUS_SUCCESS)return 3;
 for(int i=0;i<64*14*9;++i)if(!std::isfinite(((float*)y->data)[i]))return 4;
 // Noncausal and hole masks must use the original CPU operation exactly.
 for(int mode=0;mode<2;++mode) {
  std::memset(mask->data,0,ggml_nbytes(mask));if(mode)((uint16_t*)mask->data)[2]=0xFC00;
  if(ggml_backend_graph_compute(cpu,refgraph)!=GGML_STATUS_SUCCESS)return 5;
  std::vector<char> expected(ggml_nbytes(y));std::memcpy(expected.data(),reference->data,expected.size());
  if(ggml_backend_graph_compute(be,graph)!=GGML_STATUS_SUCCESS || std::memcmp(expected.data(),y->data,expected.size()))return 6;
 }
 ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(cpu);ggml_backend_free(be);
 puts("PASS strided QKV, scale rejection, disable switch, and bitexact CPU fallback for noncausal/hole masks");
}
