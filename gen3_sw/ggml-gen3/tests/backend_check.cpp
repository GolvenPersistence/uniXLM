#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-gen3.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
int main() {
 setenv("GGML_GEN3_MODE","hybrid",1);
 auto* be=ggml_backend_gen3_init();auto* dev=ggml_backend_get_device(be);
 ggml_init_params ip{1<<20,nullptr,true};auto* ctx=ggml_init(ip);
 auto* w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,64);ggml_set_name(w,"blk.0.attn_q.weight");
 auto* x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,9);
 auto* mul=ggml_mul_mat(ctx,w,x);auto* add=ggml_add(ctx,x,x);auto* norm=ggml_rms_norm(ctx,x,1e-6f);
 auto* narrow=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,32);ggml_set_name(narrow,"blk.0.ffn_up.weight");
 auto* bad=ggml_mul_mat(ctx,narrow,x);
 auto* head=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,64);ggml_set_name(head,"output.weight");auto* lm=ggml_mul_mat(ctx,head,x);
 if(!ggml_backend_dev_supports_op(dev,mul)||ggml_backend_dev_supports_op(dev,add)||ggml_backend_dev_supports_op(dev,norm)||ggml_backend_dev_supports_op(dev,bad)||ggml_backend_dev_supports_op(dev,lm))return 1;
 auto* x0=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,0);auto* empty=ggml_mul_mat(ctx,w,x0);
 if(!ggml_backend_dev_supports_op(dev,empty))return 5;
 auto* graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,empty);
 if(ggml_backend_graph_compute(be,graph)!=GGML_STATUS_SUCCESS)return 6;
 auto* buf=ggml_backend_alloc_ctx_tensors(ctx,be);if(!ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buf)))return 2;
 std::vector<float> v(32*64,1.25f),out(v.size());ggml_backend_tensor_set(w,v.data(),0,v.size()*4);ggml_backend_tensor_get(w,out.data(),0,out.size()*4);if(v!=out)return 3;
 ggml_backend_buffer_clear(buf,0);ggml_backend_tensor_get(w,out.data(),0,out.size()*4);for(float f:out)if(f!=0)return 4;
 ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(be);std::puts("PASS: matmul accepted; ADD/RMS/narrow-output/vocab-head rejected; empty output is a no-op; host readback and clear correct");
}
