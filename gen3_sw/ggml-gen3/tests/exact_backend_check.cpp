#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-gen3.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
int main(int argc,char**argv) {
 if(argc!=4)return 2;
 setenv("GGML_GEN3_MODE","hybrid",1);setenv("GGML_GEN3_FFN",argv[3],1);setenv("GGML_GEN3_VOCAB","1",1);setenv("GGML_GEN3_CACHE_MIB","16",1);
 setenv("GGML_GEN3_DEVICE",argv[1],1);
 auto* be=ggml_backend_gen3_init();auto* dev=ggml_backend_get_device(be);
 for(int test=0;test<5;++test) {
  auto* ctx=ggml_init({4u<<20,nullptr,true});
  auto* x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,33);
  auto* gw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,4864);ggml_set_name(gw,"blk.0.ffn_gate.weight");
  auto* uw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,4864);ggml_set_name(uw,"blk.0.ffn_up.weight");
  auto* dw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4864,64);ggml_set_name(dw,"blk.0.ffn_down.weight");
  auto* gate=ggml_mul_mat(ctx,gw,x);auto* up=ggml_mul_mat(ctx,uw,x);auto* act=ggml_swiglu_split(ctx,gate,up);auto* out=ggml_mul_mat(ctx,dw,act);
  auto* graph=ggml_new_graph(ctx);
  if(test==1)ggml_build_forward_expand(graph,up); // opposite topological order
  if(test==2)ggml_set_output(gate);             // must materialize this output
  ggml_build_forward_expand(graph,out);
  auto* extra=test>=3?ggml_mul_mat(ctx,dw,gate):nullptr;
  if(extra)ggml_build_forward_expand(graph,extra); // additional consumer must disable fusion
  if(!ggml_backend_dev_supports_op(dev,act))return 3;
  auto* buffer=ggml_backend_alloc_ctx_tensors(ctx,be);ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  for(auto* t:{gw,uw,dw,x}) {
   std::vector<float> values(ggml_nelements(t));for(size_t j=0;j<values.size();++j)values[j]=(int(j%23)-11)*(t==x?.04f:.006f);
   ggml_backend_tensor_set(t,values.data(),0,values.size()*4);
  }
  auto view=ggml_graph_view(graph,0,test==4?4:ggml_graph_n_nodes(graph));
  if(ggml_backend_graph_compute(be,&view)!=GGML_STATUS_SUCCESS)return 4;
  for(auto* t:{out,test>=2?gate:out,extra && test==3?extra:out}) {
   std::vector<float> values(ggml_nelements(t));ggml_backend_tensor_get(t,values.data(),0,values.size()*4);
   float largest=0;for(float v:values){if(!std::isfinite(v))return 5;largest=std::max(largest,std::abs(v));}if(largest<1e-6f)return 6;
   std::ofstream f(std::string(argv[2])+"_"+std::to_string(test)+"_"+std::to_string(t==out?0:t==gate?1:2)+".bin",std::ios::binary);f.write((char*)values.data(),values.size()*4);
  }
  ggml_backend_buffer_free(buffer);ggml_free(ctx);
 }
 ggml_backend_free(be);std::puts("PASS graph order/output/consumer cases");
}
