#include "llama.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <cmath>
int main(int argc,char**argv) {
 if(argc!=4)return 2;
 ggml_backend_load_all();llama_backend_init();
 auto mp=llama_model_default_params();mp.n_gpu_layers=std::string(argv[1])=="cpu"?0:99;
 auto* model=llama_model_load_from_file(argv[2],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.offload_kqv=false;cp.n_ctx=256;cp.n_batch=256;cp.n_ubatch=128;cp.n_threads=4;cp.n_threads_batch=4;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto* ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 const auto* vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
 const std::string prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain how a CPU and an FPGA work together in a computer. Give a short, factual answer.<|im_end|>\n<|im_start|>assistant\n";
 std::vector<llama_token> ids(256);int n=llama_tokenize(vocab,prompt.c_str(),prompt.size(),ids.data(),ids.size(),true,true);if(n<1)return 5;ids.resize(n);
 // Fixed continuation tokens: every backend sees identical contexts, independent of argmax.
 std::vector<llama_token> fixed(32);std::string continuation="The CPU manages";int nf=llama_tokenize(vocab,continuation.c_str(),continuation.size(),fixed.data(),fixed.size(),false,false);if(nf<2)return 6;
 for(int step=0;step<3;++step) {
  llama_batch batch=llama_batch_get_one(step?&fixed[step-1]:ids.data(),step?1:n);
  if(llama_decode(ctx,batch))return 7;
  const float* logits=llama_get_logits_ith(ctx,-1);int best=0;
  for(int i=0;i<nv;++i){if(!std::isfinite(logits[i]))return 8;if(logits[i]>logits[best])best=i;}
  std::ofstream out(std::string(argv[3])+"_"+std::to_string(step)+".bin",std::ios::binary);out.write((const char*)logits,nv*4);
  std::printf("step=%d prompt_tokens=%d vocab=%d argmax=%d finite=true\n",step,n+step,nv,best);std::fflush(stdout);
 }
 llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
