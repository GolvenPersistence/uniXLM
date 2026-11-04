// ggml backend of the Gen3 Node v3 (gen3_sw/ggml-gen3): the whole SmolVLM2 text model per graph (docs/gen3/node_v3.md
// s6), on the PYNQ-ZU board (GGML_GEN3_DEVICE=board) or the bit-exact emulator (default).
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_gen3_reg(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_gen3_init(void);

#ifdef __cplusplus
}
#endif
