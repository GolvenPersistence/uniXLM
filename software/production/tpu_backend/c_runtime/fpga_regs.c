#include "fpga_regs.h"

void gemm_pack_desc_words(const gemm_desc_v1_t *desc, uint32_t out_words[7]) {
    out_words[0] = ((uint32_t)desc->task_id)
                 | (((uint32_t)(desc->b_int4_mode ? 1u : 0u)) << 8)
                 | (((uint32_t)desc->m) << 16);
    out_words[1] = ((uint32_t)desc->n)
                 | (((uint32_t)desc->k) << 16);
    out_words[2] = desc->a_addr;
    out_words[3] = desc->b_addr;
    out_words[4] = desc->c_addr;
    out_words[5] = ((uint32_t)desc->a_stride)
                 | (((uint32_t)desc->b_stride) << 16);
    out_words[6] = (uint32_t)desc->c_stride;
}
