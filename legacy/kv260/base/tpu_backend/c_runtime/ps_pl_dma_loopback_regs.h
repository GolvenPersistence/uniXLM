#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOPBACK_REG_CTRL      0x00u
#define LOOPBACK_REG_STATUS    0x04u
#define LOOPBACK_REG_RX_COUNT  0x08u
#define LOOPBACK_REG_TX_COUNT  0x0Cu
#define LOOPBACK_REG_LAST_WORD 0x10u
#define LOOPBACK_REG_VERSION   0x14u

#define LOOPBACK_CTRL_ENABLE     (1u << 0)
#define LOOPBACK_CTRL_SOFT_RESET (1u << 1)

#define LOOPBACK_STATUS_ENABLE      (1u << 0)
#define LOOPBACK_STATUS_INPUT_READY (1u << 1)
#define LOOPBACK_STATUS_OUT_VALID   (1u << 2)

static inline uint32_t loopback_version_expected(void) {
    return 0x44504C31u; /* "DPL1" */
}

#ifdef __cplusplus
}
#endif
