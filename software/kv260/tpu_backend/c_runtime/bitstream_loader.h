#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int fpga_load_bitstream_fpgautil(const char *bit_path, const char *dtbo_path);
int fpga_load_bitstream_fpga_manager(const char *firmware_name, uint32_t flags);
int fpga_install_firmware_file(const char *src_path, const char *firmware_name);

#ifdef __cplusplus
}
#endif
