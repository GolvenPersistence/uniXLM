#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *xrt_cpp_device_handle_t;
typedef void *xrt_cpp_bo_handle_t;

xrt_cpp_device_handle_t xrt_cpp_device_open(unsigned device_index);
void xrt_cpp_device_close(xrt_cpp_device_handle_t handle);

xrt_cpp_bo_handle_t xrt_cpp_bo_alloc(xrt_cpp_device_handle_t device,
                                     size_t size,
                                     unsigned mem_group,
                                     int cacheable);
void *xrt_cpp_bo_map(xrt_cpp_bo_handle_t bo);
uint64_t xrt_cpp_bo_address(xrt_cpp_bo_handle_t bo);
int xrt_cpp_bo_sync_to_device(xrt_cpp_bo_handle_t bo, size_t size, size_t offset);
int xrt_cpp_bo_sync_from_device(xrt_cpp_bo_handle_t bo, size_t size, size_t offset);
void xrt_cpp_bo_free(xrt_cpp_bo_handle_t bo);

const char *xrt_cpp_last_error(void);

#ifdef __cplusplus
}
#endif
