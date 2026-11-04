#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd;
    char dev_path[128];
    char sysfs_name[64];
    uint64_t phys_addr;
    size_t size;
    void *mapping;
} udmabuf_t;

int udmabuf_open(udmabuf_t *buf, const char *device_spec);
void udmabuf_close(udmabuf_t *buf);
int udmabuf_sync_for_device(udmabuf_t *buf);
int udmabuf_sync_for_cpu(udmabuf_t *buf);

#ifdef __cplusplus
}
#endif
