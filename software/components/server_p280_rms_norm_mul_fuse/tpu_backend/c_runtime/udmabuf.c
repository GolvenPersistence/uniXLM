#define _POSIX_C_SOURCE 200809L

#include "udmabuf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int build_paths(const char *device_spec, char dev_path[128], char sysfs_name[64]) {
    const char *name = device_spec;

    if (device_spec == NULL || device_spec[0] == '\0') {
        errno = EINVAL;
        return -1;
    }

    if (strncmp(device_spec, "/dev/", 5) == 0) {
        const char *slash = strrchr(device_spec, '/');
        if (slash == NULL || slash[1] == '\0') {
            errno = EINVAL;
            return -1;
        }
        name = slash + 1;
        snprintf(dev_path, 128, "%s", device_spec);
    } else {
        snprintf(dev_path, 128, "/dev/%s", device_spec);
    }

    snprintf(sysfs_name, 64, "%s", name);
    return 0;
}

static int read_u64_file(const char *path, uint64_t *value) {
    FILE *fp;
    char buf[64];

    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    if (fgets(buf, sizeof(buf), fp) == NULL) {
        fclose(fp);
        errno = EIO;
        return -1;
    }
    fclose(fp);

    errno = 0;
    *value = strtoull(buf, NULL, 0);
    if (errno != 0) {
        return -1;
    }
    return 0;
}

static int write_sync_file(const char *sysfs_name, const char *entry) {
    char path[PATH_MAX];
    int fd;
    const char payload[] = "1\n";
    ssize_t wr;

    snprintf(path, sizeof(path), "/sys/class/u-dma-buf/%s/%s", sysfs_name, entry);
    fd = open(path, O_WRONLY);
    if (fd < 0) {
        if (errno == ENOENT) {
            return 0;
        }
        return -1;
    }
    wr = write(fd, payload, sizeof(payload) - 1);
    close(fd);
    if (wr < 0) {
        return -1;
    }
    return 0;
}

int udmabuf_open(udmabuf_t *buf, const char *device_spec) {
    char phys_path[PATH_MAX];
    char size_path[PATH_MAX];
    uint64_t size_u64;

    if (buf == NULL) {
        errno = EINVAL;
        return -1;
    }

    memset(buf, 0, sizeof(*buf));
    buf->fd = -1;

    if (build_paths(device_spec, buf->dev_path, buf->sysfs_name) != 0) {
        return -1;
    }

    snprintf(phys_path, sizeof(phys_path), "/sys/class/u-dma-buf/%s/phys_addr", buf->sysfs_name);
    snprintf(size_path, sizeof(size_path), "/sys/class/u-dma-buf/%s/size", buf->sysfs_name);

    if (read_u64_file(phys_path, &buf->phys_addr) != 0) {
        return -1;
    }
    if (read_u64_file(size_path, &size_u64) != 0) {
        return -1;
    }
    buf->size = (size_t)size_u64;

    buf->fd = open(buf->dev_path, O_RDWR | O_SYNC);
    if (buf->fd < 0) {
        return -1;
    }

    buf->mapping = mmap(NULL, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED, buf->fd, 0);
    if (buf->mapping == MAP_FAILED) {
        buf->mapping = NULL;
        close(buf->fd);
        buf->fd = -1;
        return -1;
    }

    return 0;
}

void udmabuf_close(udmabuf_t *buf) {
    if (buf == NULL) {
        return;
    }
    if (buf->mapping != NULL) {
        munmap(buf->mapping, buf->size);
    }
    if (buf->fd >= 0) {
        close(buf->fd);
    }
    memset(buf, 0, sizeof(*buf));
    buf->fd = -1;
}

int udmabuf_sync_for_device(udmabuf_t *buf) {
    if (buf == NULL) {
        errno = EINVAL;
        return -1;
    }
    return write_sync_file(buf->sysfs_name, "sync_for_device");
}

int udmabuf_sync_for_cpu(udmabuf_t *buf) {
    if (buf == NULL) {
        errno = EINVAL;
        return -1;
    }
    return write_sync_file(buf->sysfs_name, "sync_for_cpu");
}
