#include "bitstream_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_text_file(const char *path, const char *value) {
    int fd;
    size_t len;
    ssize_t wr;

    fd = open(path, O_WRONLY);
    if (fd < 0) {
        return -1;
    }
    len = strlen(value);
    wr = write(fd, value, len);
    close(fd);
    if (wr < 0 || (size_t)wr != len) {
        errno = EIO;
        return -1;
    }
    return 0;
}

static int copy_file_to(const char *src_path, const char *dst_path) {
    int src_fd;
    int dst_fd;
    char buf[8192];
    ssize_t rd;

    src_fd = open(src_path, O_RDONLY);
    if (src_fd < 0) {
        return -1;
    }

    dst_fd = open(dst_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dst_fd < 0) {
        close(src_fd);
        return -1;
    }

    while ((rd = read(src_fd, buf, sizeof(buf))) > 0) {
        ssize_t off = 0;
        while (off < rd) {
            ssize_t wr = write(dst_fd, buf + off, (size_t)(rd - off));
            if (wr < 0) {
                close(dst_fd);
                close(src_fd);
                return -1;
            }
            off += wr;
        }
    }

    close(dst_fd);
    close(src_fd);

    if (rd < 0) {
        return -1;
    }
    return 0;
}

int fpga_load_bitstream_fpgautil(const char *bit_path, const char *dtbo_path) {
    char cmd[1024];

    if (bit_path == NULL || bit_path[0] == '\0') {
        return -1;
    }

    if (dtbo_path != NULL && dtbo_path[0] != '\0') {
        snprintf(cmd, sizeof(cmd), "fpgautil -b '%s' -o '%s'", bit_path, dtbo_path);
    } else {
        snprintf(cmd, sizeof(cmd), "fpgautil -b '%s'", bit_path);
    }

    return system(cmd);
}

int fpga_install_firmware_file(const char *src_path, const char *firmware_name) {
    char dst_path[1024];

    if (src_path == NULL || src_path[0] == '\0' || firmware_name == NULL || firmware_name[0] == '\0') {
        errno = EINVAL;
        return -1;
    }

    snprintf(dst_path, sizeof(dst_path), "/lib/firmware/%s", firmware_name);
    return copy_file_to(src_path, dst_path);
}

int fpga_load_bitstream_fpga_manager(const char *firmware_name, uint32_t flags) {
    char flags_buf[32];

    if (firmware_name == NULL || firmware_name[0] == '\0') {
        errno = EINVAL;
        return -1;
    }

    snprintf(flags_buf, sizeof(flags_buf), "%u\n", flags);
    if (write_text_file("/sys/class/fpga_manager/fpga0/flags", flags_buf) != 0) {
        return -1;
    }
    if (write_text_file("/sys/class/fpga_manager/fpga0/firmware", firmware_name) != 0) {
        return -1;
    }
    return 0;
}
