#include "xrt_cpp_bridge.h"

#include <cerrno>
#include <cstring>
#include <new>
#include <string>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

namespace {

struct DeviceWrapper {
    explicit DeviceWrapper(unsigned index)
        : device(index) {}

    xrt::device device;
};

struct BufferWrapper {
    BufferWrapper(xrt::device &device, size_t size, unsigned mem_group, bool cacheable)
        : bo(device,
             size,
             cacheable ? xrt::bo::flags::cacheable : xrt::bo::flags::normal,
             mem_group) {}

    xrt::bo bo;
};

thread_local std::string g_last_error;

void set_error(const char *msg) {
    g_last_error = msg ? msg : "unknown error";
    errno = EIO;
}

void set_error(const std::exception &ex) {
    g_last_error = ex.what();
    errno = EIO;
}

}  // namespace

extern "C" {

xrt_cpp_device_handle_t xrt_cpp_device_open(unsigned device_index) {
    try {
        return new DeviceWrapper(device_index);
    } catch (const std::exception &ex) {
        set_error(ex);
        return nullptr;
    } catch (...) {
        set_error("xrt_cpp_device_open failed");
        return nullptr;
    }
}

void xrt_cpp_device_close(xrt_cpp_device_handle_t handle) {
    delete static_cast<DeviceWrapper *>(handle);
}

xrt_cpp_bo_handle_t xrt_cpp_bo_alloc(xrt_cpp_device_handle_t device,
                                     size_t size,
                                     unsigned mem_group,
                                     int cacheable) {
    DeviceWrapper *wrapper = static_cast<DeviceWrapper *>(device);
    if (wrapper == nullptr) {
        errno = EINVAL;
        set_error("device handle is null");
        return nullptr;
    }

    try {
        return new BufferWrapper(wrapper->device, size, mem_group, cacheable != 0);
    } catch (const std::exception &ex) {
        set_error(ex);
        return nullptr;
    } catch (...) {
        set_error("xrt_cpp_bo_alloc failed");
        return nullptr;
    }
}

void *xrt_cpp_bo_map(xrt_cpp_bo_handle_t bo) {
    BufferWrapper *wrapper = static_cast<BufferWrapper *>(bo);
    if (wrapper == nullptr) {
        errno = EINVAL;
        set_error("buffer handle is null");
        return nullptr;
    }

    try {
        return wrapper->bo.map<void *>();
    } catch (const std::exception &ex) {
        set_error(ex);
        return nullptr;
    } catch (...) {
        set_error("xrt_cpp_bo_map failed");
        return nullptr;
    }
}

uint64_t xrt_cpp_bo_address(xrt_cpp_bo_handle_t bo) {
    BufferWrapper *wrapper = static_cast<BufferWrapper *>(bo);
    if (wrapper == nullptr) {
        errno = EINVAL;
        set_error("buffer handle is null");
        return UINT64_MAX;
    }

    try {
        return wrapper->bo.address();
    } catch (const std::exception &ex) {
        set_error(ex);
        return UINT64_MAX;
    } catch (...) {
        set_error("xrt_cpp_bo_address failed");
        return UINT64_MAX;
    }
}

int xrt_cpp_bo_sync_to_device(xrt_cpp_bo_handle_t bo, size_t size, size_t offset) {
    BufferWrapper *wrapper = static_cast<BufferWrapper *>(bo);
    if (wrapper == nullptr) {
        errno = EINVAL;
        set_error("buffer handle is null");
        return -1;
    }

    try {
        wrapper->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, size, offset);
        return 0;
    } catch (const std::exception &ex) {
        set_error(ex);
        return -1;
    } catch (...) {
        set_error("xrt_cpp_bo_sync_to_device failed");
        return -1;
    }
}

int xrt_cpp_bo_sync_from_device(xrt_cpp_bo_handle_t bo, size_t size, size_t offset) {
    BufferWrapper *wrapper = static_cast<BufferWrapper *>(bo);
    if (wrapper == nullptr) {
        errno = EINVAL;
        set_error("buffer handle is null");
        return -1;
    }

    try {
        wrapper->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
        return 0;
    } catch (const std::exception &ex) {
        set_error(ex);
        return -1;
    } catch (...) {
        set_error("xrt_cpp_bo_sync_from_device failed");
        return -1;
    }
}

void xrt_cpp_bo_free(xrt_cpp_bo_handle_t bo) {
    delete static_cast<BufferWrapper *>(bo);
}

const char *xrt_cpp_last_error(void) {
    return g_last_error.empty() ? "" : g_last_error.c_str();
}

}  // extern "C"
