#include "dma_iface.h"

#include <errno.h>

int dma_iface_not_implemented(void) {
    errno = ENOSYS;
    return -1;
}
