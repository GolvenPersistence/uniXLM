option(FPGA_USE_MOCK "Use CPU mock TPU instead of real hardware" ON)

set(GGML_FPGA_SOURCES
    ggml-fpga.cpp
    ggml-fpga.h
    tpu-dispatch.h
    tpu-tiling.h
)

if(FPGA_USE_MOCK)
    list(APPEND GGML_FPGA_SOURCES
        mock-tpu.cpp
        mock-tpu.h
    )
else()
    list(APPEND GGML_FPGA_SOURCES
        real-tpu.cpp
        real-tpu.h
    )
endif()

ggml_add_backend_library(ggml-fpga ${GGML_FPGA_SOURCES})

if(FPGA_USE_MOCK)
    target_compile_definitions(ggml-fpga PRIVATE GGML_FPGA_USE_MOCK=1)
else()
    target_compile_definitions(ggml-fpga PRIVATE GGML_FPGA_USE_REAL=1)

    if(NOT DEFINED TPU_RUNTIME_DIR)
        if(EXISTS "${CMAKE_SOURCE_DIR}/tpu_backend/c_runtime")
            set(TPU_RUNTIME_DIR "${CMAKE_SOURCE_DIR}/tpu_backend/c_runtime")
            message(STATUS "Using bundled TPU runtime from ${TPU_RUNTIME_DIR}")
        else()
            message(FATAL_ERROR "FPGA_USE_MOCK=OFF requires -DTPU_RUNTIME_DIR=/path/to/c_runtime")
        endif()
    endif()

    target_include_directories(ggml-fpga PRIVATE ${TPU_RUNTIME_DIR})
    target_sources(ggml-fpga PRIVATE
        ${TPU_RUNTIME_DIR}/tpu_soc_v2_int8.c
        ${TPU_RUNTIME_DIR}/tpu_soc_fp32.c
        ${TPU_RUNTIME_DIR}/tpu_r5_ctrl.c
        ${TPU_RUNTIME_DIR}/axil_mmio.c
        ${TPU_RUNTIME_DIR}/xrt_cpp_bridge.cpp
    )

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(XRT REQUIRED xrt)

    target_include_directories(ggml-fpga PRIVATE ${XRT_INCLUDE_DIRS})
    target_compile_options(ggml-fpga PRIVATE ${XRT_CFLAGS_OTHER})
    target_link_directories(ggml-fpga PRIVATE ${XRT_LIBRARY_DIRS})
    target_link_libraries(ggml-fpga PRIVATE ${XRT_LIBRARIES})
    target_link_options(ggml-fpga PRIVATE ${XRT_LDFLAGS_OTHER})
endif()
