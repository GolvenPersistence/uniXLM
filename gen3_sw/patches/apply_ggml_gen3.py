"""Hook the Gen3 backend (gen3_sw/ggml-gen3, gen3_sw/gen3rt, gen3_sw/emu) into a llama.cpp tree of the AICAS base
(ggml 0.9.4). Idempotent.

    python gen3_sw/patches/apply_ggml_gen3.py [gen3_sw/llama.cpp]

then configure with -DGGML_GEN3=ON (and -DGGML_GEN3_BOARD=ON on PYNQ-ZU, which links XRT).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SW = Path(__file__).resolve().parents[1]
tree = Path(sys.argv[1]) if len(sys.argv) > 1 else SW / "llama.cpp"
ggml = tree / "ggml"


def edit(path: Path, anchor: str, text: str, after: bool = True) -> None:
    s = path.read_text(encoding="utf-8")
    if text.strip() in s:
        return
    assert anchor in s, (path, anchor)
    s = s.replace(anchor, anchor + text if after else text + anchor, 1)
    path.write_text(s, encoding="utf-8")


shutil.copyfile(SW / "ggml-gen3" / "ggml-gen3.h", ggml / "include" / "ggml-gen3.h")
d = ggml / "src" / "ggml-gen3"
d.mkdir(exist_ok=True)
(d / "CMakeLists.txt").write_text(f"""# Gen3 Node v3 backend: sources in gen3_sw (written by gen3_sw/patches/apply_ggml_gen3.py)
set(GEN3_SW "{SW.as_posix()}")
set(GEN3_NUMERIC_SOURCES
    ${{GEN3_SW}}/emu/src/mxfp8.cpp ${{GEN3_SW}}/emu/src/gemm.cpp ${{GEN3_SW}}/emu/src/epilogue.cpp
    ${{GEN3_SW}}/emu/src/fold.cpp ${{GEN3_SW}}/emu/src/attention.cpp ${{GEN3_SW}}/emu/src/node.cpp
    ${{GEN3_SW}}/emu/src/program.cpp
    ${{GEN3_SW}}/gen3rt/src/cmd.cpp ${{GEN3_SW}}/gen3rt/src/layout.cpp ${{GEN3_SW}}/gen3rt/src/device_emu.cpp
    ${{GEN3_SW}}/gen3rt/src/llm.cpp ${{GEN3_SW}}/gen3rt/src/weights.cpp ${{GEN3_SW}}/gen3rt/src/vit.cpp ${{GEN3_SW}}/gen3rt/src/profile.cpp)
if (GGML_GEN3_BOARD)
    list(APPEND GEN3_NUMERIC_SOURCES ${{GEN3_SW}}/gen3rt/src/device_board.cpp)
endif()
ggml_add_backend_library(ggml-gen3 ${{GEN3_SW}}/ggml-gen3/ggml-gen3.cpp ${{GEN3_NUMERIC_SOURCES}})
# Optional CPU conversion workers; without OpenMP the conversion stays serial.
if (GGML_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(ggml-gen3 PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
target_include_directories(ggml-gen3 PRIVATE ${{GEN3_SW}}/ggml-gen3 ${{GEN3_SW}}/gen3rt/include ${{GEN3_SW}}/emu/include
                           ${{GEN3_SW}}/emu/src)
# the Node's numerics: IEEE binary32 without FMA contraction
set_source_files_properties(${{GEN3_NUMERIC_SOURCES}} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
if (GGML_GEN3_BOARD)
    target_compile_definitions(ggml-gen3 PRIVATE G3RT_BOARD)
    target_include_directories(ggml-gen3 PRIVATE /usr/include/xrt)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(ggml-gen3 PRIVATE xrt_coreutil OpenSSL::Crypto)
endif()
""", encoding="utf-8")
edit(ggml / "CMakeLists.txt",
     'option(GGML_FPGA                            "ggml: use mock FPGA backend"                     OFF)',
     '\noption(GGML_GEN3                            "ggml: use the Gen3 Node v3 backend"              OFF)'
     '\noption(GGML_GEN3_BOARD                      "ggml: Gen3 backend on PYNQ-ZU (XRT)"             OFF)')
edit(ggml / "src" / "CMakeLists.txt", "ggml_add_backend(FPGA)", "\nggml_add_backend(GEN3)")
edit(ggml / "src" / "ggml-backend-reg.cpp", '#ifdef GGML_USE_FPGA\n#include "ggml-fpga/ggml-fpga.h"\n#endif\n',
     '\n#ifdef GGML_USE_GEN3\n#include "ggml-gen3.h"\n#endif\n')
edit(ggml / "src" / "ggml-backend-reg.cpp", "    ggml_backend_registry() {\n",
     "#ifdef GGML_USE_GEN3\n        register_backend(ggml_backend_gen3_reg());\n#endif\n")
print("ggml-gen3 hooked into", tree)
