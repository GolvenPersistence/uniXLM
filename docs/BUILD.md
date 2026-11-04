# 构建与运行

## 1. 环境

通用构建需要 Python 3.10 及以上、C++17 编译器、GNU Make 和 CMake。数值仿真代码要求 IEEE binary32，禁止 fast-math 与浮点乘加收缩；原 Makefile 已包含 `-ffp-contract=off -fno-fast-math`。

PYNQ-ZU 板端执行另外需要匹配的 Linux、PYNQ、XRT 开发库、OpenSSL 开发库、设备 overlay，以及模型 GGUF 和视觉投影权重。Cortex-A53 可使用四线程运行。

## 2. 命令级仿真器

```sh
make -C gen3_sw/emu -j4
gen3_sw/emu/build/g3emu run CASE_DIR --out results/case
```

`CASE_DIR` 需要命令程序和输入存储镜像。`gen3_sw/emu/cases/` 保留一个专项用例，其他 README 中提及的 `tb/gen3/`、Python oracle 和完整 RTL 回归集未包含在归档中。原始 `emu/out/` 是生成结果，未纳入源码分发。

## 3. Gen3 的 llama.cpp 后端

```sh
python3 tools/prepare_llama.py
cmake -S gen3_sw/llama.cpp -B gen3_sw/llama.cpp-build-x86 \
  -DGGML_GEN3=ON -DLLAMA_CURL=OFF
cmake --build gen3_sw/llama.cpp-build-x86 -j4
make -C gen3_sw/gen3rt LLAMA=../llama.cpp LLAMA_BUILD=../llama.cpp-build-x86 -j4
```

准备工具复制 `legacy/kv260/base/` 并运行原后端接入补丁。它要求目标目录尚不存在，保留原基线。可选 `--cpu-a53` 接入随附 CPU 补丁；该补丁针对特定基线和 A53 部署场景，见 `gen3_sw/patches/CPU_A53.md`。

在 PYNQ-ZU 上构建板端后端：

```sh
cmake -S gen3_sw/llama.cpp -B gen3_sw/llama.cpp-build-board \
  -DGGML_GEN3=ON -DGGML_GEN3_BOARD=ON -DLLAMA_CURL=OFF
cmake --build gen3_sw/llama.cpp-build-board -j4
make -C gen3_sw/gen3rt BOARD=1 LLAMA=../llama.cpp LLAMA_BUILD=../llama.cpp-build-board -j4
```

`gen3_sw/board/` 保留原板端脚本，其中包含原部署目录、模型文件名和匹配 overlay 的检查条件。运行前按实际设备调整路径。`run_eval.sh` 调用的外部评测客户端和数据集未包含在这些源码归档中。

## 4. RTL

Gen3 Node 顶层为 `gen3_rtl/node/g3_node.v`，文件清单为 `gen3_rtl/filelists/g3_node.f`，头文件目录为 `gen3_rtl/include`。使用该文件清单前，需要补齐 [依赖清单](DEPENDENCIES.md) 中的运算叶模块，并提供 FPGA 顶层、时钟复位、外存互连和约束。

原 RTL README 中的工艺宏与物理分支说明属于该历史实现，不代表这些宏库或对应物理工程已随包提供。

## 5. 校验范围

随包提供文件哈希校验、Python 语法检查和外部 RTL 路径检查。历史变体恢复后逐文件核对清单哈希。整理环境未配置 C++/HDL 编译器及板端设备，未重新执行 C++ 编译、RTL 仿真、综合或板端性能测试。
