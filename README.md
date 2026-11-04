# uniXLM

uniXLM 是面向文本与视觉语言模型的 FPGA 推理系统，包含计算与控制 RTL、命令级仿真器、模型运行时、ggml 后端和板端应用。

系统通过共享矩阵计算资源执行线性投影和注意力乘积，结合片上结果处理、缓冲管理及模型执行程序，支持语言模型推理与图文问答。处理器运行模型框架、组织任务并完成采样，FPGA 承担主要计算。

## 目录

```text
gen3_rtl/
  core/                   矩阵计算、向量处理及片上缓冲
  node/                   节点控制和任务定序
  noc/                    数据传输接口
  memory/                 存储模块
  include/                参数与数值函数
  filelists/              RTL 文件清单
gen3_sw/
  emu/                    命令级数值仿真器与核函数检查工具
  gen3rt/                 运行时、模型执行和测试程序
  ggml-gen3/              ggml 后端及后端测试
  patches/                llama.cpp 接入与 CPU 适配补丁
  board/                  板端加载、推理与评测入口
software/
  kv260/                  KV260 模型框架与加速后端
  production/             板端推理实现
  components/             归一化融合与图像编码缓存实现
tools/                    构建辅助工具
```

## 环境要求

- 软件构建：Python 3.10+、C++17 编译器、GNU Make 和 CMake。
- 板端运行：与目标平台匹配的 PYNQ/XRT、OpenSSL、FPGA overlay 及设备驱动。
- 硬件构建：RTL 文件清单中的 `rtl/core/rc/`、`rtl/core/sfu/` 运算模块，以及配套 FPGA 工程和时序约束。
- 模型运行：模型权重、视觉投影器和数据集按任务配置。

## 构建

在具备 C++17 编译器与 GNU Make 的 Linux 环境构建命令级仿真器：

```sh
make -C gen3_sw/emu -j4
```

准备 Gen3 的 llama.cpp 工作目录并构建模型后端和运行时：

```sh
python3 tools/prepare_llama.py
cmake -S gen3_sw/llama.cpp -B gen3_sw/llama.cpp-build-x86 \
  -DGGML_GEN3=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build gen3_sw/llama.cpp-build-x86 -j4
make -C gen3_sw/gen3rt LLAMA=../llama.cpp LLAMA_BUILD=../llama.cpp-build-x86 -j4
```

PYNQ-ZU 板端构建：

```sh
cmake -S gen3_sw/llama.cpp -B gen3_sw/llama.cpp-build-board \
  -DGGML_GEN3=ON -DGGML_GEN3_BOARD=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build gen3_sw/llama.cpp-build-board -j4
make -C gen3_sw/gen3rt BOARD=1 LLAMA=../llama.cpp LLAMA_BUILD=../llama.cpp-build-board -j4
```

## 运行

板端入口位于 `gen3_sw/board/`，运行前设置 overlay、模型和执行文件路径。Gen3 与 KV260 分别使用对应的设备协议、数值配置和硬件镜像，构建选项及调用方式见各模块 README。

第三方源码的许可文件与版权声明保留在对应目录，见 [第三方组件](THIRD_PARTY_NOTICES.md)。
