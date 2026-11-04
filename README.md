# uniXLM

FPGA 文本与视觉语言推理系统的旧版软硬件源码，包含 Gen3 Node RTL、命令级仿真器、模型运行时、ggml 后端，以及 KV260 历史软件基线与变体。

## 目录

```text
gen3_rtl/                 Gen3 Node RTL、头文件与综合文件清单
gen3_sw/
  emu/                    命令级数值仿真器与核函数检查工具
  gen3rt/                 运行时、模型执行和测试程序
  ggml-gen3/              ggml 后端及后端测试
  patches/                llama.cpp 接入与 CPU 适配补丁
  board/                  板端加载、推理与评测入口
legacy/kv260/
  base/                   历史 llama.cpp 与加速后端公共基线
  variants/               三个历史版本相对基线的差异文件和恢复清单
  evidence/               随源码附带的测试记录
tools/                    源码校验、历史版本恢复与构建准备工具
docs/                     构建说明、版本关系和依赖说明
SHA256SUMS                分发文件校验值
```

## 快速开始

检查文件完整性与外部依赖：

```sh
python3 tools/check_source.py
```

在具备 C++17 编译器与 GNU Make 的 Linux 环境构建命令级仿真器：

```sh
make -C gen3_sw/emu -j4
```

使用随附历史基线准备 Gen3 的 llama.cpp 工作目录：

```sh
python3 tools/prepare_llama.py
cmake -S gen3_sw/llama.cpp -B gen3_sw/llama.cpp-build-x86 \
  -DGGML_GEN3=ON -DLLAMA_CURL=OFF
cmake --build gen3_sw/llama.cpp-build-x86 -j4
```

板端构建和运行见 [构建与运行](docs/BUILD.md)。已有源码和测试工具保留在原模块目录，具体命令见各模块 README。

## 版本与依赖

Gen3 和 KV260 历史实现分别组织，不能直接混用二者的设备协议、overlay 和数值配置。历史软件的重复文件只存储一份，其他版本可以按清单恢复，见 [版本关系](docs/VERSIONS.md)。

Gen3 RTL 文件清单引用的十个通用运算叶模块未包含在输入归档中；完整 FPGA 工程、bitstream、模型权重和数据集另行准备。历史软件输入归档存在尾部截断，其中一个变体只能恢复已读到的文件。具体缺项见 [依赖与完整性](docs/DEPENDENCIES.md)。

第三方源码的许可文件与版权声明保留在对应目录，见 [第三方组件](THIRD_PARTY_NOTICES.md)。
