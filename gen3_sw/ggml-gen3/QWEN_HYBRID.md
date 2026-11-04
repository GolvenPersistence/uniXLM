# GEN3 的 Qwen / CPU 混合执行

`codex/qwen` 分支默认采用混合模式：只把支持的层内二维矩阵乘交给 GEN3，其他算子交给 llama.cpp 的 CPU 后端。硬件继续使用已验收的 R51 core 240 MHz / AXI 300 MHz 比特流。

## 执行分工

- FPGA：Q/K/V/O 投影、FFN gate/up/down 中满足形状条件的矩阵乘。输入和权重转换为 MXFP8，输出为 FP32。
- CPU：默认执行 Attention、KV cache、RoPE、归一化、bias、激活、残差、词表输出、采样，以及不满足 FPGA 准入条件的矩阵乘。下述可选路径可把 prefill 的 attention 核心交给 FPGA。
- ggml 调度器：按 `supports_op` 分图；混合模式的原始权重和张量放在主机可访问缓冲中，CPU 可以正常读取。
- GEN3 运行时：每次最多处理 32 行 × 1024 输出列，独立分配输入/输出 UB；不再调用整模型 `Llm::configure_memory()`。

这条路径不要求 Qwen 的整层布局或 GQA 比例符合旧版 Llama 程序。FFN 宽度 4864 可以分块执行；K 维暂时不分块，超过 8192 的矩阵乘自动留在 CPU。

## 启动

使用本分支的 `gen3_sw/patches/apply_ggml_gen3.py` 接入现有 llama.cpp，然后按原流程启用 `GGML_GEN3=ON`、`GGML_GEN3_BOARD=ON` 构建。`matmul.h` 是头文件实现，不需要新增 CMake 源文件。

Qwen2.5-0.5B-Instruct 示例（将模型路径替换为实际文件）：

```sh
sudo -n env XILINX_XRT=/usr OMP_NUM_THREADS=4 \
  GGML_GEN3_MODE=hybrid GGML_GEN3_DEVICE=board \
  G3RT_BOARD_POOL_MIB=384 GGML_GEN3_CACHE_MIB=360 \
  ./llama-server -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -ngl 99 --no-kv-offload -fa off -t 4 -np 1 \
  -c 1152 -b 512 -ub 512 --no-warmup
```

CPU 对照使用相同可执行文件、模型和参数，只把 `-ngl 99` 改为 `-ngl 0`。CPU 对照不会创建 BoardDevice。

`GGML_GEN3_MODE=fused` 显式选择原来的 SmolLM/SmolVLM 整图专用路径。该模式需要保持 KV offload 开启，不要传 `--no-kv-offload`；它依赖整图连续分配。旧性能测试必须指定该值，避免把新的混合路径结果混入旧表格。不要在同一进程中更改执行模式。

## 缓存和容量

`GGML_GEN3_CACHE_MIB` 默认 256，表示已转换 MXFP8 层矩阵的设备缓存预算。缓存按首次使用顺序分配，达到预算后将其余矩阵的 MXFP8 数据保留在主机，执行时逐块上传；不会静默改成 CPU 计算已接受的算子。

设备池还要容纳约 8.25 MiB 的最大权重块、2.28125 MiB 的输入/输出暂存区及命令环，缓存预算必须小于池大小。Qwen2.5-0.5B 的层矩阵可放入 360 MiB 缓存；词表权重留在 CPU。该设置不能直接推广为其他型号的常驻容量保证。

首个请求包含惰性权重转换和缓存填充，冷 TTFT 必须与热 TTFT 分开记录。`GEN3_HYBRID_SUMMARY` 输出实际矩阵乘次数、硬件 tile 数、运行中上传的流式权重字节数、峰值缓存占用、提交批次数和输入量化复用命中次数；可据此确认不是纯 CPU 回退。

## 当前准入范围

静态、连续的 `blk.*.weight` 二维矩阵，K 是 32 的倍数且不超过 8192，N 是 64 的倍数且不超过 16384；激活和结果为连续 FP32 二维张量。张量 view/reshape/permute/transpose 只作为元数据操作接受，不做数值运算。

不支持的算子在调度阶段交给 CPU。硬件执行失败会返回错误，不会隐瞒故障并把结果当作 FPGA 性能。当前验证针对单模型、单请求槽和固定权重；未验证动态 LoRA、运行中改权重、多模型并存或并发后端实例。

MXFP8 会改变数值，不能要求完整模型 logits 与 CPU Q8 逐位一致。先验证 primitive 的板端/位精确模拟器一致性，再比较完整模型 logits、分布和输出。少量 teacher-forcing 检查不能代替准确率数据集。

## 回归程序

- `gen3_sw/gen3rt/tests/matmul_check.cpp`：分块、非整 tile 行数、最大 K、Qwen FFN 形状、缓存/流式路径，比较 FP32 参考并导出二进制输出用于位精确比较。板端链接 `libgen3rt.a`、XRT、crypto、OpenMP。
- `gen3_sw/ggml-gen3/tests/backend_check.cpp`：支持算子边界、CPU 可访问缓冲、原始权重读回和清零。
- `gen3_sw/ggml-gen3/tests/qwen_check.cpp`：同一 prompt 和固定 continuation 的 CPU/混合 logits 对照。直接链接与 server 相同的 llama/ggml 静态库。

构建命令、运行环境、模型来源、原始日志和结果保存在本轮 `docs/reports/20261021_qwen_hybrid` 报告目录。测试时串行持有 `/tmp/gen3_board_accept.lock`，避免与其他下板任务争用硬件。

## 软件侧优化（2026-10-21）

混合模式的 `Matmul` 使用整数 ARM NEON 批量执行 MXFP8 的 RNE 量化与 P8/P32 字节转置；NaN、Infinity、FP32 subnormal 所在块保留原标量参考路径，非 AArch64 平台全部回退到参考实现。优化不改变 MXFP8 数值规则。

同一输入的 Q/K/V 与 gate/up 可复用上一次的量化布局。缓存键包含行数、列数以及完整输入字节比较，不能只依赖 ggml 的可复用内存地址。只保留一份输入快照及其打包结果，权重仍使用独立设备缓存。

设备常驻权重的所有输出列分块一次提交、一次读回；每个 tile 仍保留 fence，在重用输出 UB 和流式绑定前保证 MOVE_OUT 完成。输入每 32 行只搬入 UB 一次。未缓存权重仍按 tile 上传并同步，避免覆盖正在使用的权重暂存区；它们也复用已经载入的输入 UB。硬件 tile 数不变，减少的是主机提交和重复搬运。

设置 `GGML_GEN3_MATMUL_PROFILE=1` 可在进程退出时分别输出 prefill/decode 的量化打包、写入、命令构建、提交等待、读回、输出整理耗时。统计不含首次权重转换和 CPU 算子，诊断数据应单独运行；正式端到端基准不设置此变量。

额外回归 `gen3_sw/gen3rt/tests/quantize_fast_check.cpp` 覆盖整数 NEON 与标量参考的舍入/特殊值等价性和布局填充；`matmul_check.cpp` 覆盖相同输入复用、原地址内容变化失效，以及 N=16384 的分块输出。

## 可选的 Qwen 硬件 attention（2026-10-22）

`GGML_GEN3_ATTN=1` 接入 `FLASH_ATTN_EXT` 图节点，用已有 R51 ATTN 指令执行 QK、causal softmax、PV；无需修改比特流。Qwen2.5-0.5B 的 14 Q heads / 2 KV heads 拆成每个 KV head 的 4+3 query heads。Q 和 O 使用 P8，K 使用 P32，V 使用按 32-token 块量化的 VT；结果转换回 FP32 交给 O 投影。设备结果与位精确模拟器比较，MXFP8 attention 的结果不保证与原 CPU FP16 attention 相同。

最初版本的全硬件 decode 实测比 CPU 慢，当时采用 prefill 卸载、单 token decode 使用原普通 CPU attention。后续 F16 KV 适配优化改变了长上下文结果，见文末新节。先应用两个独立软件补丁并重建 CPU、llama、GEN3 后端：

```sh
python3 gen3_sw/patches/apply_cpu_a53.py /path/to/llama.cpp
python3 gen3_sw/patches/apply_attention_policy.py /path/to/llama.cpp
```

推荐运行参数：

```sh
sudo -n env XILINX_XRT=/usr OMP_NUM_THREADS=4 \
  GGML_GEN3_MODE=hybrid GGML_GEN3_DEVICE=board \
  GGML_GEN3_CACHE_MIB=360 G3RT_BOARD_POOL_MIB=384 \
  GGML_GEN3_CPU_A53=1 GGML_GEN3_ATTN=1 GGML_GEN3_ATTN_PREFILL_ONLY=1 \
  ./llama-server -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -ngl 99 --no-kv-offload -fa on -t 4 -np 1 -c 1152 -b 512 -ub 512
```

这里的 `--no-kv-offload` 保持 CPU 管理 KV cache；实测并不会阻止 GEN3 接管 `FLASH_ATTN_EXT`。CPU 负责 RoPE 和 KV 写入，FPGA 接管符合条件的 attention 核心。策略补丁保持转置 F16 V cache 及 32-token padding，避免 CPU decode 在每一步转置整个 V、计算多余 padding。不要在进程内切换这些环境变量，或在不同 KV 布局策略间混用保存的 KV 状态。

关闭 `GGML_GEN3_ATTN_PREFILL_ONLY` 可以诊断全硬件 attention。完全回退上一版使用 `GGML_GEN3_ATTN=0 -fa off`，A53 词表优化仍可开启。

当前硬件适配范围：单序列；F32 Q、F16 K/V；head dim 为 32/64/96/128；Q heads <=16、KV heads <=4 且整除；最多 2048 个 key；默认 1/sqrt(head_dim) 比例；无 ALiBi、softcap、attention sink。推荐策略要求 F16 KV。硬件只接受每个 query 可见连续前缀、依次增长的 causal mask；其他 mask 由 CPU 精确回退，硬件错误直接报错。转置 V 的 CPU fallback 会先物化连续副本。

KV 的设备副本按层、head 和 32-token 块缓存。每次以完整内容比较判断复用，不能只凭地址或序列长度；新请求、内容变更和尾块增长会触发相应更新。写入未完成的块不标记有效。每个 query tile 合并各 GQA 分组命令，Q MOVE_IN 后有 fence，保证 R51 能看到提交完成。只验证单模型、单请求槽，其他 Qwen 型号需要单独验证容量、数值与性能。

`GEN3_ATTN_SUMMARY` 记录 attention 调用、硬件命令、提交、KV 上传/复用和特殊 mask 回退数；`GEN3_HYBRID_SUMMARY` 继续记录线性层。详细数据和数值限制见 `docs/reports/20261022_qwen_attention`。

## FFN 批处理与词表投影（2026-10-22）

三个软件选项都使用原 R51 比特流，默认关闭：

- `GGML_GEN3_FFN=exact`：合并 gate/up 权重和主机提交，直接将 FP32 结果分别写入原来的两个张量；继续使用 ggml 原 CPU SwiGLU 和原 down 矩阵乘。复用 CPU 工作池，SwiGLU 按行分工，decode 仅启用一个工作线程。Qwen 的完整 32 行 gate/up 输出使用 1.5 MiB 设备暂存。该模式用于保留原数值路径。
- `GGML_GEN3_VOCAB=1`：将 `output.weight` / tied `token_embd.weight` 的词表投影放到 FPGA，最大 N=262144，仍受 K<=8192 等矩阵准入条件限制。按 1024 输出列分块，常驻权重的单 token 词表投影合并为一次提交。MXFP8 权重/输入不同于原 CPU Q8_0 路径，需独立准确率验收。
- `GGML_GEN3_FFN=1`：实验性的硬件 SwiGLU + down 融合，激活保留在 UB。**暂不推荐：实测“10−4”回答成“10”，而原路径回答“10 - 4 = 6”。** 原语与模拟器一致不能证明模型质量不变。速度和失败证据保留在报告中，不作为默认配置。

FFN 图匹配要求无 bias 的 parallel SwiGLU，E=64..1024、F=64..8192，均为 64 的倍数。四个节点连续、输入/形状匹配、内部结果没有 OUTPUT 标记，并使用完整图 use_counts 排除额外消费者（包括其他 backend split）。不匹配时保留物化矩阵乘和 CPU SwiGLU；空张量不提交。硬件融合要求权重常驻，容量不足和硬件失败明确报错。

实验性硬件路径的 gate/up 按 32 行交错排列，使用已有 SILU_MUL epilogue；F=4864 分成 3072+1792，down 的 FP32 部分和在 UB 累加。现有 R51 解码后的 N 存在 12 位字段，N=4096 的板端试验失败，因此规避该边界，没有修改 RTL。

容量选项：`GGML_GEN3_CACHE_KIB` 精确指定缓存 KiB，并覆盖 `GGML_GEN3_CACHE_MIB`；`GGML_GEN3_ATTN_CAPACITY` 配置设备 attention KV 容量，取 32 的倍数、范围 32..2048，默认仍为 2048。请求不得超出配置容量。普通矩阵的输出暂存为 1 MiB，exact 模式为 1.5 MiB，按输出宽度调整行 tile；权重流式暂存仅在确实需要上传时分配。

本轮 Qwen2.5-0.5B-Instruct、context 1152、单槽的复测配置（保留 CPU 词表）：

```sh
sudo -n env XILINX_XRT=/usr OMP_NUM_THREADS=4 \
  GGML_GEN3_MODE=hybrid GGML_GEN3_DEVICE=board \
  GGML_GEN3_CPU_A53=1 GGML_GEN3_ATTN=1 GGML_GEN3_ATTN_PREFILL_ONLY=1 \
  GGML_GEN3_FFN=exact GGML_GEN3_VOCAB=0 \
  G3RT_BOARD_POOL_MIB=496 GGML_GEN3_CACHE_KIB=498176 \
  GGML_GEN3_ATTN_CAPACITY=1152 \
  ./llama-server -m qwen2.5-0.5b-instruct-q8_0.gguf \
  -ngl 99 --no-kv-offload -fa on -t 4 -np 1 -c 1152 -b 512 -ub 512
```

同一配置把 `GGML_GEN3_VOCAB` 改为 `1` 可启用词表投影，层权重与词表权重均常驻 DDR。缓存预算为 486.5 MiB；496 MiB 池还要容纳输出暂存、attention KV 和命令环。不能直接增大 context 或套用于其他模型；需要重新计算设备容量并验收。未启用词表投影时，原有 384/360 MiB 池/缓存预算也能容纳层权重。两个开关关闭回退原混合路径。

回归程序：`gen3rt/tests/fusion_check.cpp`、`ggml-gen3/tests/fusion_backend_check.cpp`、`ggml-gen3/tests/exact_backend_check.cpp`。完整模型数值、生成内容、性能与适用范围见 `docs/reports/20261022_qwen_fusion/REPORT.md`。没有完成完整 AICAS 准确率验收。


## 精确数据转换与耗时归因（2026-10-22）

本轮保持 CPU 词表、CPU SwiGLU、CPU decode attention 和 FPGA prefill attention 的数值路径，优化输入转换、attention 布局及同步等待。最终同条件服务数据和可加总的耗时分解见 `docs/reports/20261022_qwen_rca/REPORT.md`。

- `GGML_GEN3_CONVERT_FAST=1`（默认）：有限数块的量化快路径、多头 NEON pack，以及合并布局解包和精确查表反量化；不改变 MXFP8 舍入。设为 `0` 可诊断原 attention 布局路径及完整的量化移位路径，有限数缩放仍使用等价内联。
- `GGML_GEN3_QUANT_THREADS=4`：对至少 1024 个 32-value 块的量化任务启用四个 OpenMP worker，范围 1..4，默认 1。小任务直接走普通循环。构建时需要 OpenMP CXX；未启用时保持串行。补丁安装脚本会在 `GGML_OPENMP` 开启且工具链支持时链接 OpenMP。
- `G3RT_SYNC_SPIN_US=1000`：同步时先主动轮询最多 1000 微秒，再回到原睡眠等待；范围 0..1000，默认 0。保留完成状态检查和超时；该配置偏向延迟，本轮未测能耗。

已验证的 Qwen2.5-0.5B-Instruct、context 1152、单槽启动方式：

```sh
sudo -n env XILINX_XRT=/usr OMP_NUM_THREADS=4 \
  GGML_GEN3_MODE=hybrid GGML_GEN3_DEVICE=board \
  GGML_GEN3_CPU_A53=1 GGML_GEN3_ATTN=1 GGML_GEN3_ATTN_PREFILL_ONLY=1 \
  GGML_GEN3_FFN=exact GGML_GEN3_VOCAB=0 \
  GGML_GEN3_CONVERT_FAST=1 GGML_GEN3_QUANT_THREADS=4 G3RT_SYNC_SPIN_US=1000 \
  G3RT_BOARD_POOL_MIB=496 GGML_GEN3_CACHE_KIB=498176 GGML_GEN3_ATTN_CAPACITY=1152 \
  /home/xilinx/gen3_300/qwen_rca_20261022/final/llama-server \
  -m /home/xilinx/gen3_300/qwen_20261021/qwen2.5-0.5b-instruct-q8_0.gguf \
  -ngl 99 --no-kv-offload -fa on -t 4 -np 1 -c 1152 -b 512 -ub 512
```

不要在进程中途切换环境变量。`OMP_WAIT_POLICY=PASSIVE` 在本轮对照中使 decode 变慢，未作为推荐设置。硬件 SwiGLU 实验和可选 FPGA 词表仍按上一节独立验收。

数值测试：`gen3rt/tests/conversion_check.cpp`，在 AArch64 上使用 1/4 量化线程对照原标量量化和布局参考；报告另含完整模型 logits 与生成 token IDs 的一致性检查。原始二进制和日志放在 `npu_pynq_zu/large_files/evidence/`，不纳入 Git。


## 既有硬件在线量化的软件接入（2026-10-22）

`GGML_GEN3_FFN=quant` 在 `exact` 的 gate/up 批处理、原 CPU SwiGLU 和完整 down 投影基础上，将符合条件的 prefill down 输入交给原 R51 在线量化后处理。不会修改 RTL、比特流、激活近似表或时钟；仍是显式可选配置，不改变默认模式。

路径：CPU 原始 SwiGLU 按 32 行切片 → FP32 上传 → 逐行 MOVE_IN 收集 C → 零 GEMM + ADD 在线量化 → 设备 DMA 拼接 P8 → 一次完整 K 的 down GEMM。CPU 不再量化或打包这份 FFN 激活，也不读回中间 P8。当前 tile 提交后，CPU 同时计算下一 tile 的原始 SwiGLU，再等待当前设备结果，形成 CPU/FPGA 流水。切片不改原张量存储和数学算子；CPU 异常会先等待已提交设备工作结束再抛出。拼接仍经设备 DDR，不能称为全程只访问 UB。FP32 上传量大于原 MXFP8 上传量，额外 GEMM/DMA 也有成本，收益以端到端实测为准。

准入条件：FFN 原有闭合子图检查不变；激活行数至少 32，K 为 128 的倍数，down 权重已常驻，N 不超过 1024。decode、小行数、非适配形状或非常驻 down 保留原 CPU 量化路径。运行期错误明确报错，不静默回退。量化器复用 68 KiB 对齐后的零权重缓存；沿用 exact 的 1.5 MiB 输出暂存，不额外申请大设备缓冲。`GEN3_ONLINE_QUANT_SUMMARY ops` 可核验路径是否执行；`GGML_GEN3_MATMUL_PROFILE=1` 包含其写入、命令、等待和读回时间；等待区间包含与设备工作重叠的 CPU SwiGLU，不能视为纯 FPGA 时间。

在上一节启动命令中使用本轮新二进制 `/home/xilinx/gen3_300/qwen_resident_20261022/pipeline/llama-server`，并将 `GGML_GEN3_FFN=exact` 改为 `GGML_GEN3_FFN=quant`。旧二进制不支持此模式。Qwen2.5-0.5B-Instruct Q8_0、context 1152、CPU 词表、FPGA prefill / CPU decode attention 的验证结果见 `docs/reports/20261022_qwen_online_quant/REPORT.md`。

`gen3rt/tests/online_quant_check.cpp` 比较板端在线量化、板端 CPU 量化和模拟器，覆盖尾行、修改输入、K=8192、N=1024 及 Qwen 4864 维输入。`gen3rt/tests/online_pipeline_check.cpp` 覆盖 65 行尾块、按需 CPU 准备、零权重复用，以及提交后 CPU 异常的排空与恢复。此前硬件 SiLU 的数值问题另行消融；不能把在线量化接通等同于完整硬件 FFN 已通过质量验收。


## F16 KV 缓存与 decode attention（2026-10-22）

当前后端默认启用 `GGML_GEN3_ATTN_F16=1`：直接验证原始 F16 KV，避免每步转换整个历史；常见 interleaved 布局批量验证完整前缀，只转换变化行，并增量量化 K。V 尺度跨 32 tokens，仍重新量化变化尾块。缓存按内容验证，覆盖回退、恢复及同地址改写；部分上传异常使前缀元数据失效。设为 `0` 诊断旧转换路径。

`GGML_GEN3_HOST_BUFFER=1` 可选启用普通 CPU buffer 共享，减少 scheduler 复制；默认关闭。新增 F16/前缀/K 编码数组在本次 Qwen 配置约占 30.49 MiB 主机内存，不增加设备 KV 容量。用 `GGML_GEN3_ATTN_PREFILL_ONLY=0` 启用 FPGA decode，`=1` 保持 CPU decode；必须启动前决定，不可进程内切换。FPGA attention 结果仍回读到 CPU 后再送 O 投影。

Qwen2.5-0.5B 的长上下文 FPGA decode 已快于原 CPU 路径，短上下文仍无优势。完整数据、原始 FPGA 对照、CPU 对照及 DDR 上限说明见 `docs/reports/20261022_qwen_decode_attention/REPORT.md`。这次软件优化保持原 FPGA logits 和生成 token IDs，不表示 MXFP8 FPGA attention 与 CPU attention 逐位一致，也未完成完整 AICAS 准确率复测。

`GEN3_ATTN_F16_SUMMARY` 记录检查/转换值数、前缀命中与 K 量化值数。回归源码为 `ggml-gen3/tests/attention_f16_check.cpp`；板端与模拟器覆盖内容变化、布局、边界、F32/F16 交替和上传异常恢复。
