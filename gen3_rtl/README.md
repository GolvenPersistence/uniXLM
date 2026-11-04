# gen3_rtl：Gen3 Node RTL

Gen3 Node 的完整 RTL：一个 32×64 阵列，向量列放在阵列中间，所有格式转换都在写回路径上完成。
架构设计见 [`docs/gen3/README.md`](../docs/gen3/README.md)。

- 目录层次仿照 `rtl/`：`node/`、`core/{array,vector,ub,vru}`、`noc/`、`memory/`、`include/`、`filelists/`。
- 运算叶单元（GEMU、dot2、FP32 运算、重量化、倒数）直接用 `rtl/core/rc`、`rtl/core/sfu` 中的原有模块，不复制。
- 文件清单：[`filelists/g3_node.f`](filelists/g3_node.f)，include 目录 `gen3_rtl/include`。

## 1. 层次

```
g3_node                              node/g3_node.v       顶层；外部接口与 Gen2 Node 相同
 +- u_ctrl   g3_ctrl                 node/g3_ctrl.v       命令 FIFO、准入检查、URF + 所有者表、事件仲裁、完成
 +- u_gemm   g3_gemm_seq             node/g3_gemm_seq.v   GEMM 定序：行 pass x 列 pass x K 块，UB 段，uop，STREAM 拉取程序，提交
 +- u_attn   g3_attn_seq             node/g3_attn_seq.v   ATTN 定序：(KV 头, 行 pass)，QK/PV/O，因果跳块，上下文槽
 +- u_ub     g3_ub                   core/ub/g3_ub.v      端口 0 A、1 B、2 C、3 VRU、4 NI
 |   +- g_bank[0..3] g3_ub_bank      memory/g3_ub_bank.v  1024 x 256，两个同步口（宏：4 x 1024x64 双口 SRAM）
 +- u_core   g3_core                 core/g3_core.v       uop 队列、类别互锁、阵列时钟使能（R = 4）
 |   +- u_feed   g3_feed             core/array/g3_feed.v       A/B/C 拍 -> 阵列 tick；权重缓冲读出
 |   +- u_array  g3_array            core/array/g3_array.v      A 斜移（行 r 延迟 r），两个半阵列
 |   |   +- u_left / u_right  g3_array_half   core/array/g3_array_half.v   32 行 x 16 GEMU
 |   +- u_vc     g3_vector_column    core/vector/g3_vector_column.v   保持 FIFO、tile 描述、C tile、P 缓冲、
 |       |                                                          softmax/O 控制、输出引擎
 |       +- g_lane[0..31] g3_vector_lane   core/vector/g3_vector_lane.v   一行 x 64 列
 +- u_wbuf   g3_weight_buffer        memory/g3_weight_buffer.v   两个 132 字的 cell 环（宏：每环 4 x 136x64 2-port RF）
 +- u_vru    g3_vru                  core/vru/g3_vru.v    Gen2 VRU（RMSNorm），段格式在顶层加宽
 +- u_ni     g3_ni                   noc/g3_ni.v          RX 推送、MOVE_OUT x2、拉取 x2（写权重缓冲）
     +- g_pull[0..1] g3_pull         noc/g3_pull.v        Gen2 拉取引擎
```

规模：19 个文件、6,976 行（Gen2 Node 用到的 Gen2 专有文件为 6,908 行）。VRU、拉取引擎、数值函数、寄存器堆是 Gen2 的副本，
只改了名字，并移植了 Gen2 物理分支（`codex/gen2-physics`）对 VRU 与控制器的 lint 修复。UB 由 488 行降到 287 + 54 行（去掉 CTU 与多播，
bank 存储独立成模块）。Gen2 的租用定序器 `g2_ls`、每核 `g2_walker`、`g2_attn` + `g2_awalk`
合并为 `g3_gemm_seq`（429 行）与 `g3_attn_seq`（374 行），因为只有一个阵列，不再需要租用与按核分配工作。

## 2. 与 Gen2 命令 ABI 的差别

命令格式、opcode、binding、布局（P8 / P32 quad 记录）、状态码、外部 AXI 接口都与 Gen2 相同，
Gen2 的 Node 测试程序可以直接驱动 Gen3。差别只有以下几处：

| 项 | Gen3 |
| --- | --- |
| BIND [127] | 转置布局（V^T 缓存）：cell (key 组 j, 维度块 c) 在 quad j·G + c，内容为 32 维 x 32 key，每维一个 scale |
| GEMM width [117:116] | 忽略（只有一个阵列）；折叠时是 eps（RMSNorm 的 eps 档位，同 VECTOR [121:120]） |
| GEMM [127] | RMSNorm 折叠（设计文档 §4.4）。epi = ADD 为产生者：D 写 x，D idx+1 写 x ⊙ g，D idx+2 写每行的 1/rms（FP32，每字 8 行，每个行遍 4 字），g 取 D idx+3（1 行 x N，P8；C 是残差，常常就是上一个产生者的 x，它的 idx+1 已是那个产生者的 x ⊙ g，所以 g 不放在 C 旁边）；其他 epi 为消费者：1/rms 取 A idx+1，FP32 累加结果先乘本行 1/rms 再做 epilogue。派生 binding 与它所跟的 binding 同 gen（同 BIND count 的约定）。产生者要求 D 为单个 P8 张量、N 是 64 的倍数 |
| GEMM 写 V^T（D 为转置 APPEND） | 要求 epi = NONE；M > 1 时追加起点必须 32 行对齐；M = 1 时任意位置（重写尾部 cell） |
| GEMM 行追加（P32 APPEND，M > 1） | 追加起点 8 行对齐（Gen2 为 16 行） |
| ATTN | 总是 64 key 的块（block-max softmax）；bc64 位忽略；V 必须是转置 binding，K 必须不是；取消了 Gen2 “奇数 panel + 因果 + GQA” 的限制；Dh ≤ 256 不再拆分 Dh 部分 |
| 同时执行 | 阵列一次只执行一个 GEMM 或 ATTN；VRU 与两个 MOVE_OUT 通道可并行 |

## 3. 计算核

### 3.1 阵列（`g3_array`, `g3_array_half`）

- 两个半阵列各 32 行 x 16 GEMU（每个 GEMU 2 个输出列），即 32 x 64 个输出；GEMU 为 Gen1/Gen2 的 `rc_mxfp8_gemu`。
- A 在中间注入：32 条斜移线（行 r 延迟 r）共享给两个半阵列，A 从中间向外走；B 从上沿进入，第 c 列延迟 c/2，向下走。
- 每行每半边每 tick 送出一个列对（113 bit，Gen2 格式），所以一条 lane 每 tick 收到 4 个值，与 Gen2 相同。
- 仍使用 Gen2 的同步推进：`array_advance = rc_clk_en && array_ready`（见 §8）。

### 3.2 operand 供给（`g3_feed`）

- A：UB 端口 A 的拍 -> 原始 FIFO（8）-> A quad（4 个 panel 的同一个字）+ A scale 组。
- B：每个半阵列一个解析器。来源有两种：UB 端口 B（上下文 0 左、1 右，各 8 个 credit），或权重缓冲（STREAM）。
  STREAM 时，每半边一个 cell 读出器按 quad 顺序读环中整个 cell，并把读完的 cell 释放给拉取引擎。
- PV 窗口的 A 来自向量列的 P 缓冲。

### 3.3 向量列（`g3_vector_column`, `g3_vector_lane`）

- **lane**：由 Gen2 block-max lane 改成一行 x 64 列。Gen2 的 “2 行 x 32 列” 改为 “左右两半 x 32 列”，存储布局不变。改动有：
  - softmax 的行统计量合并两半：块最大值、参考值、c_j、P scale、l；
  - gated SILU_MUL：左半边的 gate 写 T0，右半边的 up 在 epilogue 中读它；
  - PV 一个块含两个 K32 组：2^-D 缩放与 O 从 0 开始只作用于第一组；
  - O 的行 scale 覆盖整行；
  - GEMM 输出三种模式：行 scale、幅值（求列最大值用）、外部列 scale。
- **tile 描述**：在描述符压入时为每行算出 key 上限（因果、token 位置、有效 key 数），出 FIFO 时据此生成 keep 位。
- **C tile**：两份，每份 2 半 x 4 panel x 8 字，直接连线到各 lane。
- **P 缓冲**：2 bank x 2 个 K 组 x 16 tick x 512 bit（`rc_sram_1r1w`）。每个字携带自己的 P scale。
- **上下文槽互锁**：下一个工作项可以在同一槽的 O 还没输出时就开始。每个槽记三个 O job 计数：已发出、已读 l、已读 O。
  每个 tile 记下发出时该槽已发出的 O job 数，用它卡两处：
  - softmax 要等这些 O job 都读完 l；
  - PV 累加要等这些 O job 都读完 O。
  如果不加这两处互锁，后一个工作项会覆盖前一个工作项还要读的 l 或 O。
- **RMSNorm 折叠**（设计文档 §4.4，命令位 [127]）：
  - 消费者：GEMM 最后一个 K 组的累加和先经 M0（GEMM 中空闲）乘本行 1/rms，再进入 epilogue 和 gate 写入。1/rms 由端口 C 的上下文 1 按行遍读入，列里按行遍奇偶双缓冲，行遍最后一遍输出后释放。
  - 产生者（epi ADD）：M2 乘 g（D idx+3 的一行，与 C tile 一起按遍读入，存放在对应 C 条目的槽里）而不是 1。e12 同时得到 x（A1）和 x ⊙ g（M2）：x 进 T1，x ⊙ g 进 T0（非门控 GEMM 中 T0 空闲），各有自己的行最大值和 scale。
  - 输出引擎对产生者的每个 tile 走两遍编码：先 x，再 x ⊙ g（地址加 x ⊙ g 与 x 的基址差）；行遍的最后一遍之后再写 4 个字的 1/rms。整遍只有最后一次写带 mark。
  - Σx²：x 编码时，lane 用码字和行 scale 解码出 x，在 M0 上平方，在 A2（GEMM 中空闲）上累加进 16 个部分和。顺序与 g3_vru 相同：第 k 列进第 k mod 16 个，k 递增。左半 32 列随编码累加，右半 32 列的码字先存下，等这一遍编码完再补上。行遍的前 16 列从 +0 开始。
  - 行遍最后一遍补完后，lane 在 M0、A2 上跑 g3_vru 的 FINAL 程序（两两求和、×1/H、+eps、5f375a86 − (v >> 1)、两次牛顿），得到 1/rms。1/H 由定序器按 g3_vru 的 inv_count 算出，随 uop 下发。因此 1/rms 与 g3_vru 逐位相同；与 VRU 路径的差别只在“x ⊙ g 先量化、1/rms 后乘”。
- **输出引擎**：GEMM tile 与 O pass 共用。
  - 行布局：每 2 个 go 组装 8 个字（4 panel x 2 半）。
  - V^T：先做一遍幅值 pass，用 32 路最大值得到列 scale；再做编码 pass，每 4 个 go 组装 16 个 V^T 字。
  - V^T 单 token 追加到部分填充的 cell：从端口 C 读旧 cell 两次。第一次读出旧 scale 和“全零维”；第二次把旧码重新舍入到新 scale、插入新 token、写回。
  - 时序：
    - 32 路列最大值是两级寄存的树（32 → 8 → 2 → 1），比原来多 2 周期；
    - 尾部重写分两步：先寄存旧码与 scale 增量，再重新舍入。尾部规则保证新 scale 不小于旧 scale，所以重新舍入只是指数下移，加 4 bit 有效数的 RNE，结果与 `quant_e4m3(decode_element(...))` 逐位相同；
    - 引擎写口带一级寄存器。
    这三处来自第一次 DC（§9.2）。Node 用例多 1–37 周期。

## 4. UB（`g3_ub`）

Gen2 UB 去掉 CTU 与多播，段格式加宽到 128 bit：每个内层索引有自己的偏移（o1..o3），
因为 P8 张量中 4 个 panel 的间距不是常数。

- 存储：4 个 bank（`g3_ub_bank`），每个 1024 x 256 bit、两个同步口，每口每周期一次读或一次带字节写使能的写。
  在 `G3_USE_TSMC28_MC2` 下映射到 4 片 TSDN28 1024x64 双口 SRAM，与 Gen2 物理分支的 UB 相同。
- 仲裁（单周期）：先写（按端口序），再按上下文轮询读。每个 bank 最多 2 次访问，各占一个口。
  读在授权的同一周期访问 bank，出口延迟仍是 2 周期。同一周期内，同一个字若已有写访问，后来的读或写都推迟一周期。
- 端口与上下文：

| 端口 | 读上下文 | 写 |
| --- | --- | --- |
| 0 阵列 A | 0：A | D（向量列） |
| 1 阵列 B | 0：左 B，1：右 B | — |
| 2 阵列 C | 2：C tile，0：旧 V^T cell | — |
| 3 VRU | 0 MAIN，2 AUX | RESULT |
| 4 NI | 0 TX0，1 TX1 | RX 推送 |

## 5. NI 与权重缓冲

STREAM 权重用 Gen2 的拉取程序格式：两个引擎分别拉偶数和奇数 32 行 band（步长 2 x 33G 字），
每轮行 pass 重拉一遍。数据写入半阵列旁边的 132 字 cell 环，不经过 UB。RX 推送与 MOVE_OUT 不变。
每个环是一写一读（同步读）。在 `G3_USE_TSMC28_MC2` 下映射到 4 片 136x64 2-port RF：MC2 要求深度是 8 的倍数，所以用 136 字。

## 6. 控制

- **`g3_gemm_seq`**：
  - 遍历顺序：行 pass（32 行）x 列 pass（64 列，即 B band 2c 与 2c+1；gated 时每个 pass 输出一个 tile：左 gate、右 up）x K 块（≤ 32 组）。
  - 每个 K 组发出：A scale（每 panel 一拍）、A payload（一个段，4 个 panel 偏移）、UB 中的 B cell。
  - 最后一个 K 块的 uop 携带输出描述。C tile 在 K 循环之后发出；V^T 尾部追加时，额外发出两遍旧 cell 读取。
  - 提交：P8 单张量按行 pass 提交 quad 前缀；其余情况在结束时按元素提交。
- **`g3_attn_seq`**：
  - 工作项为 (KV 头, 行 pass)，一个 KV 头的 q 头按 8 行 panel 堆叠，每个 pass 32 行。
  - 顺序：QK(0) QK(1) PV(0) QK(2) PV(1) …；每个 PV 覆盖所有 64 维的 Dh pass，之后是每个 Dh pass 的 O 描述。
  - 每行的 token 由 4 个 panel token 基址给出（因果掩码不再受 panel 奇偶限制）。
  - Dh ≤ 128 时两个上下文槽交替，下一个工作项的 QK(0) 在当前工作项最后一个 PV 之前发出，与该块的 softmax 重叠；
    更大时一个槽、两个 O bank，下一个工作项要等上一个写完 O 才开始。
- **`g3_ctrl`**：Gen2 控制器去掉 core 租用。父槽为：0 阵列（GEMM 或 ATTN）、1–2 TX、3 VRU。另外增加转置 binding 和 §2 中的准入检查。

## 7. 验证

```
python -B tb/gen3/run_node_tb.py            # 29 个 Node 用例
python -B tb/gen3/run_layer_smolvlm.py      # SmolVLM-256M 一层：32 token prefill + 1 步 decode（两个 norm 都折叠；--norm0 / --norm1 vru 走 VRU）
```

- bench：[`tb/gen3/node/g3_node_tb.sv`](../tb/gen3/node/g3_node_tb.sv)，与 Gen2 的程序格式相同；`+TRACE=1` 记录 UB D 写。
- 用例：
  - Gen2 的 GEMM / RMSNorm / binding 用例原样复用（n1–n7、n9–n15）；
  - ATTN 用例改为 V^T 缓存与 Gen3 oracle（`tb/gen3/g3lib.py`，block-max Bc64）；
  - KV 追加用例校验 K 行与 V^T cell，包括多步 decode 的尾部重写；
  - RMSNorm 折叠（n26–n29，2026-10-19）：一条产生者 GEMM（ADD）后接消费者 GEMM（NONE、ROPE、SILU_MUL），x、x ⊙ g、每行 1/rms 与每个消费者的输出都逐位比较；oracle 是 `tb/gen3/g3lib.py` 的 `fold_producer` / `fold_consumer`（1/rms 按 g3_vru 的 `rmsnorm_rows` 求，x ⊙ g 从 FP32 的 x 乘 g 后量化，消费者先乘 1/rms 再做 epilogue）。n27 有两个行遍（1/rms 双缓冲的两个奇偶），n28 为 1 行、N = 576（9 遍，同层的 decode），n29 是准入错误（x ⊙ g 的 gen 不同、N 不是 64 的倍数、消费者 1/rms 的 gen 不同）。g 从 C idx+1 移到 D idx+3 后（§2）重跑，29/29 通过，只有 n27 少 4 周期（binding 顺序变了）。
- 判定：host 内存逐字比较，加上完成状态检查。

结果（2026-10-17，Questa 2021.1）：25/25 通过，均逐位一致。2026-10-19 加入折叠后 29/29 通过，前 25 个用例的周期与此前逐个相同；折叠用例另用种子 12–16 各跑一遍，均通过。各用例的 Node 周期如下：

| 用例 | 周期 | 用例 | 周期 |
| --- | ---: | --- | ---: |
| n1 single | 1,759 | n14 RMSNorm | 1,175 |
| n2 tail | 1,303 | n15 RMSNorm 576 + GEMM | 4,735 |
| n3 ADD epilogue | 1,629 | n16 ATTN prefill | 1,619 |
| n4 STREAM B | 2,381 | n17 ATTN decode GQA | 1,738 |
| n5 decode STREAM | 1,461 | n18 ATTN 多块 Dh128 | 2,559 |
| n6 chain | 1,378 | n19 ATTN 错误 | 1,059 |
| n7 heads | 1,208 | n20 ATTN Dh256 | 8,053 |
| n8 KV 追加 | 10,567 | n21 ATTN Dh160 | 3,573 |
| n9–n11 binding / 同步 | — | n22 ATTN Dh96 | 1,618 |
| n12 gated STREAM | 1,045 | n23 ATTN Dh224 | 2,218 |
| n13 ROPE heads | 962 | n24 ATTN 因果奇数 panel | 4,215 |
| | | n25 KV 追加尾部（3 头 x 5 步） | 20,643 |
| n26 折叠：产生者 + 3 个消费者 | 5,305 | n28 折叠：1 行 x 576 列 | 7,400 |
| n27 折叠：40 行（两个行遍） | 5,026 | n29 折叠准入错误 | 1,558 |

SmolVLM 层的结果见 §7.1。

### 7.1 SmolVLM-256M 层

测试流程：32 token 的 prefill 层，然后在它建立的 KV 缓存上做 1 步 decode（KV 33）。权重 STREAM，V 缓存为 V^T，oracle 用 Gen3 数值。
每种配置的所有输出都逐位一致：prefill/decode 的 x3、K 缓存、V^T 缓存（下游折叠时另有 x3 ⊙ g0 和 1/rms）。

三种配置（2026-10-19，同一个测试脚本）：
- **VRU**（`--norm0 vru --norm1 vru`）：两个 norm 都走 VRU；
- **norm1 折叠**（`--norm0 vru`）：o GEMM 是产生者（x2、x2 ⊙ g1、1/rms），gate/up 是消费者；
- **两个都折叠**（默认）：层的输入是 x、x ⊙ g0 和 1/rms（真实运行中来自上一层的 down GEMM，这里由主机推入），q/k/v 是消费者；down GEMM 是下一层 norm0 的产生者（x3、x3 ⊙ g0、1/rms），但只在 decode 中：prefill 里 down 运行时 h、x2、x3 和 KV 缓存都活着，再加 x3 ⊙ g0 和 g0（1 行的 P8 也占满一个 8 行 panel，168 字）约 4,630 字，超过 UB 的 4,096 字，所以 prefill 的 down 只写 x3。这一层没有 VRU 命令。

| | Gen2 RTL | Gen3 模型 | VRU | norm1 折叠 | 两个都折叠 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 132,639 | 131,991 | 132,553（−0.06%） | 128,989（−2.75%） | **125,945（−5.05%）** |
| decode | 122,489 | 121,697 | 122,077（−0.34%） | 121,529（−0.78%） | **121,379（−0.91%）** |

括号内相对 Gen2 RTL。此前（g 在 C idx+1、binding 顺序不同）VRU 与 norm1 折叠两种配置是 132,465 / 121,989 和 128,901 / 121,441，现在各多 88 周期，来自 binding 的先后。

每条命令的周期（prefill / decode；q 从输入阶段结束算起，VRU 配置下 q 的启动与 norm0 重叠，所以 q 的差别里含这部分）：

| 命令 | Gen2 RTL | VRU | norm1 折叠 | 两个都折叠 |
| --- | ---: | ---: | ---: | ---: |
| 输入（推入；VRU 配置含 norm0） | 7,597 / 2,613 | 7,685 / 2,702 | 7,685 / 2,701 | 4,194 / 2,000 |
| q | 10,874 / 10,900 | 10,938 / 10,906 | 10,938 / 10,907 | 11,378 / 11,249 |
| k | 4,064 / 4,028 | 4,088 / 4,056 | 4,088 / 4,056 | 4,092 / 4,060 |
| v | 4,035 / 4,027 | 4,154 / 4,152 | 4,154 / 4,152 | 4,158 / 4,156 |
| attn | 3,452 / 2,220 | **2,967 / 1,567** | 2,967 / 1,567 | 2,966 / 1,566 |
| o | 11,096 / 11,083 | 11,146 / 11,116 | 11,441 / 11,315 | 11,441 / 11,315 |
| norm1 | 4,534 / 1,134 | 4,535 / 1,134 | — | — |
| gate/up | 57,233 / 57,237 | 57,248 / 57,227 | 57,925 / 57,615 | 57,925 / 57,615 |
| down | 29,157 / 29,077 | 29,185 / 29,047 | 29,184 / 29,046 | 29,184 / 29,253 |

- **attn**：比 Gen2 快 14%（prefill）和 30%（decode），与模型相差 6% 和 10%。
  - 第一版 RTL 每个工作项都把 QK→softmax→PV 的延迟露在外面：prefill attn 5,029、decode 2,145，整层 prefill 134,516，比 Gen2 慢 1.4%。
  - 加了跨工作项的 QK 早发（§6）和上下文槽互锁（§3.3）之后，得到上表的结果。
- **v**：每次多 110 多个周期，因为 V^T 要先做一遍幅值 pass 求列 scale，再做编码 pass。
- **其余 GEMM**：都在 Gen2 的 ±0.7% 以内。它们受 NI 权重带宽限制，Gen3 的结构对它们没有影响。
- **VRU 配置**：与 Gen2 基本持平。这与设计文档的结论一致：没有 RMSNorm 折叠时，Gen3 的收益主要是 UB 访问和控制的简化，而不是周期。
- **norm1 折叠**：prefill −3,564（−2.7%），decode −548（−0.45%）。
  - norm1 这条命令没有了（4,535 / 1,134）；
  - o 作为产生者多 295 / 199：每遍多编码一遍 x ⊙ g，行遍最后要等 FINAL 并写 1/rms；
  - gate/up 作为消费者多 677 / 388：先按行遍取 1/rms，累加结果再乘 1/rms（多 4 周期）后进 epilogue。
- **两个都折叠**：再 −3,044（prefill）、−150（decode），相对 VRU 配置共 −5.0% / −0.57%。
  - 输入阶段 −3,491 / −701：VRU 的 norm0 没有了，推入的是 x ⊙ g0 和 1/rms 而不是 g0；
  - q 多 440 / 342（消费者，加上原来与 norm0 重叠的启动），k、v 各多 4；
  - decode 的 down 作为产生者多 207；prefill 的 down 若也是产生者，按 o 的 +295 估计，prefill 约 126,240（−4.8%）。
  - 真实的多层运行中层的输入已在 UB 里，不用推入；本测试的输入阶段（4,194 / 2,000，几乎全是推入）在那时也会大部分消失。
- 模型给的“折叠 + 重启重叠”是 −10.3% / −4.7%（设计文档表 1），其中重启重叠没有实现。
- **UB 容量**：设计文档 §4.4 说折叠不增加 UB 容量，这不对。产生者在运行时就要写 x ⊙ g，而 VRU 的 xn 是在产生者结束、它的输入释放之后才写的，所以峰值多出一个 x ⊙ g 和一个 g。o 放得下（O、x、x2、x2 ⊙ g1、g1、KV 和 STREAM 环共约 3,670 字）；down 放不下（h 有 1,584 字，共约 4,630 字），32 token prefill 时差约 540 字。要在 Gen3 的 128 KB UB 中折叠 norm0，需要把 prefill 分成更小的块、减小 KV 容量或加大 UB；FPGA 上 UB 用 URAM，没有这个限制。

## 8. 与设计文档的偏差、未实现项

| 项 | 状态 |
| --- | --- |
| 无全局停顿（准入 + 波前） | 未实现：保留 Gen2 的同步 advance（扇出 1,024 个 GEMU，R = 4 多周期） |
| RMSNorm 折叠进向量列 | 已实现（§3.3，命令位 [127]）。SmolVLM 层两个 norm 都折叠；32 token prefill 时 down 作为 norm0 产生者放不进 UB（§7.1），只在 decode 中折叠。第一层（embedding 之后）由主机算 x ⊙ g 和 1/rms |
| 单 K 组的 GEMM | K = 32 且列遍多时会死锁（行之间的导出偏斜超过 T1 的两个 tile；与折叠无关，K ≥ 64 正常）：待修 |
| 重启重叠 | 未实现：阵列一次一个父命令，上一个完成后才准入下一个 |
| 长上下文 K/V 走 NI | 未实现：K/V 必须在 UB 中（与 Gen2 相同，ctx ≠ ONESHOT 返回 UNSUPPORTED） |
| 存储宏 | UB、权重缓冲、lane 寄存器堆、P 缓冲在 `G3_USE_TSMC28_MC2` / `CORE_USE_TSMC28_MC2` 下映射到宏（§9）；C tile、URF、FIFO、VRU 暂存仍是触发器 |
| 时序 / 面积 | 向量列与半阵列已 DC（§9.2）；Node 级 DC 内存不足未完成，物理实现已暂停，风险与估计见 §9.3；没有做 PT、ICC2 |

## 9. Lint 与 DC

流程与重现命令见 [`scripts/eda/gen3/README.md`](../scripts/eda/gen3/README.md)。运行在 ic-canopsys 上：
目录 `~/Desktop/runs/npu/20261017/gen3_rtl`。源码快照有三份：
- `src_v4`：lint 用；
- `src_v5`：RTL 与 v4 相同，只更新了脚本；
- `src_v6`：最终 RTL，含 §9.2 的时序修改。

报告摘要在 `docs/gen3/evidence/eda/`。
lint 与 DC 依次运行，不并行。

### 9.1 Lint（SpyGlass T-2022.06-1，`lint/lint_rtl`，top `g3_node`）

| 运行 | fatal | error | warning |
| --- | ---: | ---: | ---: |
| v1 | 2 | — | — |
| v2 | 0 | 170 | 2,218 |
| v4 行为级存储（最终） | 0 | **0** | 2,202 |
| v4 宏接口（MC2 Liberty） | 0 | **0** | 2,207 |
| v6 行为级存储（§9.2 的时序修改后，最终 RTL） | 0 | **0** | 2,205 |

v1 是解析错误：generate 块里 `wire [63:0] a = …, b = …;` 的第二个名字被当成标量，已拆成两行。

v2 的 170 个 error 及修法：

| 规则 | 数量 | 位置 | 修法 |
| --- | ---: | --- | --- |
| W415 多驱动 | 128 | VRU 累加表 | 每个累加字一个写进程（移植 Gen2 物理分支的修复） |
| UndrivenInTerm-ML、W123 | 25 | 控制器未用的父槽来源 | 未用槽固定为常数 |
| W505 阻塞/非阻塞混用 | 9 | 控制器读者计数 | 每个计数器一个驱动（移植 Gen2 物理分支的修复，写成单个连续赋值） |
| W416 函数返回位宽 | 6 | `blk`、`a_blk`、`p_code`、`rne` | 显式位宽 |
| W336 时序块中的阻塞赋值 | 2 | feed 解析器、输出引擎装载 | 改为 wire |

同一轮还修了：多位条件（`H ? …` 改为 `H != 0 ? …`）、显式截断（`8'(...)`）、未用变量；UB 也重写成独立的 bank 模块，并去掉跨 always 块共用的循环变量。
每轮修改后 Node 25 个用例都逐位通过、周期不变；SmolVLM 层在 lint 修改后仍为 132,452 / 121,981（§9.2 的时序修改后为 132,465 / 121,989）。

v4 剩下的 2,202 个 warning 都已归类，按规则：

| 规则 | 数量 | 说明 |
| --- | ---: | --- |
| FlopEConst | 1,732 | Gen1 叶子 `rc_fp32_stage`：advance 恒为 1 的流水寄存器 |
| W415a / STARC05-2.2.3.3 | 333 / 55 | 同一 always 块中先给默认值再覆盖、或在循环里赋值；Gen2 也未改 |
| SYNTH_5166 | 18 | 仿真用的 `$fatal` 检查 |
| W362、STARC05-2.1.3.1、W116 | 29 | 比较或函数实参位宽不同，值域受 ABI 限制 |
| W216 / W215 | 10 | 对 integer 循环变量取位 |
| W287b | 8 | 未接的统计输出 |
| STARC05-2.11.3.1 | 7 | 状态机的次态与寄存器写在一个块里 |
| W240、W528、STARC05-1.3.1.3 | 9 | 零深度延迟线的 clk/rstn 未用；GEMU 与 dot2（Gen1 叶子）的写法 |
| CMD_define02 | 1 | 行为级运行中 `SYNTHESIS` 没被用到 |

宏接口那次运行多出 5 个 WarnAnalyzeBBox，是 5 种宏按 Liberty 当黑盒读入的提示。

### 9.2 DC（T-2022.03-SP2，TSMC28 HPC+ 7T SS 0.81 V / 125 C，1.25 ns）

约束与 Gen2 向量单元的 DC（`scripts/eda/vector_unit`）相同，所以向量列可以直接和它的每核 374,654 µm² 比较。
阵列寄存器加 R = 4 多周期约束（setup 4 / hold 3）。综合自下而上：先综合向量列和一个半阵列，再综合 Node，其中这两个块已映射、固定不动。

| 块 | 面积 µm² | 其中宏 | WNS ns | TNS ns | 违例路径 | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 向量列，第一次（src_v5） | 1,456,758 | 574,902 | −3.94 | −28,008 | 45,753 | 1.8 h |
| 向量列，时序修改后（src_v6） | **1,489,207** | 574,902 | **−0.20** | −2,283 | 22,121 | 1.7 h |
| 半阵列（32 x 16 GEMU，R = 4 多周期） | **947,394** | 0 | −0.15 | −290 | 2,301 | 2.5 h |
| 参考：Gen2 向量单元 x 4 核 | 1,498,616 | 609,114 | −0.13 | −392 x 4 | 5,689 x 4 | — |

- **第一次向量列 DC**：最差路径（−3.94 ns，235 级逻辑）是 V^T 的 32 路列最大值，原来写成 32 级串行比较。
  第二差的是尾部重写到 UB 写口的组合路径：选旧码 → decode → quant → 写数据，约 4.2 ns。
  其余数万条违例里，DC 把加法器留成了 ripple 结构：DC 默认 critical range 为 0，只对每组最差路径做延时优化。
  这三处的修法见 §3.3。
- **修改后的向量列**：剩余违例均匀地落在 −0.17 到 −0.20 ns，集中在 lane 的 FP32 流水级（Gen1 叶子）和 lane 的统计量寄存器，
  形态与 Gen2 向量单元（−0.13 ns）相同，属于同一流程下 critical range 为 0 的结果。没有做时序收敛（例如 `set_critical_range`、retiming）。
- **面积**：
  - 向量列比 4 个 Gen2 向量单元少 0.6%，基本持平。设计文档 §7 的估计是持平到 −11%。
  - 32 条 lane 共 1,239,683 µm²，每条 38,740，与 Gen2 lane（39,660）相当。
  - 共享部分 249,524 µm²：P 缓冲 49,045、倒数 2,123，其余是顶层逻辑（C tile 32 Kbit 触发器、保持 FIFO、输出引擎、tile 描述）。
    Gen2 每核共享部分约 57,400 µm²，4 核约 229,500 µm²，而且 Gen2 的 C tile 在 feed 里，不在向量单元中。
- **半阵列**：512 个 GEMU 连同网格寄存器，每个约 1,850 µm²。109,664 个寄存器加了 R = 4 多周期约束。
  剩下的违例都在 `row_data` 输出口：这是 16 个 PE 的 OR 汇出，块级约束按单周期算；在 Node 里它进保持寄存器，属于多周期路径。
- **Node 级**：把两个已映射的块固定后再综合 Node，在映射开始时报内存不足（OPT-1603，31 GiB VM），与 Gen2 物理分支的整 Node DC 相同。
  改为把两个块当黑盒、只综合其余逻辑的那一次运行，在映射阶段按决定停止（物理实现暂停）。所以没有 Node 面积和跨块时序的实测。

### 9.3 风险与估计（物理实现暂停时的记录）

面积（单元面积，SS）：

| 项 | mm² | 依据 |
| --- | ---: | --- |
| 向量列 | 1.489 | DC |
| 阵列（2 x 半阵列） | 1.895 | DC |
| UB 宏（16 x TSDN28 1024x64，每个 46,310 µm²） | 0.741 | Liberty |
| 权重缓冲宏（8 x 136x64，每个 8,769 µm²） | 0.070 | Liberty |
| 其余逻辑（控制器与 URF、两个定序器、UB 控制、NI、VRU、feed、阵列 A 斜移） | 约 0.5（0.4–0.7） | 估计：约 11 万个触发器（A 斜移 1.5 万、VRU 2.9 万、feed 1.2 万、URF 与所有者表约 2.4 万、UB FIFO 约 1.5 万等）x 1.7 µm²，组合逻辑按 1–2 倍，加 VRU 的 FP 单元约 0.1 |
| **Node** | **约 4.7（4.6–4.9）** | 按 70% 布局利用率约 6.7 mm² |

频率：两个块在 SS 下 WNS 为 −0.15 和 −0.20 ns，按现状约 1.40–1.45 ns（690–710 MHz）。
到 800 MHz 需要做收敛：剩余违例浅而均匀，主要是 ripple 加法器，预计可以收敛，但没有验证。

风险（按影响排序）：

1. **全局 advance**：`array_advance` 单周期驱动约 25 万个寄存器的使能，其中 `array_ready` 是向量列 64 个 FIFO 余量的与。
   块级 DC 覆盖不到，Node 级没有计时。这是设计文档“无全局停顿”要解决的问题。
   处理方向：阵列的起停放在入口，即斜移之前的 feed FIFO 与缓冲，阵列内部不停。
   - 斜移线与网格只随 `rc_clk_en` 前进，数据带 valid 位，入口不能前进时注入空拍。
     这样 `rc_clk_en` 是固定的四分之一节拍，可以在本地复制，或直接用分频时钟。
   - 入口在注入一个 tile 之前，先确认它在向量列侧需要的资源都已就绪：输出引擎与 D 写回空间、C tile、T 缓冲、O bank。
     这样 tile 进入后，向量列能按导出速率（每半行每 tick 一项）接收。
   - 保持 FIFO 只吸收剩下的短停顿：O 读出占用存储读口、ROPE 间隔、gated 左右半的先后。
     深度要用仿真里的最大占用确定。
2. **UB 读地址**：上下文仲裁组合地驱动 16 个 TSDN 宏的地址口，没有计时。若不满足，地址加一级寄存器，读延迟 +1 周期。
3. **floorplan**：
   - 阵列 1.9 mm²，宽度为 Gen2 单核的 4 倍；
   - 中间的向量列 1.5 mm²、104 个宏，每半行 113 bit 的导出线进入列中；
   - B 总线横跨全宽；
   - UB 的 16 个大宏（每个 109 x 425 µm）要靠近 A/B/C 端口。
   都没有评估。
4. **块内时序与保持时间**：保持时间违例 −0.20 ns 没有修，要靠 PT/ICC2。
5. **工具容量**：Node 级综合需要块抽象（ILM）或更大内存的机器。

