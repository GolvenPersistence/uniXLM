# gen3_sw：Gen3 Node v3 的软件栈

整模型 SmolVLM2（视觉编码器 + 投影 + 30 层语言模型 + lm_head）经 llama.cpp 跑在 Gen3 Node v3 上。设计见
`docs/gen3/node_v3.md`（§1 数值、§3 ABI、§4–5 主机接口与存储、§6 程序结构）。

| 目录 | 内容 |
|---|---|
| `emu/` | C++ 命令级逐位模拟器：在命令层执行 Node 程序，写出与 RTL 相同的外部存储和完成状态（40/40 个 RTL 用例逐位一致，见 `emu/README.md`）。数值内核可单独复用 |
| `gen3rt/` | 运行时：命令编码（`cmd.h`）、字布局（`layout.h`）、设备（`device.h`：`EmuDevice`；`board.h`：`BoardDevice`，PYNQ-ZU）、程序构建（`program.h`）、语言模型（`llm.h`）、视觉编码器（`vit.h`）、验证工具 `tools/llm_run.cpp` |
| `ggml-gen3/` | ggml 后端：设备 “GEN3”（GPU 类型，llama.cpp 把层卸到它上面）；权重在 set_tensor 时转成 Node 镜像；graph_compute 识别语言模型图和视觉图，交给 gen3rt |
| `patches/apply_ggml_gen3.py` | 把后端接进 llama.cpp 树（AICAS 基线，ggml 0.9.4）：头文件、`ggml/src/ggml-gen3/CMakeLists.txt`、CMake 选项 `GGML_GEN3` / `GGML_GEN3_BOARD`、后端注册 |
| `board/load_overlay.py` | 板上加载 v3 overlay（`gen3_fpga/node_board_v3`），把 pl_clk0 设为 100 MHz |
| `llama.cpp/`、`llama.cpp-build*/` | llama.cpp 树（从 AICAS 复制，不入版本库）与构建目录 |

## 语言模型程序（gen3rt::Llm）

- 权重：MXFP8 P32 镜像在设备内存；gate/up 按 32 行 band 交错（SILU_MUL）；归一化的 g 为 FP32 行。
- 每层（tile 最多 32 行，不跨 32 token 的 band；UB 16K 字）：
  - q（RoPE，C = 一张 m × 64 的三角表，各头共用）、k、v：fold 消费者（读 x·g 与 1/rms）；
  - KV：m = 1（decode），或 tile 从 band 起点开始（pos % 32 = 0）时在 Node 上追加：MOVE_IN K band 与 V^T cell（band 起点时取一块全零，不留旧上下文的行）→ 重新绑定为 APPEND → k/v 追加 m 行 → MOVE_OUT → FENCE，整个 tile 一次提交；只有不从 band 起点开始的 tile（prefill 的第一段）把 k（P8 行）、v（FP32 行）写到 DDR，由 CPU 按 decode 尾部重写的逐 token 规则并入 cache；
  - ATTN：一条命令覆盖 9 个 Q 头 / 3 个 KV 头，K/V^T 从 DDR 流式读入；
  - o（FP32 残差 C）→ x2、x2·g1、1/rms（fold 产生者）；gate/up（SILU_MUL）；down → x3、x3·g0'、1/rms（下一层的产生者）。
- lm_head：fold 消费者、FP32 D，每条 GEMM 1,024 列（logits 缓冲复用 lm_head 时已经不用的 x … h 区域）。

### 超出单条命令的模型（2026-10-22 软件拆分）

命令格式的限制是：一条 ATTN 最多 4 个 KV 头、group 最多 4；一条 GEMM 最多 4,064 列。超出时，程序结构不变，只在软件里按这些限制拆分。
不需要拆分的模型（SmolVLM2、SmolLM-135M 等）权重镜像和命令流与拆分前完全相同。

- **注意力分部**：
  - KV 头按每段不超过 4 个分段，每个 KV 头的 q 头按每轮不超过 4 个分轮；
  - 每个（段，轮）各用一条 q GEMM 和一条 ATTN；
  - W_q 的行按这个顺序存放（部内以 KV 头为主序，与 ATTN 一致）；
  - o 投影是一串 K 分段的 GEMM，每个注意力分部一段，W_o 的列按段依次存放；各段之间用 FP32 C 连接，部分和在 XA / XB 两个 F32 区之间交替，最后一段是 fold 产生者；
  - 取代了原来 >4 KV 头时经 DDR 搬运 Q / O 和每层一次主机同步的做法。
- **FFN 分部**：
  - gate/up 按整 32 行 band 分成不超过 4,064 列的几部分；交错镜像中每部分是连续的一段，不需要拷贝；
  - down 是对应的 K 分段链，W_d 的列按部分依次存放。
- **NEOX RoPE**（Qwen2）：在上传时把 W_q / W_k 每个头内的行重排（新行 2i ← 原行 i，新行 2i+1 ← 原行 i + dh/2），Node 的相邻对旋转就等于 NEOX 的 (i, i + dh/2) 对。q·k 与 KV cache 对公共置换不变，V 不重排。
- **UB**：32 行 tile 放不下时，x2·g1 / x3·g0' 与两份 1/rms 共用一块区域。产生者的 BIND 按 R5 等待另一份的读者（本层 q/k/v 或 gate/up），这些读者此时已经结束或即将结束。Qwen2.5-0.5B 因此从 16,936 字降到 16,000 字。
- **权重缓存**：拆分或重排过的镜像用版本 2（`G3WGT002`），头部带拆分描述，不会与旧缓存混用。
- **限制参数**：`LlmConfig::max_group / max_kv_heads / max_ffn_columns` 默认取硬件限制；调小只用于在小模型上触发拆分做等价检查。对应的 `gen3rt_llm` 选项是 `--max-group`、`--max-kv-heads`、`--max-ffn-columns`；`--test-neox` 先对 W_q/W_k 做逆置换，再走 NEOX 程序，结果应与原模型逐位相同。
- **qwen2 架构**：`gen3rt_llm` 和 ggml-gen3 的融合路径都接受 qwen2。q/k/v 偏置必须为零，否则拒绝，因为 Node 程序还没有偏置；偏置留给下一步的 RTL 改动。

结果与 Qwen 的数值分析见 [软件拆分报告](../docs/reports/20261022_software_decoupling/REPORT.md)。

## 视觉编码器（gen3rt::Vit，混合方式）

- CPU：LayerNorm、残差、注意力输出与 fc2 的偏置、im2col、pixel shuffle（FP32）。
- Node：patch embedding（像素拆成 hi/lo 两个 K 段，偏置用额外 K 组）、q/k/v 与 fc1（偏置用额外 K 组）、fc1 的 GELU、注意力（12 头分 3 条命令，每条 4 个 KV 头，K/V^T 镜像在 DDR）、o（按头组拆成 3 个 K 段，FP32 C 链接）、fc2、投影（K = 12,288 拆成两段，FP32 C 链接）。
- tile 为 32 个 patch（1,024 个 patch 共 32 个 tile）。
- 与 ggml CPU 结果相比，图像特征相对误差 0.119、余弦 0.993（`GGML_GEN3_VIT_CHECK=1`），与数值研究的预期一致。

## 构建与运行（x86，模拟器）

```bash
export PATH=/d/llvm-mingw-20260224-ucrt-x86_64/bin:$PATH
python gen3_sw/patches/apply_ggml_gen3.py
cmake -S gen3_sw/llama.cpp -B gen3_sw/llama.cpp-build-gen3 -G "MinGW Makefiles" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DGGML_FPGA=OFF -DGGML_GEN3=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_STANDARD_LIBRARIES="-lws2_32 -lmswsock"
cmake --build gen3_sw/llama.cpp-build-gen3 -j 6 --target llama-cli llama-mtmd-cli
gen3_sw/llama.cpp-build-gen3/bin/llama-cli -m SmolVLM2-256M-Video-Instruct-f16.gguf -p "The capital of France is" -n 16 -ngl 99 -fa off --temp 0 -no-cnv
gen3_sw/llama.cpp-build-gen3/bin/llama-mtmd-cli -m SmolVLM2-256M-Video-Instruct-f16.gguf --mmproj mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf --image img.jpg -p "what is written in the image?" -ngl 99 -fa off --temp 0
```

环境变量：`GGML_GEN3_DEVICE=emu|board`、`GGML_GEN3_THREADS`（模拟器线程数）、`GGML_GEN3_VERBOSE`、`GGML_GEN3_DUMP`（打印收到的图）、`GGML_GEN3_VIT_CHECK`（视觉特征与 CPU 对比）、`GGML_GEN3_PROFILE`（板上：每条命令都报告完成，按命令角色统计 Node 周期，进程结束时打印；`gen3rt_llm --profile` 相同）。

`gen3rt/tools/llm_run.cpp`（`make -C gen3_sw/gen3rt`）：每步把 Node 的 logits 与 llama.cpp CPU 比较；`--prefill` 用 prefill 处理提示词；`--board --check-emu` 在板上运行并逐位对比模拟器。

## 当前结果（模拟器，逐位等同 RTL）

- llama-cli，`-ngl 99`：图分成 CPU（embedding）与一个 GEN3 split；“The capital of France is” → “ Paris.”；与 CPU 的 top-1 一致率约 80–100%（不一致处 CPU 的第一名在 Node 排第 2–3，MXFP8 数值造成）。
- llama-mtmd-cli：视觉编码 + 投影 + 语言模型整段在 Node 程序上，回答与 CPU 相同。

## 2026-10-21 软件优化与复现

`gen3rt_llm --benchmark` 为权重驻留的 TTFT/TPOT 路径；加 `--host-timing` 可记录 input/build/submit/sync/read 阶段时间。阶段计时用于定位，正式延迟用未开启阶段计时的结果。`submit` 和 `sync` 包含设备执行重叠/等待，不等同于 CPU 运算时间。

`--weights-cache model.g3w` 首次创建预打包权重文件，后续启动完整核对原 GGUF 的 SHA256、形状、布局版本和缓存内容校验值后加载。首次生成缓存与缓存命中必须分别计时。板端构建需要现有 OpenSSL 开发库（`-lcrypto`）；非板端有可移植 SHA256 实现。缓存不包含 KV。权重格式或量化语义变更时必须升级文件版本。

`Llm::upload_layer` 在主机直接交错 gate/up，只上传最终 Wgu；传统逐 tensor 的 upload/attach 接口仍保持原行为，不能将 54.7 MB 的节省宣称为所有 ggml 加载路径都已获得。首层 norm 缓存在 upload/attach 更新时失效；RoPE 按固定配置、位置和 tile 长度缓存。

`Vit` 默认缓存上一份归一化 patch tensor 的完整编码结果，逐字节比较输入，无哈希碰撞；不同图片/预处理结果不会误命中，`upload()` 使缓存失效，单实例存储约 3.3 MB。可用 `set_encode_cache(false)` 测未命中成本。缓存命中只复用视觉特征，不复用文本回答或 KV；命中耗时不是首次图像请求的端到端 TTFT。

单元验收：`make -C gen3_sw/gen3rt check`（板端加 `BOARD=1`，仅使用内存设备，不操作 FPGA）。测试覆盖 80 万条命令与旧逐 bit 编码器的差分、直接交错与旧上传路径的字节一致性、权重缓存往返、SHA256 标准向量和损坏/错误模型拒绝。视觉板测工具为 `gen3rt/tests/vision_cache_test.cpp`，输入为真实 projector GGUF 和固定的 FP32 patches。

本轮软件改动的板端常规数值与性能记录见 [软件优化实测](../docs/reports/20261021_software_optimization/REPORT.md)。CPU/内存压力下原始程序和优化候选都出现长时间阻塞，候选尚未通过稳定性验收；板端保留候选用于复现，没有替换稳定部署入口。不要将常规延迟测量当作压力验收通过。
