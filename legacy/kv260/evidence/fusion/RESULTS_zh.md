# SmolVLM2 + P302d 最小融合版：重新定位与验收

本轮交付保留 prefill attention 卸载和 P302d decode 路径，并新增一个仅作用于 decode 小型 CPU 子图的线程策略。
本次新增的CPU线程调整不改变模型、数学公式或量化方式；硬件沿用已验证P302d。未覆盖原机器人服务。

## 1. 最终干净对照

官方相同图像/提示：501 输入 token、4096 输出 token；prefill 包含视觉编码。每次启动新服务，关闭 prompt 复用。
同一新编译 CPU 库、同一 libllama、同一 P302d，只有 `GGML_CPU_DECODE_SMALL_GRAPH_SERIAL` 为0/1。
顺序 off/on/on/off/off/on；无并行归档、编译或外部采样。下表为全部三次均值±样本标准差，不删除慢轮。

| 配置 | Prefill tok/s | Decode tok/s | Prefill+decode / s |
|---|---:|---:|---:|
| Prefill attention + P302d，CPU小子图优化关 | 59.811 ± 2.224 | 42.464 ± 1.008 | 104.879 |
| 交付融合版：上述配置 + CPU小子图优化 | 59.907 ± 0.073 | 43.995 ± 0.265 | 101.468 |

Decode均值提升 3.60%，完整请求阶段总时间降低 3.25%。Prefill约60 tok/s，不写成每轮均超过60。
这里只证明该固定工作负载上的小样本收益，不宣称消除全部长尾或全局最优。

| 顺序 | 开关 | Prefill tok/s | Decode tok/s | 阶段合计 / s |
|---:|---|---:|---:|---:|
| 1 | small_off | 57.245 | 43.401 | 103.129 |
| 2 | small_on | 59.837 | 43.700 | 102.102 |
| 3 | small_on | 59.983 | 44.216 | 100.989 |
| 4 | small_off | 61.026 | 42.594 | 104.374 |
| 5 | small_off | 61.163 | 41.397 | 107.135 |
| 6 | small_on | 59.901 | 44.068 | 101.312 |

## 2. Decode时间闭合（同轮三次均值）

| level | 项目 / s | 优化关 | 交付版 |
|---:|---|---:|---:|
| 0 | decode eval wall | 96.494807 | 93.104757 |
| 1 | 　graph_compute | 91.604590 | 88.044964 |
| 2 | 　　FPGA backend wall（包含后端软件，不等于纯PL时间） | 56.272796 | 56.261191 |
| 2 | 　　其余CPU子图及图调度 | 35.331794 | 31.783773 |
| 1 | 　graph之外的decode工作 | 4.890217 | 5.059793 |

每层按减法余量闭合，不能把嵌套项目再次相加。最后两类仍不是算子级独立测量；要进一步细分需另行做CPU算子/调度采样，不把诊断成本加进clean时间。

## 3. 改了什么

- `libllama.so`：采用已在板上编译验证的P288同源prefill-only逻辑batch门控（fc6cc779…），仅prefill启用额外QK/PV卸载。
- P302d bit：SHA256 4bcbec9b30717fcc5acf3f38bf400e5a62d65cf40928d51de67b22f5057fbcd6，保留B预取改进；未启用W4。
- 本次实际代码修改仅在`ggml-cpu.c`：只对decode、<=16节点、白名单、单节点<=4096个F32元素、合计<=16384元素的小子图使用单线程。矩阵乘法、softmax、大张量、encoder/prefill原策略不变。
- 新CPU库板上编译SHA256 bdad4cb9a7e9cd68f17310d4d542b891bdaf14b83d8698c5eee9888b31685d18。server、ggml-base、FPGA backend、mtmd保留生产文件。
- 初次配置的CURL依赖及源文件排除规则错误已修正；首次单元测试把单行softmax原策略误写为4线程，核对源码后改为1线程，同时增加4行softmax仍为4线程的检查。失败日志保留，未把失败当通过。

## 4. 正确性与诊断证据

- 板上单元测试：策略开关、4096/4097边界、encoder/prefill不变、矩阵/softmax不变、ADD/MUL逐位一致全部通过。
- 固定子集按原评测器：1/1、4/5、24/30。30例预测逐项与未修改线程策略的版本一致；六轮4096-token生成文本完全一致。
- 每轮decode FPGA MUL_MAT计数618345不变。不是减少输出token、关闭计算、改KV裁剪策略获得速度。
- 独立诊断记录495495次CPU子图调用，其中372645次非矩阵子图；含ADD/归一化子图7.656秒、GLU子图6.707秒。这些是含调度的子图时长，不是裸算子，也不是可全部省掉的时间。
- 独立45秒49Hz CPU-clock perf共8026样本，丢失0；热点为libgomp的比较/回跳忙等指令。跨CPU累计采样占比不能当作端到端墙钟占比。
- 官方说明：[GOMP_SPINCOUNT](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html)、[OMP_WAIT_POLICY](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html)。本交付未修改其等待环境变量。

## 5. 对先前慢轮判断的更正

首轮原库/新库关闭新增卸载/新库开启卸载的decode分别41.8592/41.9161/41.9159 tok/s。不能拿此前独立批次43.02均值作为固定常数来断言融合退化。
本轮41.11/36.05慢轮与异步SCP下载完整运行库存在重叠；另三轮被1Hz线程观察部分覆盖。它们全部保存在all_runs.csv并明确标注，不能混入最终无观测干扰对照，也不能直接用来证明某个代码路径有根本问题。
未完成开/关SCP干扰的因果量化，因此不宣称全部慢轮差异都由SSH导致。针对性能测试的修复是将大文件归档与測试串行化，诊断和验收分开。

## 6. 交付与使用边界

板上包：`/home/xilinx/workspace_gyq/smol_fusion_p302d_20261008`。先运行`python3 fusionctl.py verify`。
该包面向当前官方负载和合并比赛模型，启动模式包含ignore-eos/4096；不是机器人fixed64-LoRA的自动替代品。
模型、系统XRT、loader和R5依赖沿用板上现有安装并锁定哈希；它不是空白SD卡完整镜像。硬件bit/HWH、运行库、源码改动、验证日志和CSV随包提供。
沿用当前对照时钟：PS FCLK读数333333330 Hz、CPU1333333 kHz；P302d构建约束为300 MHz，不把PS读数称作实测PL频率。
未重测完整官方score、能效、TTFT或全数据集准确率。没有宣称刷新比赛提交分数。
测试后恢复原机器人服务，交付版未设置为开机默认服务。
