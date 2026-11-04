# 依赖与完整性

## 1. Gen3 RTL 外部运算模块

`gen3_rtl/filelists/g3_node.f` 引用以下十个文件，它们未包含在三个输入归档的可读取内容中：

```text
rtl/core/rc/rc_fp32_stage.v
rtl/core/rc/rc_mxfp8_fp32_math.v
rtl/core/rc/rc_mxfp8_fp32_mul.v
rtl/core/rc/rc_mxfp8_dot2.v
rtl/core/rc/rc_mxfp8_gemu.v
rtl/core/rc/rc_mxfp8_delay_line.v
rtl/core/rc/rc_mxfp8_requant2.v
rtl/core/rc/rc_sram_1r1w.v
rtl/core/sfu/sfu_fp32_exp.v
rtl/core/sfu/sfu_fp32_recip.v
```

上述文件决定运算叶单元实现，需使用与此历史版本匹配的源码。完整硬件构建还需要 Node 外围、互连、约束和 FPGA 工程。归档内的 Gen3 Node 文件无法单独重建完整板卡系统。

## 2. 软件依赖

- llama.cpp 与 ggml 基线：随附于 `legacy/kv260/base/`，可由 `tools/prepare_llama.py` 复制并接入 Gen3。
- 板端环境：PYNQ、XRT、OpenSSL，以及匹配的 overlay。
- 模型与数据：GGUF、视觉投影模型、OCRBench 数据及原板端脚本调用的外部评测客户端另行准备。
- 原验证依赖：RTL testbench、Python oracle、工艺宏库与部分设计说明未包含在这些归档中。

## 3. 历史归档截断

`kv260_smol_fusion_source_20261026.tar.gz` 在读取 `software/components/mtmd_p225_image_encode_cache/models/ggml-vocab-falcon.gguf` 附近发生 `unexpected end of data`。文件尾之后的内容无法确认。

分发只使用完整读出的文件，未写入部分文件、空文件占位或补造实现。最后一个历史变体带有 `partial_archive_subtree` 标记，默认恢复会拒绝执行；传入 `--allow-partial` 后可恢复已读出的子集。其他已读子树也不代表完整板卡工程。

## 4. 自动检查

```sh
python3 tools/check_source.py
python3 tools/check_source.py --strict
```

普通检查确认分发文件哈希和 Python 语法，并列出外部 RTL 缺项；严格模式在缺少外部 RTL 时返回状态码 2。文件损坏或 Python 语法错误返回状态码 1。
