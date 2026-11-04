# 版本关系

## Gen3

`gen3_rtl/` 和 `gen3_sw/` 分别来自同名源码 ZIP，保留原相对目录、模块划分和调用入口。RTL 包包含计算与控制模块，软件包包含命令仿真、运行时、模型程序和 ggml 接入代码。

源码只规范化文本换行；未对计算、调度、量化或设备协议做功能改写。已有构建文件、模块 README 和测试源文件均保留。

## KV260 历史软件

`base/` 对应历史软件归档中的 `software/fusion/`。其余三个目录用公共基线与差异文件表示，`snapshot.json` 保存各自的实际文件集合和 SHA-256。恢复时仅输出该集合中的文件。

| 变体 | 原目录 | 状态 |
|---|---|---|
| production | software/production | 对应子树内可读取文件已恢复 |
| server_p280_rms_norm_mul_fuse | software/components/server_p280_rms_norm_mul_fuse | 对应子树内可读取文件已恢复 |
| mtmd_p225_image_encode_cache | software/components/mtmd_p225_image_encode_cache | 输入归档在该子树内截断，仅恢复可读取部分 |

恢复示例：

```sh
python3 tools/materialize_variant.py production --output restored/production
python3 tools/materialize_variant.py server_p280_rms_norm_mul_fuse --output restored/server_p280
python3 tools/materialize_variant.py mtmd_p225_image_encode_cache \
  --allow-partial --output restored/mtmd_p225
```

目录去重是逐文件、逐字节比较后的存储整理。差异文件包含对应版本的实际源码，不以推测生成缺失实现。恢复目标必须是新目录。

## 分发内容

不包含已有可执行程序、目标文件、静态库、构建缓存、仿真输出、模型权重、开发会话配置和实验随笔。模型词表测试夹具中的 GGUF 也未打包；需要这些夹具的上游测试应另行准备数据。

归档输入和各分发文件的对应关系保存在 `docs/SOURCE_INDEX.csv`；`SHA256SUMS` 校验分发文件，源码原始压缩包校验值见 `docs/ARCHIVES.json`。
