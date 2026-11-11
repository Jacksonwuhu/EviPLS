# 数据与代码来源

## 实现来源

主模型基于 [NREP](https://github.com/Jacksonwuhu/NREP/tree/5708f81e0)（*Resisting Noise in Pseudo-Labels: Audible Video Event Parsing with Evidential Learning*），教师采用 [UWAV](https://github.com/Franklin905/UWAV) 的 TemporalTransformer。EviPLS 提供 U-HNM 训练调度、IPR 标签精炼及统一训练入口。

[source_files.json](source_files.json) 记录源文件的路径、来源和 SHA256。默认主训练采用 `loss_semantics=legacy`；配置见 [README](../README.md)。

## 数据来源

- [UnAV-100 数据集](https://unav100.github.io/)，[官方代码](https://github.com/ttgeng233/UnAV)。
- [UWAV 教师实现及作者数据说明](https://github.com/Franklin905/UWAV)。
- [UWAV 作者特征仓库](https://huggingface.co/datasets/NTUBarista/uwav_features/tree/71dcb419fb8ff88ce289b396470abdf8b597f27e)，固定 revision `71dcb419fb8ff88ce289b396470abdf8b597f27e`。
- UnAV JSON 来自 UWAV README 链接的[标注目录](https://drive.google.com/drive/folders/1X4eoCPPtqi0_IKd2JGOv_q383xd27Ybb)。SHA256：`2fdfc587ad36fc405496b5520d967b5df54b2d82027d16d69e968b5cf04d3dba`。
- LLP 主模型特征、冻结标签、LLP 教师特征、冻结教师/主模型来自原项目已保留文件，按逐文件 SHA256 收录于 `assets/file_manifest.json.gz`。学生和教师分别使用 768 维、512 维音频特征，具体目录见 README。

| 官方特征包 | 字节数 | SHA256 |
|---|---:|---|
| UnAV_CLAP.zip | 875118163 | `09c162c9efb0944ef25d2c84151533dd60bff3314eb4067b90f6da6a36d06484` |
| UnAV_CLIP.zip | 1304627715 | `3987fa662687776c8cd68af78023038397b5cdfd1461972c72d800f553c1fd19` |

UnAV 为 6,489 train / 2,134 validation / 2,167 test；训练与选模只读取前两者。分发内容为提取后的特征，第三方数据、特征和实现适用各自的原始许可；UnAV 特征保留作者的原始压缩文件，记录固定作者源和校验值，并提供缺失时的下载入口。

## 冻结权重

- `pretrain/checkpoints/teacher_unav100_best.pt`：原项目保存的五层 TemporalTransformer 权重。
- `weights/llp/seed{6,17,42}_round2.pt`：已有三种子主模型。
- `weights/llp/seed6_round1.pt`：对应已保留第一轮权重。
- `weights/llp/historical_full_round2.pt`：历史单次实验的第二轮权重；三种子实验的 seed 6 权重为 `seed6_round2.pt`。

权重哈希及参考指标见 [frozen_reference.json](frozen_reference.json)，评估命令见[测试指南](TESTING.md)。新训练的产物按运行 ID 保存在 `artifacts/` 和 `models/` 下。
