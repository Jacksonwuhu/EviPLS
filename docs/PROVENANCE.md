# 来源与复现边界

整理基础：原工作区当前 AVVP 主线及 `deliverables/EviPLS_core_20261012` 中的便携教师适配器。原 Git 基线为 `5708f81e0`；工作区的后续未提交主线改动已按范围纳入新版本，原工作区保留。主模型数学实现和 legacy 损失语义保持一致；CPU 设备、路径与核验变更另有记录。

## 数据来源

- [UnAV-100 数据集](https://unav100.github.io/)，[官方代码](https://github.com/ttgeng233/UnAV)。
- [UWAV 教师实现及作者数据说明](https://github.com/Franklin905/UWAV)。
- [UWAV 作者特征仓库](https://huggingface.co/datasets/NTUBarista/uwav_features/tree/71dcb419fb8ff88ce289b396470abdf8b597f27e)，固定 revision `71dcb419fb8ff88ce289b396470abdf8b597f27e`。
- UnAV JSON 来自 UWAV README 链接的[标注目录](https://drive.google.com/drive/folders/1X4eoCPPtqi0_IKd2JGOv_q383xd27Ybb)。SHA256：`2fdfc587ad36fc405496b5520d967b5df54b2d82027d16d69e968b5cf04d3dba`。
- LLP 主模型特征、冻结标签、LLP 教师特征、冻结教师/主模型来自原项目已保留文件，按逐文件 SHA256 收录于 `assets/file_manifest.json.gz`。其中学生音频与教师音频为不同维度/提取阶段，目录名 CLAP 不能作为可互换依据。

| 官方特征包 | 字节数 | SHA256 |
|---|---:|---|
| UnAV_CLAP.zip | 875118163 | `09c162c9efb0944ef25d2c84151533dd60bff3314eb4067b90f6da6a36d06484` |
| UnAV_CLIP.zip | 1304627715 | `3987fa662687776c8cd68af78023038397b5cdfd1461972c72d800f553c1fd19` |

UnAV 为 6,489 train / 2,134 validation / 2,167 test；训练与选模只读取前两者。原始视频不随包分发。第三方数据、特征和实现保留原来源及适用许可，本整理不授予额外许可；UnAV 特征保留作者的原始压缩文件，记录固定作者源和校验值，并提供缺失时的下载入口。

## 冻结权重

- `pretrain/checkpoints/teacher_unav100_best.pt`：已有五层 TemporalTransformer，不是本次重新训练所得。
- `weights/llp/seed{6,17,42}_round2.pt`：已有三种子主模型。
- `weights/llp/seed6_round1.pt`：对应已保留第一轮权重。
- `weights/llp/historical_full_round2.pt`：历史单次实验，不能混同于最终三种子 seed 6。

新教师、重新生成伪标签与新训练使用独立运行标识。CPU 冒烟检查、全量冻结模型评估、完整重训是不同证据；本次未重新跑完 80 epoch UnAV 教师和正式 GPU 主训练。新教师训练不承诺逐位重现历史权重。

主解析器沿用 NREP（论文 *Resisting Noise in Pseudo-Labels: Audible Video Event Parsing with Evidential Learning*）；[历史代码基线](https://github.com/Jacksonwuhu/NREP/tree/5708f81e0)。教师采用 UWAV TemporalTransformer。EviPLS 的训练调度、后续适配与本次可复现性修正按来源清单单独追踪。
