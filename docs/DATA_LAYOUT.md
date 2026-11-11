# 数据目录与格式

## 分发文件

Git LFS 中包含 8 个大文件，总字节数 6,041,094,954（约 6.04 GB）：

- 3 份 LLP 主模型音频特征包。
- 1 份 LLP 主模型视觉/时空特征包。
- 1 份 LLP 教师特征及 25 类文本嵌入包。
- 1 份冻结伪标签、阈值和教师/主模型权重包。
- UnAV_CLAP.zip、UnAV_CLIP.zip 两份作者原始压缩包。

六份 LLP 包包含 86,705 个文件；其中包括原特征目录中不在 LLP 11,849 视频划分中的额外时空特征。训练、验证、测试分别按三个 TSV 加载对应视频。

`assets/manifest.json` 与 `assets/unav_manifest.json` 给出压缩包的大小、SHA256、目标布局；`assets/file_manifest.json.gz` 给出 LLP 包逐文件 SHA256。原始 CSV 和 UnAV JSON 标注直接跟随代码版本。训练产物按运行 ID 另存。

全新克隆后运行 `git lfs pull`、`python prepare_data.py --assets`、`python prepare_data.py --unav`，随后 `python verify.py --installed`。LFS 指针记录大文件的地址和哈希，`git lfs pull` 会将其替换为实际压缩包。

数据划分：LLP 10,000 train / 649 validation / 1,200 test；UnAV 6,489 train / 2,134 validation / 2,167 test。教师使用 UnAV train 训练、validation 选模。教师推理和主模型采用不同音频特征维度，详情见 README 的目录表。

默认教师的配套阈值及其来源清单位于 `pretrain/thresholds/llp/`，作为小文件直接跟随 Git 版本。压缩包内旧阈值保留，安装不会覆盖新默认目录。

## 特征与标签

| 用途 | 目录 | 单视频形状 |
|---|---|---|
| UnAV 标注 | `data/UnAV/unav100_annotations.json` | 100 类，train/validation/test |
| UnAV 教师音频/视觉 | `data/UnAV/feats_CLAP/segment_feats`、`feats_CLIP/segment_feats` | `(L,512)` / `(L,768)` |
| UnAV 文本类别嵌入 | 各模态 `event_feats/all_event_feats.npy` | `(100,512)` / `(100,768)` |
| LLP 教师音频/视觉 | `pretrain/data/LLP/feats_CLAP`、`feats_CLIP` | 片段 `(10,512)` / `(10,768)`，文本为 25 类 |
| LLP 主模型音频 | `data/CLAP/features` | `(64,768)` |
| LLP 主模型视觉 | `data/CLIP/features` | `(10,768)` |
| LLP 主模型时空 | `pretrain/data/LLP/feats/r2plus1d_18` | `(10,512)` |
| 冻结伪标签 | `data/CLAP/a_segmentwise_pseudo_labels`、`data/CLIP/v_segmentwise_pseudo_labels` | `(10,25)` |
| 默认教师阈值 | `pretrain/thresholds/llp/{a,v}_classwise_threhsolds.npy` | `(25,)`，验证集校准；文件名沿用 `threhsolds` 拼写 |
| 历史阈值存档 | `data/CLAP/{a,v}_classwise_threhsolds.npy` | 旧版阈值，见[常见问题](TROUBLESHOOTING.md) |
| 新伪标签 | `artifacts/<run-id>/labels/{audio,visual}` | `(10,25)` |
| IPR 精炼标签 | `artifacts/<run-id>/pseudo_labels/round_2/{audio,visual}` | `(10,25)` 软标签 |


教师和主模型分别读取各自的音频特征目录。`L` 为 UnAV 视频片段数；LLP 每个视频包含 10 个一秒片段。
