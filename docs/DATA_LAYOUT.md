# 数据交付清单

Git LFS 中包含 8 个大文件，总字节数 6,041,094,954（约 6.04 GB）：

- 3 份 LLP 主模型音频特征包。
- 1 份 LLP 主模型视觉/时空特征包。
- 1 份 LLP 教师特征及 25 类文本嵌入包。
- 1 份冻结伪标签、阈值和教师/主模型权重包。
- UnAV_CLAP.zip、UnAV_CLIP.zip 两份作者原始压缩包。

六份 LLP 包包含 86,705 个文件；其中包括原特征目录中不被本次 11,849 视频划分使用的额外时空特征。训练、验证、测试分别按三个 TSV 加载对应视频。

`assets/manifest.json` 与 `assets/unav_manifest.json` 给出压缩包的大小、SHA256、目标布局；`assets/file_manifest.json.gz` 给出 LLP 包逐文件 SHA256。原始 CSV 和 UnAV JSON 标注直接跟随代码版本。训练产物按运行 ID 另存。

全新克隆后运行 `git lfs pull`、`python prepare_data.py --assets`、`python prepare_data.py --unav`，随后 `python verify.py --installed`。LFS 指针记录大文件的地址和哈希，`git lfs pull` 会将其替换为实际压缩包。

数据划分：LLP 10,000 train / 649 validation / 1,200 test；UnAV 6,489 train / 2,134 validation / 2,167 test。教师使用 UnAV train 训练、validation 选模。教师推理和主模型采用不同音频特征维度，详情见 README 的目录表。

本次重新校准的默认阈值及其来源清单位于 `pretrain/thresholds/llp/`，作为小文件直接跟随 Git 版本。压缩包内旧阈值保留，安装不会覆盖新默认目录。
