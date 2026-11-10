# 数据交付清单

Git LFS 中包含 8 个大文件，总字节数 6,041,094,954（约 6.04 GB）：

- 3 份 LLP 主模型音频特征包。
- 1 份 LLP 主模型视觉/时空特征包。
- 1 份 LLP 教师特征及 25 类文本嵌入包。
- 1 份冻结伪标签、阈值和教师/主模型权重包。
- UnAV_CLAP.zip、UnAV_CLIP.zip 两份作者原始压缩包。

六份 LLP 包包含 86,705 个文件；其中包括原特征目录中不被本次 11,849 视频划分使用的额外时空特征。训练/验证/测试严格以已提供的三个 TSV 为准，不通过扫描目录决定数据划分。这些额外文件不会混入训练。

`assets/manifest.json` 与 `assets/unav_manifest.json` 给出压缩包的大小、SHA256、目标布局；`assets/file_manifest.json.gz` 给出 LLP 包逐文件 SHA256。原始 CSV 和 UnAV JSON 标注直接跟随代码版本。新运行不会覆盖这些输入。

全新克隆后运行 `git lfs pull`、`python prepare_data.py --assets`、`python prepare_data.py --unav`，随后 `python verify.py --installed`。GitHub 网页只能显示 LFS 指针/下载入口，不能把指针文本当作数据文件使用。

正式训练的数据边界：LLP 10,000 train / 649 validation / 1,200 test；UnAV 6,489 train / 2,134 validation / 2,167 test。UnAV test 不用于教师训练或选模。教师推理和主模型采用不同音频特征维度，详情见 README 的目录表。

本次重新校准的默认阈值及其来源清单位于 `pretrain/thresholds/llp/`，作为小文件直接跟随 Git 版本。压缩包内旧阈值保留，安装不会覆盖新默认目录。
