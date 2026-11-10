# 核验报告

本报告记录代码整理后的检查范围、结果和待完成项，由本次整理执行者完成核查。

## 验证环境

本地检查使用现有 `nrep` 环境：Python 3.10、PyTorch 1.12.1，CPU 配额 0.5 核、内存上限 2 GiB，无可用 CUDA。GitHub CI 在全新 Python 3.10 环境中安装 CPU 依赖并通过 91 项测试及 shell 语法检查，见[首次发布 CI](https://github.com/Jacksonwuhu/EviPLS/actions/runs/42823406379)。Conda 环境重建和 GPU 全量训练列于待完成项。

## 修正内容

1. 分别配置教师 512 维音频与主模型 768 维音频及配套视觉、时空特征路径；推荐入口按仓库根目录解析路径。
2. 获取固定版本的完整 UnAV CLAP/CLIP 特征及原始标注，教师 loader 读取模型实际使用的特征。
3. 通过 `reproduce.py` 统一教师训练、校准、标签生成、主训练和测试，按运行 ID 保存产物与日志。
4. 逐文件检查 LLP 特征和伪标签，以及 UnAV 训练、验证特征的维度、有限值和类别名称/ID。
5. 补齐调度器与风险函数依赖，添加设备和 worker 参数，沿用模型及默认 legacy 损失的数学实现。
6. 运行超时后清理子进程组并保存失败记录；检查输出目录是否已存在。
7. 提供压缩包、逐文件哈希清单、来源记录、安装检查和 CI。
8. 记录 LLP 标注中的一条反向时间区间，按原解析器的空切片语义处理；其他非法区间报告错误。
9. 冻结教师使用 meta 初始化后绑定 checkpoint 张量，减少加载时的内存占用；张量及预测一致性检查通过。
10. 默认教师阈值改为 `pretrain/thresholds/llp` 中的完整验证集校准产物，并绑定教师权重、文本嵌入及数据哈希；旧阈值保留在原目录。

## 已完成检查

| 检查 | 范围与结果 | 记录 |
|---|---|---|
| 单元、数值与接口测试 | 91 项通过 | `python -m unittest discover -s tests -v` |
| 全量输入 | LLP 11,849 视频：主模型 59,245 份数组、教师 23,698 份片段特征；UnAV 6,489 个训练视频和 2,134 个验证视频的所需特征检查通过 | [input_audit.json](input_audit.json) |
| 数据包安装 | 六份 LLP 包在空目录解压，86,705 个文件逐一匹配 SHA256；两份 UnAV ZIP 的 SHA256 与固定作者版本一致，解压 CRC 通过 | [installation_check.json](installation_check.json) |
| 教师训练接口 | 2 个 UnAV 训练视频、1 个验证视频，缩小骨干完成 AdamW 更新和验证 | [teacher_step_smoke.json](teacher_step_smoke.json) |
| 历史阈值推理 | 五层冻结教师处理 train/val/test 各 2 个视频；3,000 个标签位置中有 19 个与旧缓存不同 | [legacy_threshold_label_check.json](legacy_threshold_label_check.json) |
| 两轮主训练流程 | 使用上述六视频标签，每划分 2 视频、每阶段 1 epoch，完成 U-HNM、IPR、验证选模和两轮权重保存 | [two_round_smoke.json](two_round_smoke.json) |
| 冻结主模型评估 | seed 6 完整 1,200 个测试视频的十项指标与历史记录精确一致；三种子权重哈希与原记录匹配 | [frozen_evaluation_check.json](frozen_evaluation_check.json) |
| 阈值校准 | 相同冻结教师在完整 649 视频验证集完成校准，记录权重、文本嵌入、划分和阈值哈希 | [calibration_check.json](calibration_check.json) |
| 新阈值标签对比 | 2 个训练视频、1,000 个标签位置与冻结缓存逐位一致 | [label_regeneration_check.json](label_regeneration_check.json) |
| 启动与语法 | 不同工作目录下的启动计划、Python 入口语法和 shell 语法检查通过 | `reproduce.py --help`、`bash -n scripts/run_full_pipeline.sh` |

## 待完成项

- 新建完整 Conda 环境的安装验证。
- 从 UnAV 开始的 80 epoch 教师训练，以及使用新教师伪标签的完整 GPU 主训练和最终评估。
- 新阈值下全部 11,849 视频的标签生成与缓存对比。当前已完成的对比范围为两个训练视频。
- 使用新阈值生成标签后的两轮主训练流程复核。已通过的两轮流程测试使用历史阈值生成的六视频标签。
- seed 17、42 冻结主模型的全量测试集推理。

## 运行问题记录

- 原始 LLP 音频标注有一条 `onset=9, offset=0` 的反向区间，按原解析器作为空切片处理，详情见 [annotation_anomalies.json](annotation_anomalies.json)。
- 教师加载及部分推理运行收到 SIGKILL。低内存加载实现完成后，教师推理和完整验证集校准均有成功记录；后续追加的验证、测试小样本推理仍被终止，具体终止来源尚未定位。
- 全量校准曾在与其他模型检查并发时中断，之后串行运行完成，结果记录在 `calibration_check.json`。
- 首次全量测试评估在 1,158/1,200 视频时达到 15 分钟时限；改用新运行 ID 和 1 小时预算后完成全部评估。

失败运行的日志保留在本地 `artifacts/` 对应目录。复现流程以已提取特征为起点，数据目录与各阶段命令见 [README](../README.md)。
