# 测试与参考结果

## 单元测试

在项目环境中运行：

```bash
python -m unittest discover -s tests -v
bash -n scripts/run_full_pipeline.sh
DRY_RUN=1 RUN_ID=preview PYTHON_BIN=python bash scripts/run_full_pipeline.sh
```

测试覆盖安装中断重试、流水线阶段衔接、数据完整性、阈值与权重绑定、模型加载、风险计算、损失函数、训练调度和运行超时处理。单元测试使用小型输入，可在 CPU 上运行。GitHub Actions 在 Python 3.10 / PyTorch 1.12.1 CPU 环境执行相同检查，状态见 [Actions](https://github.com/Jacksonwuhu/EviPLS/actions)。

## 数据检查

安装数据后运行：

```bash
python verify.py --installed
python -m nrep.validation --json
python reproduce.py train-teacher --run-id input_check --check-only
python scripts/audit_inputs.py --output artifacts/input_audit.json
```

`verify.py` 检查压缩包和安装文件的 SHA256。`nrep.validation` 检查 LLP 划分、视频键、数组维度和标签范围。`audit_inputs.py` 进一步检查 LLP 教师输入、UnAV train/validation 特征及类别映射。

## 小样本训练检查

以下命令需要已安装的特征数据。每次运行使用新的输出路径或运行 ID。

```bash
python scripts/smoke_teacher.py --output artifacts/smoke_teacher
python scripts/smoke_main.py --run-id smoke_main --timeout-seconds 1200
python reproduce.py generate-labels --run-id smoke_labels --device cpu --limit 3
```

教师检查使用 2 个训练视频和 1 个验证视频、单层小型骨干，执行 AdamW 更新和验证。主模型检查使用每划分 2 个视频，每阶段 1 epoch，覆盖两轮 U-HNM、IPR、验证选模和权重保存。标签检查使用五层教师为前三个训练视频生成标签。运行日志和结果位于相应 `artifacts/` 目录。

## 预训练权重评估

```bash
python reproduce.py evaluate-llp --run-id test_seed6 \
  --checkpoint weights/llp/seed6_round2.pt
```

默认读取冻结伪标签和完整 1,200 视频测试集。seed 6 的 CPU 评估与发布记录的十项指标逐项一致，详见 [frozen_evaluation_check.json](frozen_evaluation_check.json)。seed 17、42 的权重哈希及参考指标见 [frozen_reference.json](frozen_reference.json)。

## 参考数据

| 内容 | 配置与结果 | 文件 |
|---|---|---|
| 输入检查 | LLP 59,245 份主模型数组、23,698 份教师片段特征；UnAV train/validation 所需特征通过 | [input_audit.json](input_audit.json) |
| 安装检查 | 86,705 个 LLP 文件哈希匹配；UnAV ZIP 的 SHA256 和解压 CRC 通过 | [installation_check.json](installation_check.json) |
| 教师训练接口 | 小型骨干的优化器更新和验证通过 | [teacher_step_smoke.json](teacher_step_smoke.json) |
| 两轮主训练接口 | 使用历史阈值生成的六视频标签，完成两轮训练 | [two_round_smoke.json](two_round_smoke.json) |
| 阈值校准 | 五层冻结教师，完整 649 视频验证集 | [calibration_check.json](calibration_check.json) |
| 配套阈值标签对比 | 2 个训练视频的 1,000 个标签位置与缓存一致 | [label_regeneration_check.json](label_regeneration_check.json) |
| 历史阈值标签对比 | 6 个视频的 3,000 个标签位置中有 19 个差异 | [legacy_threshold_label_check.json](legacy_threshold_label_check.json) |

参考记录来自 Python 3.10 / PyTorch 1.12.1 CPU 环境。端到端 GPU 重训、完整 Conda 环境重建、配套阈值的全量标签对比及后续两轮训练、seed 17/42 的再次全量评估尚无对应验证记录。训练配置和命令见 [README](../README.md)。
