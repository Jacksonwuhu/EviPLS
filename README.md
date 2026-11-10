# EviPLS：UnAV 教师 → LLP 伪标签 → U-HNM + IPR

这是从原 NREP 工作区整理出的 **AVVP/LLP 完整特征级复现版本**。统一入口为 `reproduce.py`；训练数据、文本嵌入、冻结伪标签、教师和主模型权重都有明确目录及校验记录。原始研究工作区和旧实验没有被覆盖。

本版本从作者发布/已有的 CLAP、CLIP、R2Plus1D 特征开始，不包含从原始视频提取特征的流水线。UnAV 使用时序强标注训练教师；LLP 验证集的模态时序标注用于阈值校准。不能把这一信息使用条件描述成“全过程只使用视频级弱标签”。

## 1. 获取代码、数据和环境

```bash
# 先跳过自动下载，避免 Git 在检出时启动大文件传输。
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Jacksonwuhu/EviPLS.git
cd EviPLS
git lfs install --local
git lfs pull

conda env create -f environment.yml
conda activate evipls-main-v3
python prepare_data.py --assets
python prepare_data.py --unav
```

`assets/` 是 Git LFS 管理的分块压缩包，包含 LLP 全链路输入及已有权重；`prepare_data.py` 校验 SHA256 后安装到项目目录。UnAV 的两份原始特征压缩包也随 Git LFS 提供；`--unav` 优先读取本地包，缺失时从固定 UWAV 作者源下载，并按固定 SHA256 校验后分别解压到两个模态目录。仓库包含原始 UnAV JSON 标注。

网络无法连接 Hugging Face 时，可显式指定兼容镜像，校验标准不变：

```bash
python prepare_data.py --unav --base-url https://hf-mirror.com
```

下载预算默认为 2 小时，最多下载约 2.18 GB UnAV 压缩数据。可用 `--cache /path/to/cache` 分离缓存位置；错误代理环境可加 `--ignore-proxy`。缺失、损坏、未下载的 LFS 指针和已有不同内容会报错；不会用伪造特征代替。磁盘建议为代码、LFS 缓存、压缩包、解压数据共预留 25 GB，正式训练另留权重空间。

Linux / Python 3.10 / PyTorch 1.12.1 / torchvision 0.13.1 / CUDA 11.3。直接依赖按已验证运行环境固定，见 `requirements-runtime.txt`；这不是完整的传递依赖锁。本次未重新创建完整 Conda 环境。当前执行机器没有可用 GPU，正式训练建议在 GPU 主机执行；CPU 适用于输入检查、推理及小批量诊断。

## 2. 先核验全部输入

```bash
python verify.py --installed
python -m unittest discover -s tests -v
python -m nrep.validation --json
python reproduce.py train-teacher --run-id input_check --check-only
python scripts/audit_inputs.py --output artifacts/input_audit.json
```

检查包括：LLP 10,000/649/1,200 三个划分、11 位视频键不冲突、全部输入形状及有限值、全部伪标签范围、UnAV train/validation 的标注与类别顺序、CLAP/CLIP 特征。`audit_inputs.py` 还检查 LLP 教师输入；不会训练或使用测试集选模。

## 3. 从 UnAV 开始运行全部阶段

```bash
RUN_ID=unav_to_llp_s6 PYTHON_BIN=python bash scripts/run_full_pipeline.sh
```

默认流程：UnAV 教师 80 epoch → LLP 验证集逐类阈值校准 → 全部 LLP 视频伪标签 → 两轮 U-HNM + IPR 主训练 → 最终测试。教师种子 1000，主训练种子 6；可通过 `SEED=17` 等调整主训练种子。

教师阶段默认最多 24 小时，后续每阶段最多 4 小时；可以显式设置 `TEACHER_HOURS`、`STAGE_HOURS`。超时会终止子进程组，失败记录保留。每阶段有独立 `artifacts/<run-id>/run.json` 和 `launcher.log`；同名运行拒绝覆盖。完整默认训练不等同于一轮冒烟检查。

也可以逐步执行，便于检查和从已完成阶段继续：

```bash
python reproduce.py train-teacher --run-id teacher_s1000 --data-root data/UnAV

python reproduce.py calibrate-teacher --run-id calibration_s1000 \
  --checkpoint artifacts/teacher_s1000/teacher/checkpoints/checkpoint_best.pt

python reproduce.py generate-labels --run-id labels_s1000 \
  --checkpoint artifacts/teacher_s1000/teacher/checkpoints/checkpoint_best.pt \
  --threshold-dir artifacts/calibration_s1000/thresholds

python reproduce.py train-llp --run-id full_s6 --seed 6 \
  --label-root artifacts/labels_s1000/labels

python reproduce.py evaluate-llp --run-id test_s6 \
  --checkpoint models/full_s6/full_s6_r2.best \
  --label-root artifacts/labels_s1000/labels
```

原始 LLP 音频标注有一条 `onset=9, offset=0` 的反向区间。本版本记录该异常并保留原教师/解析器的空切片语义，不猜测修补为 9–10 秒；详见 `docs/annotation_anomalies.json`。

教师只按 UnAV validation Seg-AV 选模；新教师必须重新校准阈值，不能沿用旧教师阈值。校准仅使用 649 个 LLP 验证视频。权重和阈值文件哈希绑定，伪标签生成不读取模态时序真值。生成的标签与冻结标签分开保存。

主训练保持历史 `loss_semantics=legacy`、每轮音频/视觉各 10 epoch、两轮、batch 128、lr 5e-4、IPR η=0.3 的设置；其他实验选项不代表这个默认配置。历史训练器在每轮末计算测试指标，选权重仍依据验证集；禁止用这些中途测试值调整超参数。

## 4. 使用已有教师或伪标签

```bash
# 固定教师、固定阈值重新生成标签；默认覆盖所有 LLP 划分。
python reproduce.py generate-labels --run-id frozen_teacher_labels
python reproduce.py train-llp --run-id regenerated_s6 \
  --label-root artifacts/frozen_teacher_labels/labels

# 直接使用冻结伪标签训练。
python reproduce.py train-llp --run-id frozen_labels_s6 --seed 6

# 复核已保存的主模型；CPU 主机自动使用 CPU。
python reproduce.py evaluate-llp --run-id frozen_test_s6 \
  --checkpoint weights/llp/seed6_round2.pt
```

**阈值路径修正：** 旧 `data/CLAP` 阈值在六视频检查中产生 19 个缓存差异，原文件保留供追溯。本版本默认改用 `pretrain/thresholds/llp`：由相同冻结教师在完整 649 视频验证集重新校准，并绑定权重、类别嵌入与数据哈希。新阈值在已完成的两个训练视频检查中与缓存逐位一致；这不证明全量历史标签等价。见 `docs/label_regeneration_check.json` 和 `docs/legacy_threshold_label_check.json`。复核旧结果用冻结标签；从教师重新运行则全量生成新的标签，并作为新的实验记录。

所有启动命令可加 `--dry-run` 打印完整参数。`generate-labels --device cpu --limit 3` 可作教师推理检查；这种有限输出会被完整训练入口拒绝，不能冒充全量伪标签。`train-llp --epochs 1` 仍使用完整数据，但只检验缩短训练流程，不能作为正式结果。

## 5. 数据与代码位置

| 用途 | 目录 | 单视频形状 |
|---|---|---|
| UnAV 标注 | `data/UnAV/unav100_annotations.json` | 100 类，train/validation/test |
| UnAV 教师音频/视觉 | `data/UnAV/feats_CLAP/segment_feats`、`feats_CLIP/segment_feats` | `(L,512)` / `(L,768)` |
| UnAV 文本类别嵌入 | 各模态 `event_feats/all_event_feats.npy` | `(100,512)` / `(100,768)` |
| LLP 教师音频/视觉 | `pretrain/data/LLP/feats_CLAP`、`feats_CLIP` | 片段 `(10,512)` / `(10,768)`，文本为 25 类 |
| LLP 主模型音频 | `data/CLAP/features` | `(64,768)`，不能与教师 512 维特征混用 |
| LLP 主模型视觉 | `data/CLIP/features` | `(10,768)` |
| LLP 主模型时空 | `pretrain/data/LLP/feats/r2plus1d_18` | `(10,512)` |
| 冻结伪标签 | `data/CLAP/a_segmentwise_pseudo_labels`、`data/CLIP/v_segmentwise_pseudo_labels` | `(10,25)` |
| 默认教师阈值 | `pretrain/thresholds/llp/{a,v}_classwise_threhsolds.npy` | `(25,)`，完整验证集重新校准；历史文件名拼写保留 |
| 历史阈值存档 | `data/CLAP/{a,v}_classwise_threhsolds.npy` | 仅追溯旧数据，不作默认输入 |
| 新伪标签 | `artifacts/<run-id>/labels/{audio,visual}` | `(10,25)` |
| IPR 精炼标签 | `artifacts/<run-id>/pseudo_labels/round_2/{audio,visual}` | `(10,25)` 软标签 |

`teacher_train.py`、`calibrate_teacher.py`、`generate_labels.py` 是上游三阶段；`main_ipr.py`、`dataloader.py`、`nets/`、`nrep/` 是下游主流程。`main_baseline.py`、`main_uhnm.py` 和 `experiments/` 保留相关控制/依赖，默认入口不自动运行扩展实验。

所有推荐命令基于项目自身目录解析路径，不依赖原服务器的 `/root/...` 路径。完整验证范围、修正内容和未完成项见 [核验报告](docs/VERIFICATION.md)，数据来源和冻结结果见 [来源记录](docs/PROVENANCE.md)。
