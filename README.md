# EviPLS：UnAV 教师 → LLP 伪标签 → U-HNM + IPR

EviPLS 是用于音视频事件解析（AVVP）的训练项目，提供 UnAV 教师训练、LLP 伪标签生成和 U-HNM + IPR 主模型训练。仓库包含特征数据、预训练权重和统一命令行入口 `reproduce.py`。

流程以已提取的 CLAP、CLIP、R2Plus1D 特征为输入。教师使用 UnAV 时序标注训练，逐类阈值使用 LLP 验证集的模态时序标注校准，主模型使用生成的伪标签训练。

## 安装

需要 Linux x86_64、Git、Git LFS 和 Conda。Ubuntu/Debian 可通过 `sudo apt-get install git git-lfs` 安装 Git 工具；Conda 可使用 Miniconda。完整训练还需要可用的 NVIDIA GPU 和兼容 CUDA 11.3 的驱动。

```bash
# 分别下载代码和 LFS 数据。
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Jacksonwuhu/EviPLS.git
cd EviPLS
git lfs install --local
git lfs pull

conda env create --solver libmamba -f environment.yml
conda activate evipls-main-v3
python prepare_data.py --assets
python prepare_data.py --unav
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

GPU 训练前，上述环境检查应输出 `CUDA available: True`。

`assets/` 是 Git LFS 管理的分块压缩包，包含 LLP 全链路输入及已有权重；`prepare_data.py` 校验 SHA256 后安装到项目目录。UnAV 的两份原始特征压缩包也随 Git LFS 提供；`--unav` 优先读取本地包，缺失时从固定 UWAV 作者源下载，并按固定 SHA256 校验后分别解压到两个模态目录。仓库包含原始 UnAV JSON 标注。安装中断后可重复执行安装命令；已安装文件会检查一致性后跳过，内容冲突时会给出文件路径。

网络无法连接 Hugging Face 时，可显式指定兼容镜像，校验标准不变：

```bash
python prepare_data.py --unav --base-url https://hf-mirror.com
```

下载预算默认为 2 小时，最多下载约 2.18 GB UnAV 压缩数据。可用 `--cache /path/to/cache` 分离缓存位置；错误代理环境可加 `--ignore-proxy`。安装程序检查文件完整性，并在文件缺失、损坏、仍为 LFS 指针或与已有内容冲突时报告具体错误。磁盘建议为代码、LFS 缓存、压缩包、解压数据共预留 25 GB，正式训练另留权重空间。

环境配置：Linux / Python 3.10 / PyTorch 1.12.1 / torchvision 0.13.1 / CUDA 11.3。`requirements-runtime.txt` 固定直接依赖版本，传递依赖由包管理器解析。完整训练使用 GPU；CPU 可用于输入检查、推理及小批量流程测试。测试命令与参考结果见[测试指南](docs/TESTING.md)。

## 数据检查

```bash
python verify.py --installed
python -m nrep.validation --json
python reproduce.py train-teacher --run-id input_check --check-only
python scripts/audit_inputs.py --output artifacts/input_audit.json
```

检查包括：LLP 10,000/649/1,200 三个划分、11 位视频键不冲突、全部输入形状及有限值、全部伪标签范围、UnAV train/validation 的标注与类别顺序、CLAP/CLIP 特征。`audit_inputs.py` 还检查 LLP 教师输入，检查范围为数据读取、形状、数值和文件哈希。

## 训练

先预览五个阶段的命令，再启动训练：

```bash
DRY_RUN=1 RUN_ID=unav_to_llp_s6 PYTHON_BIN=python bash scripts/run_full_pipeline.sh
```

```bash
RUN_ID=unav_to_llp_s6 PYTHON_BIN=python bash scripts/run_full_pipeline.sh
```

脚本首先检查全部训练输入，随后运行以下流程：UnAV 教师 80 epoch → LLP 验证集逐类阈值校准 → 全部 LLP 视频伪标签 → 两轮 U-HNM + IPR 主训练 → 最终测试。教师种子 1000，主训练种子 6；可通过 `SEED=17` 等调整主训练种子。

教师阶段默认最多 24 小时，后续每阶段最多 4 小时；可以显式设置 `TEACHER_HOURS`、`STAGE_HOURS`。超时会终止子进程组，失败记录保留。每阶段有独立 `artifacts/<run-id>/run.json` 和 `launcher.log`。请为每次运行设置新的运行 ID；同名运行会报错。

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

教师按 UnAV validation Seg-AV 选模。每个新教师在 649 个 LLP 验证视频上校准配套阈值，并通过哈希绑定权重和阈值文件。伪标签生成阶段读取特征、视频级标签和配套阈值，输出保存到新的运行目录。

主训练默认设置为 `loss_semantics=legacy`、每轮音频/视觉各 10 epoch、两轮、batch 128、lr 5e-4、IPR η=0.3。权重选择和超参数调整依据验证集；训练器在每轮末额外记录测试指标。

## 预训练模型与伪标签

```bash
# 使用配套阈值为所有 LLP 划分生成标签，输出到新目录。
python reproduce.py generate-labels --run-id frozen_teacher_labels
python reproduce.py train-llp --run-id regenerated_s6 \
  --label-root artifacts/frozen_teacher_labels/labels

# 直接使用冻结伪标签训练。
python reproduce.py train-llp --run-id frozen_labels_s6 --seed 6

# 复核已保存的主模型；CPU 主机自动使用 CPU。
python reproduce.py evaluate-llp --run-id frozen_test_s6 \
  --checkpoint weights/llp/seed6_round2.pt
```

默认教师使用 `pretrain/thresholds/llp` 中的配套阈值。复现已发布权重的结果时，使用随数据包提供的冻结伪标签；训练新教师时，运行校准和生成命令创建对应标签。阈值选择与缓存差异见[常见问题](docs/TROUBLESHOOTING.md)。

所有启动命令可加 `--dry-run` 打印完整参数。`generate-labels --device cpu --limit 3` 用于三个视频的教师推理检查。完整训练入口要求所有划分的标签齐全；用于训练的标签请按上面的默认命令全量生成。`train-llp --epochs 1` 使用完整数据，将每个训练阶段缩短为 1 epoch，便于检查训练流程。

## 项目结构

| 路径 | 功能 |
|---|---|
| `reproduce.py` | 训练、校准、标签生成与评估入口 |
| `teacher_train.py`、`calibrate_teacher.py`、`generate_labels.py` | 教师训练及伪标签流程 |
| `main_ipr.py`、`nets/`、`nrep/` | U-HNM + IPR 主训练和模型组件 |
| `main_baseline.py`、`main_uhnm.py`、`experiments/` | 对照方法和调度实验 |
| `prepare_data.py`、`verify.py` | 数据安装与完整性检查 |
| `data/`、`pretrain/data/`、`assets/` | 标注、特征和分发包 |
| `tests/`、`scripts/smoke_*.py` | 单元测试和小样本训练检查 |
| `artifacts/`、`models/`、`logger/` | 运行配置、训练产物和日志 |

所有推荐命令基于仓库根目录解析相对路径。

## 文档

- [数据目录与格式](docs/DATA_LAYOUT.md)
- [测试与参考结果](docs/TESTING.md)
- [常见问题](docs/TROUBLESHOOTING.md)
- [数据与代码来源](docs/PROVENANCE.md)
