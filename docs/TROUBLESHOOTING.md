# 常见问题

## 数据文件仍是 LFS 指针

运行 `git lfs install --local` 和 `git lfs pull` 获取压缩包，然后执行 `python prepare_data.py --assets` 和 `python prepare_data.py --unav`。八个压缩包约 6.04 GB，代码、缓存和解压数据建议预留 25 GB 空间。

## UnAV 下载失败

`prepare_data.py --unav` 优先使用 `assets/` 中的压缩包。需要在线下载时，可通过 `--base-url https://hf-mirror.com` 指定镜像，通过 `--cache /path/to/cache` 指定缓存位置。代理配置错误时可加 `--ignore-proxy`；所有来源均按相同 SHA256 校验。

## 数据安装中断

重新运行 `python prepare_data.py --assets` 或 `python prepare_data.py --unav`。已安装内容按归档校验后跳过；新文件写入临时路径，完整写入后才发布。若提示某个已有文件内容不同，先检查该文件并移到备份位置，再运行安装。

## Conda 读取了额外镜像

`environment.yml` 使用 `nodefaults` 限定项目声明的渠道。可通过 `conda config --show-sources` 检查本机渠道、代理和 `channel_alias` 配置。安装日志会列出实际访问的渠道。

## 输出目录已经存在

每个运行 ID 对应独立目录。将 `--run-id` 改为新名称；逐阶段运行时，可将已完成阶段的 checkpoint 或标签目录传给下一阶段。日志位置为 `artifacts/<run-id>/launcher.log`，运行配置和状态为同目录的 `run.json`。

## 教师权重与阈值不匹配

为该 checkpoint 执行 `calibrate-teacher`，再将输出目录通过 `--threshold-dir` 传给 `generate-labels`。校准清单绑定教师权重、文本嵌入、验证划分和阈值哈希。

默认冻结教师的阈值位于 `pretrain/thresholds/llp`，由 649 个 LLP 验证视频校准。`data/CLAP` 下的旧阈值与冻结标签存在差异：六视频检查中有 19 个二值位置不同，见 [差异记录](legacy_threshold_label_check.json)。配套阈值的两个训练视频检查与缓存一致，全量等价性尚未验证。评估已发布主模型时使用冻结标签；训练新教师时使用对应校准流程生成标签。

## 主训练提示标签不完整

主训练需要 train、val、test 三个划分的标签。生成时使用默认的 `--split all`，省略 `--limit`；训练时将完整输出目录传给 `--label-root`。每个视频需有音频、视觉两个 `(10,25)` 数组。

## LLP 标注中的反向时间区间

音频标注 `8DF2PxdT5tE_390_400` 的 Cheering 事件为 `onset=9, offset=0`。解析器沿用空切片处理，具体条目见 [annotation_anomalies.json](annotation_anomalies.json)。其他非法时间区间会报告错误。

## 运行超时或收到 SIGKILL

单阶段可使用 `--timeout-hours` 调整时限；完整脚本使用 `TEACHER_HOURS` 和 `STAGE_HOURS`。超时会结束子进程组并记录失败状态，重新运行时使用新 ID。

SIGKILL 表示进程被外部终止。检查主机内存、容器限额和系统日志，减少并发任务；资源较少时可尝试 `--workers 0 --threads 1`（适用于主训练）。2 GiB 内存容器中存在教师推理被终止的记录，具体来源尚未定位。冻结教师采用 meta 初始化以减少加载时的重复内存占用。
