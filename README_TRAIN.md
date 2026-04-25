# 训练与评估启动命令

本文档只说明 Jo-SNC 的常用训练、评估启动方式，以及日志相关参数的含义。

## 环境准备

建议使用 Python 3.9 环境安装依赖：

```bash
pip install -r requirements.txt
```

仓库中的启动脚本是 Bash 脚本。Linux、WSL、Git Bash 可以直接运行；如果在 PowerShell 中使用，建议直接执行等价的 `python main.py` 命令。

## 参数优先级

`main.py` 的参数读取顺序是：

1. 先读取 `--cfg config/*.yaml`
2. 再用命令行参数覆盖 YAML 中的同名字段

因此，使用 `scripts/*.sh` 启动时，脚本里传入的参数优先级高于 YAML 文件。

## 指定数据集位置

YAML 中的 `data_root` 控制数据集根目录，例如：

```yaml
data_root: ../datasets/
dataset: web-bird
```

此时 Web-Bird 的实际读取路径是：

```text
../datasets/web-bird/
```

训练和评估也支持用命令行覆盖数据集根目录：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg config/bird.yaml \
  --data-root /path/to/datasets \
  --gpu 0
```

```bash
python demo.py \
  --cfg config/bird.yaml \
  --model-path web_bird.pth \
  --data-root /path/to/datasets \
  --gpu 0
```

## 使用脚本训练

脚本第一个参数是物理 GPU id：

```bash
bash scripts/bird.sh 0
bash scripts/aircraft.sh 0
bash scripts/car.sh 0
bash scripts/food101n.sh 0
bash scripts/animal10n.sh 0
bash scripts/webvision_mini.sh 0
```

CIFAR noisy 实验：

```bash
bash scripts/cifar.sh 0
```

## 直接启动训练

使用 `config/bird.yaml` 中的 Web-Bird 参数训练：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg config/bird.yaml \
  --gpu 0 \
  --log-proj benchmark_webfg \
  --log-name resnet50-web-bird \
  --save-model \
  --enable-progress-bar True
```

Web-Aircraft 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg config/aircraft.yaml \
  --gpu 0 \
  --arch resnet50 \
  --batch-size 64 \
  --lr 0.01 \
  --epochs 120 \
  --log-proj benchmark_webfg \
  --log-name resnet50-web-aircraft \
  --save-model
```

说明：

- `--cfg` 指定基础 YAML 配置文件。
- `--gpu` 是进程内可见的 CUDA 设备编号。
- `--save-model` 会保存 `model_best.pth` 和 `model_last.pth`。
- `--save-ckpt` 会额外保存可恢复训练的 checkpoint 状态。
- `--enable-progress-bar True` 会显示 `tqdm` 进度条。

## 断点续跑

训练时添加 `--save-ckpt` 会保存 `checkpoint-latest.pth`：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg config/bird.yaml \
  --gpu 0 \
  --log-proj benchmark_webfg \
  --log-name resnet50-web-bird \
  --save-model \
  --save-ckpt
```

恢复训练时使用 `--ckpt-path` 指向 checkpoint 文件：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg config/bird.yaml \
  --gpu 0 \
  --ckpt-path ../results/benchmark_webfg/<run-dir>/checkpoint-latest.pth \
  --log-proj benchmark_webfg \
  --log-name resnet50-web-bird-resume \
  --save-model \
  --save-ckpt
```

checkpoint 会保存主模型、动量模型、optimizer、AMP scaler、队列状态、阈值状态、最佳准确率和 epoch。恢复时会从 `checkpoint['epoch'] + 1` 继续训练。

## 评估模型

使用 `demo.py` 加载配置和 checkpoint：

```bash
python demo.py --cfg config/bird.yaml --model-path web_bird.pth --gpu 0
python demo.py --cfg config/aircraft.yaml --model-path web_aircraft.pth --gpu 0
python demo.py --cfg config/car.yaml --model-path web_car.pth --gpu 0
python demo.py --cfg config/food101n.yaml --model-path food101n.pth --gpu 0
```

miniWebVision 使用 Inception-ResNet-v2 时：

```bash
python demo.py \
  --cfg config/mini_webvision.yaml \
  --model-path mini_webvision_inception_resnetv2.pth \
  --arch InceptionResNetV2 \
  --gpu 0
```

`demo.py` 会根据 YAML 构建数据集，读取 `--model-path` 指定的权重，并输出测试准确率。

## 日志参数说明

训练日志由 `main.py` 中的 `build_logger` 创建。

- `--log-proj`：实验组名称，例如 `benchmark_webfg`。
- `--log-name`：单次实验名称，例如 `resnet50-web-bird`。
- `--enable-progress-bar True`：是否显示训练和评估进度条。
- `--save-model`：保存最佳模型和最后一轮模型。
- `--save-ckpt`：保存包含 optimizer、epoch 等信息的恢复训练 checkpoint。

当使用：

```bash
--log-proj benchmark_webfg --log-name resnet50-web-bird
```

输出目录格式为：

```text
../results/benchmark_webfg/web-bird-resnet50-web-bird-<timestamp>/
```

如果 `log_proj` 中包含 `ablation`，目录格式为：

```text
<log_root>/<dataset>/<log_proj>/<log_name>-<timestamp>/
```

如果 `log_proj` 中包含 `benchmark`，目录格式为：

```text
<log_root>/<log_proj>/<dataset>-<log_name>-<timestamp>/
```

其他情况目录格式为：

```text
<log_root>/<dataset>/<log_proj>/<timestamp>-<log_name>/
```

## 主要输出文件

一次训练通常会生成：

```text
log.txt
debug-log.txt
msg-log.txt
config.yaml
network.txt
test_acc.csv
threshold.csv
prfa_metric.csv
pll_topk_acc.csv
loss_acc_curve.png
pr_curve.png
model_best.pth
model_last.pth
```

含义：

- `log.txt`：每轮训练 loss、训练准确率、测试准确率、运行时间、最佳准确率。
- `debug-log.txt`：学习率、动态 `topK` 等调试信息。
- `msg-log.txt`：重要提示和样本检测指标。
- `config.yaml`：实际生效的配置，包括命令行覆盖后的参数。
- `network.txt`：模型结构。
- `test_acc.csv`：每轮测试准确率。
- `threshold.csv`：clean 和 OOD 阈值变化。
- `prfa_metric.csv`：clean / ID noisy / OOD noisy 的 P、R、F1、AUROC。
- `pll_topk_acc.csv`：partial-label top-1 / top-k 匹配统计。
- `model_best.pth`、`model_last.pth`：仅在启用 `--save-model` 时生成。

训练结束后，结果目录会被重命名，目录名中会追加 best accuracy 和 mean accuracy。

## 常见注意点

- 如果设置 `CUDA_VISIBLE_DEVICES=2`，进程内只看到一张卡，此时仍应传 `--gpu 0`。
- Web-Aircraft、Web-Bird、Web-Car 使用 ImageFolder 风格目录，需要包含 `train/` 和 `val/`。
- warmup 阶段的 `omega_tau` 在 `main.py` 中固定为 `0.75`。
- warmup 之后的 `omega_tau` 由 `tau_m` 控制，可在 YAML 中设置 `tau_m: 0.98`，也可用命令行传 `--tau-m 0.98`。

## Web-Bird 数据集要求

`web-bird` 不使用单独的标注文件，代码会通过 `IndexedImageFolder` 从目录结构生成标签。要求如下：

```text
<data_root>/
  web-bird/
    train/
      class_001/
        image_001.jpg
      class_002/
        image_002.jpg
      ...
    val/
      class_001/
        image_101.jpg
      class_002/
        image_102.jpg
      ...
```

需要满足：

- `config/bird.yaml` 中 `dataset` 必须是 `web-bird`。
- `n_classes` 应为 `200`。
- `train/` 和 `val/` 下都应按类别建子目录。
- 训练集和验证集的类别目录名要一致，避免类别编号不匹配。
- 支持的图片后缀包括 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp` 等。
- 类别编号由文件夹名排序后自动生成，不需要额外的 label txt 或 json。
