# Jo-SNC

Jo-SNC is a PyTorch implementation of **Combating Noisy Labels through Fostering Self- and Neighbor-Consistency**. The code trains noisy-label classifiers, separates clean / in-distribution noisy / out-of-distribution noisy samples, and evaluates trained checkpoints on supported datasets.

![JoSNC](assets/josnc.jpeg)

## Environment

Recommended environment:

- Python 3.9
- CUDA-enabled PyTorch
- `torch==1.12.1`
- `torchvision==0.13.1`

Install dependencies:

```bash
pip install -r requirements.txt
```

The launch scripts are Bash scripts. On Windows, run them through WSL or Git Bash, or call `python main.py` directly from PowerShell.

## Dataset Layout

Place datasets outside the code directory. The default configs use `data_root: ../datasets/` and logs use `log_root: ../results/`.

Expected layout:

```text
project_root/
  datasets/
    web-aircraft/
      train/
      val/
    web-bird/
      train/
      val/
    web-car/
      train/
      val/
    animal10n/
    food101n/
    mini-webvision/
  results/
  code/
    main.py
    demo.py
    config/
    scripts/
```

`web-aircraft`, `web-bird`, and `web-car` use an ImageFolder-style loader: each class is represented by a subdirectory under `train/` and `val/`.

## Training

The main training entry point is `main.py`. It first reads the YAML config passed with `--cfg`, then overwrites matching fields with command-line arguments. Therefore, values in `scripts/*.sh` have higher priority than values in YAML files.

Run a prepared script with one GPU id:

```bash
bash scripts/bird.sh 0
bash scripts/aircraft.sh 0
bash scripts/car.sh 0
bash scripts/food101n.sh 0
```

Run training directly:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg config/bird.yaml \
  --gpu 0 \
  --log-proj benchmark_webfg \
  --log-name resnet50-web-bird \
  --save-model \
  --enable-progress-bar True
```

For Web-Bird, the YAML now contains the paper-style settings: ResNet50, SGD, `lr=0.01`, `batch_size=64`, 5 warmup epochs, 120 total epochs, `eps=0.3`, `fdim=512`, `queue_length=32000`, `n_neighbors=10`, `topK=10`, and `tau_m=0.98`. Warmup `omega_tau=0.75` is fixed in `main.py`; after warmup, `omega_tau` is controlled by `tau_m`.

Use `--save-model` to write `model_best.pth` and `model_last.pth`. Use `--save-ckpt` only when optimizer state and epoch resume data are needed.

## Evaluation

Download a checkpoint from the model zoo or use a checkpoint produced by training. Evaluate with `demo.py`:

```bash
python demo.py --cfg config/bird.yaml --model-path web_bird.pth --gpu 0
python demo.py --cfg config/aircraft.yaml --model-path web_aircraft.pth --gpu 0
python demo.py --cfg config/car.yaml --model-path web_car.pth --gpu 0
```

`demo.py` builds the dataset from the YAML file, loads the checkpoint with `--model-path`, and prints test accuracy. For WebFG datasets, it automatically uses ResNet50 during evaluation.

## Logging Parameters

Training logs are created by `build_logger` in `main.py`.

- `log_root`: base output directory. Default configs use `../results/`.
- `log_proj`: experiment group name. It controls the log directory pattern.
- `log_name`: run name used inside the final directory name.
- `enable_progress_bar`: enables `tqdm` progress bars during training and evaluation. In YAML use `true` or `false`; on CLI pass `--enable-progress-bar True` to enable it.

Directory rules:

```text
if "ablation" in log_proj:
  <log_root>/<dataset>/<log_proj>/<log_name>-<timestamp>/

elif "benchmark" in log_proj:
  <log_root>/<log_proj>/<dataset>-<log_name>-<timestamp>/

else:
  <log_root>/<dataset>/<log_proj>/<timestamp>-<log_name>/
```

Example:

```bash
python main.py --cfg config/bird.yaml --gpu 0 \
  --log-proj benchmark_webfg \
  --log-name resnet50-topK10
```

This creates a directory similar to:

```text
../results/benchmark_webfg/web-bird-resnet50-topK10-20260425213000/
```

## Training Outputs

Each training run stores:

- `log.txt`: epoch-level train loss, train accuracy, test accuracy, runtime, and best accuracy.
- `debug-log.txt`: debug details such as epoch learning rate and dynamic `topK`.
- `msg-log.txt`: important messages and sample detection metrics.
- `config.yaml`: the effective config after command-line overrides.
- `network.txt`: model architecture.
- `threshold.csv`: clean and OOD threshold history.
- `test_acc.csv`: per-epoch test accuracy.
- `prfa_metric.csv`: clean / ID noisy / OOD noisy precision, recall, F1, and AUROC when detection labels are available.
- `pll_topk_acc.csv`: partial-label top-1 and top-k matching statistics.
- `loss_acc_curve.png` and `pr_curve.png`: generated plots.
- `model_best.pth` and `model_last.pth`: saved only when `--save-model` is enabled.

At the end of training, the result directory is renamed with best and mean accuracy statistics.

## Model Zoo

| Dataset | Backbone | Checkpoint |
| --- | --- | --- |
| Web-Aircraft | ResNet50 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/web_aircraft.pth) |
| Web-Bird | ResNet50 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/web_bird.pth) |
| Web-Car | ResNet50 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/web_car.pth) |
| Animal-10N | VGG19 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/animal10n.pth) |
| miniWebVision | ResNet50 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/mini_webvision_resnet50.pth) |
| miniWebVision | Inception-ResNet-v2 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/mini_webvision_inception_resnetv2.pth) |
| Food101N | ResNet50 | [model](https://josnc.oss-cn-shanghai.aliyuncs.com/food101n.pth) |

## Notes

- `config/*.yaml` provides defaults; command-line arguments override those values.
- For `web-aircraft`, `web-bird`, and `web-car`, class ids come from folder order in the ImageFolder dataset.
- Keep datasets, checkpoints, logs, and local environments out of Git. `.gitignore` already excludes `.env/`, `*.log`, `*.pth`, and `*.pt`.
