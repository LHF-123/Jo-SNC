# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from data.anmal10n import Animal10N
from data.food101n import Food101N
from data.webvision import webvision_dataset
from utils.builder import build_cifar100n_dataset, build_transform, build_webfg_dataset, get_dataset_normalization
from utils.local_evidence import erase_by_mask, generate_cam
from utils.model import DualHeadModel
from utils.utils import set_seed


def load_config(cfg_path, data_root=None):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    if data_root is not None:
        cfg.data_root = data_root
    return cfg


def build_train_diagnostic_dataset(cfg):
    # 训练集诊断要尽量关闭随机增强，因此统一构建 test/eval 视图。
    transforms = build_transform(cfg.rescale_size, cfg.crop_size, dataset=cfg.dataset)
    root = os.path.join(cfg.data_root, cfg.dataset)

    if cfg.dataset.startswith('cifar100n') or cfg.dataset.startswith('cifar80n'):
        return build_cifar100n_dataset(
            os.path.join(cfg.data_root, 'cifar100'),
            transforms['cifar_train'],
            transforms['cifar_test'],
            getattr(cfg, 'noise_type', 'clean'),
            getattr(cfg, 'ood_noise_rate', 0.0),
            getattr(cfg, 'idn_noise_rate', 0.0),
        )['eval_train']

    if cfg.dataset == 'animal10n':
        return Animal10N(split='train', root_dir=root, transform=transforms['cifar_test'])

    if cfg.dataset in ['web-aircraft', 'web-bird', 'web-car']:
        return build_webfg_dataset(root, transforms['train'], transforms['test'])['eval_train']

    if cfg.dataset == 'food101n':
        return Food101N(root, transform=transforms['test'])

    if cfg.dataset in ['mini-webvision', 'webvision']:
        return webvision_dataset(root, transform=transforms['test'], mode='train', num_class=cfg.n_classes)

    raise NotImplementedError(f'{cfg.dataset} is not supported.')


def build_model(cfg, device):
    # 直接复用训练时的双头模型结构，checkpoint 加载后即可用于纯前向诊断。
    model = DualHeadModel(
        arch=getattr(cfg, 'arch', 'resnet50'),
        num_classes=cfg.n_classes,
        mlp_hidden=getattr(cfg, 'hdim', 2),
        feature_dim=getattr(cfg, 'fdim', 256),
        pretrained=False,
        use_bn=True,
    ).to(device)
    return model


def _strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def load_checkpoint(model, ckpt_path, device):
    # 同时兼容完整 checkpoint 和纯 state_dict，避免为诊断额外改训练保存逻辑。
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)}')
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    return model


def _forward_logits_and_feat(model, images):
    output = model(images)
    if isinstance(output, tuple):
        logits = output[0]
        feat = output[1] if len(output) > 1 else None
    else:
        logits = output
        feat = None
    return logits, feat


def _denormalize_image(image, norm_mean, norm_std):
    # 保存可视化前先反归一化，便于人工核查低 CAM 区域是否真像背景。
    if norm_mean is None or norm_std is None:
        return image.clamp(0, 1)
    mean = torch.tensor(norm_mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(norm_std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def _tensor_to_pil(image):
    image = image.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def _overlay_cam(image, cam, alpha=0.45):
    cam = cam.float()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    heatmap = torch.zeros_like(image)
    heatmap[0] = cam
    heatmap[1] = cam * 0.35
    return (image * (1 - alpha) + heatmap * alpha).clamp(0, 1)


def _mask_to_pil(mask):
    gray = mask.float().clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)
    return _tensor_to_pil(gray)


def _concat_images(images, gap=6):
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width + gap
    return canvas


def _draw_text_block(image, lines):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle((0, 0, canvas.width, 72), fill=(255, 255, 255))
    draw.text((8, 6), '\n'.join(lines), fill=(0, 0, 0), font=font)
    return canvas


def _get_sample_path(dataset, index):
    # 诊断图片命名尽量保留原始路径，方便回到训练集定位样本。
    if hasattr(dataset, 'samples'):
        sample = dataset.samples[index]
        if isinstance(sample, (tuple, list)) and sample:
            return str(sample[0])
        return str(sample)
    if hasattr(dataset, 'train_imgs'):
        return str(dataset.train_imgs[index])
    if hasattr(dataset, 'val_imgs'):
        return str(dataset.val_imgs[index])
    if hasattr(dataset, 'image_files'):
        return str(dataset.image_files[index])
    return ''


def _sanitize_name(name):
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ('-', '_', '.'):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep)[:120]


def _build_bg_view(images, bg_mask, fill_value=0.0):
    # x_bg 只保留低 CAM 区域，其余像素填充为 0，对应归一化后的均值。
    return erase_by_mask(images, ~bg_mask, fill_value=fill_value)


def _build_fg_view(images, bg_mask, fill_value=0.0):
    # x_fg 是背景的补集，用来检查前景视图是否仍然携带背景 shortcut。
    return erase_by_mask(images, bg_mask, fill_value=fill_value)


def _save_visualization(output_dir, sample_name, ratio, image, cam_up, bg_mask, x_bg, x_fg,
                        ori_pred, ori_conf, bg_pred, bg_conf, fg_pred, fg_conf, label,
                        norm_mean, norm_std):
    # 只保存少量样本，供人工判断 low-CAM 区域到底是背景还是目标弱部位。
    image_vis = _tensor_to_pil(_denormalize_image(image, norm_mean, norm_std))
    overlay_vis = _tensor_to_pil(_overlay_cam(_denormalize_image(image, norm_mean, norm_std), cam_up))
    mask_vis = _mask_to_pil(bg_mask)
    bg_vis = _tensor_to_pil(_denormalize_image(x_bg, norm_mean, norm_std))
    fg_vis = _tensor_to_pil(_denormalize_image(x_fg, norm_mean, norm_std))
    canvas = _concat_images([image_vis, overlay_vis, mask_vis, bg_vis, fg_vis])
    canvas = _draw_text_block(
        canvas,
        [
            f'label={int(label)}  ratio={ratio:.2f}',
            f'ori={int(ori_pred)} p={float(ori_conf):.3f}',
            f'bg ={int(bg_pred)} p={float(bg_conf):.3f}',
            f'fg ={int(fg_pred)} p={float(fg_conf):.3f}',
        ],
    )

    ratio_tag = f'low_{int(round(ratio * 100)):02d}'
    sample_tag = _sanitize_name(sample_name if sample_name else f'idx_{int(label)}')
    out_dir = os.path.join(output_dir, 'low_cam_bg_images', ratio_tag)
    os.makedirs(out_dir, exist_ok=True)
    canvas.save(os.path.join(out_dir, f'{sample_tag}.png'))


def run_diagnosis(cfg, args):
    set_seed(args.seed)
    device = torch.device(
        f'cuda:{str(args.gpu).strip()}'
        if torch.cuda.is_available() and str(args.gpu).strip().lower() != 'cpu'
        else 'cpu'
    )

    dataset = build_train_diagnostic_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_checkpoint(build_model(cfg, device), args.ckpt_path, device)
    model.eval()

    norm_mean, norm_std = get_dataset_normalization(cfg.dataset)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'low_cam_bg_images'), exist_ok=True)

    csv_path = os.path.join(output_dir, 'bg_diag.csv')
    summary_path = os.path.join(output_dir, 'bg_diag_summary.json')

    ratios = [float(x) for x in args.ratios.split(',') if str(x).strip()]
    sample_limit = args.max_samples if args.max_samples > 0 else None
    saved_counts = defaultdict(int)
    saved_class_counts = defaultdict(lambda: defaultdict(int))
    stats_by_ratio = {ratio: defaultdict(float) for ratio in ratios}
    count_by_ratio = {ratio: 0 for ratio in ratios}

    processed = 0
    writer = None
    with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv, torch.no_grad():
        for batch in loader:
            images = batch['data'].to(device, non_blocking=True)
            labels = batch['label'].long().to(device, non_blocking=True)
            indices = batch['index'].long().cpu().tolist()

            if sample_limit is not None and processed >= sample_limit:
                break
            if sample_limit is not None and processed + images.size(0) > sample_limit:
                keep = sample_limit - processed
                images = images[:keep]
                labels = labels[:keep]
                indices = indices[:keep]

            # 先做一次原图前向，既能拿到原图特征用于余弦诊断，也能支持 pred 作为 CAM 目标。
            ori_logits_pre, ori_feat = _forward_logits_and_feat(model, images)
            cam_targets = labels if args.cam_target == 'label' else ori_logits_pre.argmax(dim=1)

            # CAM 仍然按训练时的实现生成，保持诊断和训练语义一致。
            cam, logits_ori = generate_cam(model, images.float(), cam_targets, cam_type='weightcam')
            ori_probs = logits_ori.softmax(dim=1)
            ori_conf, ori_pred = ori_probs.max(dim=1)
            batch_count = images.size(0)

            cam_up = F.interpolate(cam[:, None, :, :], size=images.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

            for ratio in ratios:
                # 每张图独立取低响应分位数，避免不同样本的 CAM 尺度差异互相干扰。
                flat = cam_up.flatten(1)
                threshold = torch.quantile(flat, ratio, dim=1, keepdim=True)
                bg_mask = cam_up <= threshold.view(-1, 1, 1)
                bg_view = _build_bg_view(images, bg_mask, fill_value=0.0)
                fg_view = _build_fg_view(images, bg_mask, fill_value=0.0)

                bg_logits, bg_feat = _forward_logits_and_feat(model, bg_view)
                fg_logits, fg_feat = _forward_logits_and_feat(model, fg_view)

                bg_probs = bg_logits.softmax(dim=1)
                fg_probs = fg_logits.softmax(dim=1)
                bg_conf, bg_pred = bg_probs.max(dim=1)
                fg_conf, fg_pred = fg_probs.max(dim=1)

                p_ori_y = ori_probs.gather(1, labels[:, None]).squeeze(1)
                p_bg_y = bg_probs.gather(1, labels[:, None]).squeeze(1)
                p_fg_y = fg_probs.gather(1, labels[:, None]).squeeze(1)
                bg_drop = p_ori_y - p_bg_y
                fg_drop = p_ori_y - p_fg_y
                bg_area_ratio = bg_mask.float().mean(dim=(1, 2))

                feat_cos_ori_bg = F.cosine_similarity(ori_feat, bg_feat, dim=1) if ori_feat is not None and bg_feat is not None else None
                feat_cos_fg_bg = F.cosine_similarity(fg_feat, bg_feat, dim=1) if fg_feat is not None and bg_feat is not None else None

                for i in range(images.size(0)):
                    sample_name = _get_sample_path(dataset, indices[i])
                    row = {
                        'checkpoint': os.path.basename(args.ckpt_path),
                        'dataset': cfg.dataset,
                        'sample_id': int(indices[i]),
                        'sample_path': sample_name,
                        'ratio': float(ratio),
                        'cam_target': int(cam_targets[i].item()),
                        'label': int(labels[i].item()),
                        'ori_pred': int(ori_pred[i].item()),
                        'ori_conf': float(ori_conf[i].item()),
                        'ori_correct': int(ori_pred[i].item() == labels[i].item()),
                        'bg_pred': int(bg_pred[i].item()),
                        'bg_conf': float(bg_conf[i].item()),
                        'bg_correct': int(bg_pred[i].item() == labels[i].item()),
                        'bg_pred_eq_ori_pred': int(bg_pred[i].item() == ori_pred[i].item()),
                        'fg_pred': int(fg_pred[i].item()),
                        'fg_conf': float(fg_conf[i].item()),
                        'fg_correct': int(fg_pred[i].item() == labels[i].item()),
                        'p_ori_y': float(p_ori_y[i].item()),
                        'p_bg_y': float(p_bg_y[i].item()),
                        'p_fg_y': float(p_fg_y[i].item()),
                        'bg_drop': float(bg_drop[i].item()),
                        'fg_drop': float(fg_drop[i].item()),
                        'bg_area_ratio': float(bg_area_ratio[i].item()),
                        'cam_low_threshold': float(threshold[i].item()),
                        'feat_cos_ori_bg': float(feat_cos_ori_bg[i].item()) if feat_cos_ori_bg is not None else '',
                        'feat_cos_fg_bg': float(feat_cos_fg_bg[i].item()) if feat_cos_fg_bg is not None else '',
                    }

                    if writer is None:
                        writer = csv.DictWriter(f_csv, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)

                    stats = stats_by_ratio[ratio]
                    count_by_ratio[ratio] += 1
                    stats['ori_correct'] += row['ori_correct']
                    stats['bg_correct'] += row['bg_correct']
                    stats['bg_pred_eq_ori_pred'] += row['bg_pred_eq_ori_pred']
                    stats['p_bg_y_sum'] += row['p_bg_y']
                    stats['bg_conf_sum'] += row['bg_conf']
                    stats['bg_drop_sum'] += row['bg_drop']
                    stats['bg_area_sum'] += row['bg_area_ratio']
                    stats['p_bg_y_gt_05'] += float(row['p_bg_y'] > 0.5)
                    stats['p_bg_y_gt_07'] += float(row['p_bg_y'] > 0.7)
                    stats['feat_cos_ori_bg_sum'] += float(row['feat_cos_ori_bg']) if row['feat_cos_ori_bg'] != '' else 0.0
                    stats['feat_cos_fg_bg_sum'] += float(row['feat_cos_fg_bg']) if row['feat_cos_fg_bg'] != '' else 0.0
                    stats['feat_cos_count'] += float(row['feat_cos_ori_bg'] != '')

                    label_id = int(labels[i].item())
                    if args.save_images and saved_class_counts[ratio][label_id] < args.max_image_per_class:
                        # 按类限额保存，避免前面少数类别把所有可视化名额占满。
                        _save_visualization(
                            output_dir=output_dir,
                            sample_name=sample_name or f'idx_{indices[i]}',
                            ratio=ratio,
                            image=images[i].detach().cpu(),
                            cam_up=cam_up[i].detach().cpu(),
                            bg_mask=bg_mask[i].detach().cpu(),
                            x_bg=bg_view[i].detach().cpu(),
                            x_fg=fg_view[i].detach().cpu(),
                            ori_pred=ori_pred[i].detach().cpu(),
                            ori_conf=ori_conf[i].detach().cpu(),
                            bg_pred=bg_pred[i].detach().cpu(),
                            bg_conf=bg_conf[i].detach().cpu(),
                            fg_pred=fg_pred[i].detach().cpu(),
                            fg_conf=fg_conf[i].detach().cpu(),
                            label=labels[i].detach().cpu(),
                            norm_mean=norm_mean,
                            norm_std=norm_std,
                        )
                        saved_counts[ratio] += 1
                        saved_class_counts[ratio][label_id] += 1

                # processed 只统计真实处理过的样本数，每个 batch 只加一次，不能按 ratio 重复累加。
                processed += batch_count
                if sample_limit is not None and processed >= sample_limit:
                    break

    summary = {
        'dataset': cfg.dataset,
        'checkpoint': args.ckpt_path,
        'num_samples': processed,
        'ratios': {},
    }
    for ratio in ratios:
        stats = stats_by_ratio[ratio]
        count = max(1, count_by_ratio[ratio])
        feat_count = max(1, int(stats.get('feat_cos_count', 0)))
        summary['ratios'][str(ratio)] = {
            'ori_acc': stats['ori_correct'] / count,
            'bg_acc': stats['bg_correct'] / count,
            'bg_pred_eq_label_ratio': stats['bg_correct'] / count,
            'bg_pred_eq_ori_pred_ratio': stats['bg_pred_eq_ori_pred'] / count,
            'p_bg_y_mean': stats['p_bg_y_sum'] / count,
            'bg_conf_mean': stats['bg_conf_sum'] / count,
            'bg_drop_mean': stats['bg_drop_sum'] / count,
            'bg_area_ratio_mean': stats['bg_area_sum'] / count,
            'p_bg_y_gt_0.5_ratio': stats['p_bg_y_gt_05'] / count,
            'p_bg_y_gt_0.7_ratio': stats['p_bg_y_gt_07'] / count,
            'feat_cos_ori_bg_mean': stats['feat_cos_ori_bg_sum'] / feat_count,
            'feat_cos_fg_bg_mean': stats['feat_cos_fg_bg_sum'] / feat_count,
        }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'CSV saved to: {csv_path}')
    print(f'Summary saved to: {summary_path}')
    for ratio, stat in summary['ratios'].items():
        print(
            f'ratio={ratio} | bg_acc={stat["bg_acc"]:.4f} | '
            f'p_bg_y_mean={stat["p_bg_y_mean"]:.4f} | '
            f'bg_pred_eq_ori_pred={stat["bg_pred_eq_ori_pred_ratio"]:.4f} | '
            f'feat_cos_ori_bg={stat["feat_cos_ori_bg_mean"]:.4f}'
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Forward-only background shortcut diagnosis on the training split.')
    parser.add_argument('--cfg', type=str, required=True, help='训练配置 YAML。')
    parser.add_argument('--ckpt-path', type=str, required=True, help='C1 训练好的 checkpoint。')
    parser.add_argument('--data-root', type=str, default=None, help='覆盖配置里的 data_root。')
    parser.add_argument('--output-dir', type=str, default=None, help='诊断结果输出目录。')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id，或直接传 cpu。')
    parser.add_argument('--batch-size', type=int, default=64, help='诊断 batch size。')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader worker 数。')
    parser.add_argument('--seed', type=int, default=0, help='随机种子。')
    parser.add_argument('--ratios', type=str, default='0.1,0.2,0.3', help='低 CAM 区域分位数列表，用逗号分隔。')
    parser.add_argument('--cam-target', type=str, default='label', choices=['label', 'pred'], help='CAM 目标类别：训练标签或原图 top1。')
    parser.add_argument('--max-samples', type=int, default=0, help='最多诊断多少个样本，0 表示全量。')
    parser.add_argument('--save-images', action='store_true', help='保存少量可视化图片。')
    parser.add_argument('--max-image-per-class', type=int, default=20, help='每个 ratio 内每个类别最多保存多少张图片。')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg, args.data_root)

    if args.output_dir is None:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
        args.output_dir = os.path.join('results', 'bg_diag', cfg.dataset, f'{ckpt_name}-{stamp}')

    run_diagnosis(cfg, args)


if __name__ == '__main__':
    main()
