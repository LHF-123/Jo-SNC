# -*- coding: utf-8 -*-
import math
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from utils.model import MLPHead


LOCAL_EVIDENCE_CSV_HEADER = (
    'epoch,batch_idx,sample_id,group,noisy_label,cam_target,'
    'pred_top1,pred_conf,teacher_top1,teacher_conf,'
    'p_ori_y,p_part_y,p_erase_y,erase_drop,evidence_score,'
    'bbox_area,bbox_x1,bbox_y1,bbox_x2,bbox_y2,cam_quantile,cam_type,erase_fill'
)

PART_CE_CSV_HEADER = (
    'epoch,batch_idx,group,num_selected,num_valid,valid_part_ratio,'
    'josnc_loss,part_ce_loss,part_ce_weight,weighted_part_ce_loss,loss_ratio,'
    'p_ori_y_mean,p_part_y_mean,p_erase_y_mean,erase_drop_mean,'
    'evidence_score_mean,bbox_area_mean,num_gated,gate_ratio,'
    'gate_threshold,gated_evidence_score_mean,filtered_evidence_score_mean,'
    'gated_erase_drop_mean,filtered_erase_drop_mean'
)

PART_CE_GATE_SAMPLE_CSV_HEADER = (
    'epoch,batch_idx,sample_id,noisy_label,group,evidence_score,gate,'
    'p_ori_y,p_part_y,p_erase_y,erase_drop,bbox_area,pred_top1,pred_conf'
)


def format_local_evidence_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['sample_id']},{row['group']},"
        f"{row['noisy_label']},{row['cam_target']},{row['pred_top1']},"
        f"{row['pred_conf']:.6f},{row['teacher_top1']},{row['teacher_conf']:.6f},"
        f"{row['p_ori_y']:.6f},{row['p_part_y']:.6f},{row['p_erase_y']:.6f},"
        f"{row['erase_drop']:.6f},{row['evidence_score']:.6f},"
        f"{row['bbox_area']:.6f},{row['bbox_x1']},{row['bbox_y1']},"
        f"{row['bbox_x2']},{row['bbox_y2']},{row['cam_quantile']:.4f},"
        f"{row['cam_type']},{row['erase_fill']}"
    )


def format_part_ce_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['group']},"
        f"{row['num_selected']},{row['num_valid']},{row['valid_part_ratio']:.6f},"
        f"{row['josnc_loss']:.6f},{row['part_ce_loss']:.6f},"
        f"{row['part_ce_weight']:.6f},{row['weighted_part_ce_loss']:.6f},"
        f"{row['loss_ratio']:.6f},{row['p_ori_y_mean']:.6f},"
        f"{row['p_part_y_mean']:.6f},{row['p_erase_y_mean']:.6f},"
        f"{row['erase_drop_mean']:.6f},{row['evidence_score_mean']:.6f},"
        f"{row['bbox_area_mean']:.6f},{row['num_gated']},"
        f"{row['gate_ratio']:.6f},{row['gate_threshold']:.6f},"
        f"{row['gated_evidence_score_mean']:.6f},"
        f"{row['filtered_evidence_score_mean']:.6f},"
        f"{row['gated_erase_drop_mean']:.6f},"
        f"{row['filtered_erase_drop_mean']:.6f}"
    )


def format_part_ce_gate_sample_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['sample_id']},"
        f"{row['noisy_label']},{row['group']},{row['evidence_score']:.6f},"
        f"{row['gate']},{row['p_ori_y']:.6f},{row['p_part_y']:.6f},"
        f"{row['p_erase_y']:.6f},{row['erase_drop']:.6f},"
        f"{row['bbox_area']:.6f},{row['pred_top1']},{row['pred_conf']:.6f}"
    )


def generate_cam(model, images, cam_targets, cam_type='weightcam'):
    if cam_type != 'weightcam':
        raise ValueError(f'{cam_type} CAM is not supported for A1 local evidence diagnostics.')

    # A1 默认使用无反传的 weight-CAM，避免 Grad-CAM 影响训练图和显存。
    spatial_features, classifier_features = _extract_spatial_features(model, images)
    logits, class_weights = _classifier_weights(model.classifier, classifier_features)
    batch_indices = torch.arange(images.size(0), device=images.device)
    target_weights = class_weights[batch_indices, cam_targets]

    cam = _build_weight_cam(spatial_features, target_weights)
    return _normalize_cam(cam), logits


def cam_to_bbox(cam, image_size, quantile=0.8, min_area=0.05, max_area=0.7, padding=0.05):
    image_h, image_w = image_size
    bboxes, areas, masks = [], [], []
    cam = cam.float()
    for cam_i in cam:
        bbox, mask = _single_cam_to_bbox(
            cam_i, image_h, image_w, quantile=quantile,
            min_area=min_area, max_area=max_area, padding=padding
        )
        x1, y1, x2, y2 = bbox
        area = ((x2 - x1) * (y2 - y1)) / float(image_h * image_w)
        bboxes.append(bbox)
        areas.append(area)
        masks.append(mask)
    return bboxes, areas, torch.stack(masks, dim=0).to(device=cam.device)


def crop_by_bbox(images, bboxes):
    crops = []
    _, _, image_h, image_w = images.shape
    for image, bbox in zip(images, bboxes):
        x1, y1, x2, y2 = bbox
        crop = image[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(crop, size=(image_h, image_w), mode='bilinear', align_corners=False)
        crops.append(crop.squeeze(0))
    return torch.stack(crops, dim=0)


def erase_by_mask(images, masks, fill_value=0.0):
    erased = images.clone()
    erased = erased.masked_fill(masks[:, None, :, :].bool(), fill_value)
    return erased


def compute_evidence_values(logits_ori, logits_part, logits_erase, labels):
    # C1 统一计算 teacher/no-grad 局部证据，保证门控分数和日志字段使用同一套定义。
    label_indices = labels.long()
    p_ori_y = logits_ori.detach().softmax(dim=1).gather(1, label_indices[:, None]).squeeze(1)
    p_part_y = logits_part.detach().softmax(dim=1).gather(1, label_indices[:, None]).squeeze(1)
    p_erase_y = logits_erase.detach().softmax(dim=1).gather(1, label_indices[:, None]).squeeze(1)
    erase_drop = p_ori_y - p_erase_y
    evidence_score = p_ori_y * p_part_y * erase_drop.clamp(min=0)
    return {
        'p_ori_y': p_ori_y,
        'p_part_y': p_part_y,
        'p_erase_y': p_erase_y,
        'erase_drop': erase_drop,
        'evidence_score': evidence_score,
    }


def build_gate_mask(evidence_score, gate_type='percentile', threshold=0.10, keep_ratio=0.50):
    # C1 支持固定阈值和 batch 内分位门控；返回实际使用的阈值便于写日志。
    evidence_score = evidence_score.detach()
    gate_type = str(gate_type).lower()
    if evidence_score.numel() == 0:
        return evidence_score.new_zeros((0,), dtype=torch.bool), evidence_score.new_tensor(0.0)

    if gate_type == 'fixed':
        gate_threshold = evidence_score.new_tensor(float(threshold))
        return evidence_score > gate_threshold, gate_threshold

    if gate_type == 'percentile':
        keep_ratio = max(0.0, min(float(keep_ratio), 1.0))
        if keep_ratio <= 0:
            return torch.zeros_like(evidence_score, dtype=torch.bool), evidence_score.new_tensor(0.0)
        k = max(1, int(evidence_score.numel() * keep_ratio))
        k = min(k, evidence_score.numel())
        topk_values, topk_indices = torch.topk(evidence_score, k, largest=True, sorted=False)
        gate_mask = torch.zeros_like(evidence_score, dtype=torch.bool)
        # percentile 按 top-k 索引严格保留 k 个样本，避免 evidence 并列时超额放行。
        gate_mask[topk_indices] = True
        gate_threshold = topk_values.min()
        return gate_mask, gate_threshold

    raise ValueError(f'part_ce_gate_type should be fixed or percentile, got {gate_type}.')


def build_local_part_batch(
        model,
        images,
        labels,
        selected_indices,
        cam_quantile=0.8,
        min_area=0.05,
        max_area=0.7,
        bbox_padding=0.05,
        cam_type='weightcam'):
    selected_indices = selected_indices.detach().long()
    num_selected = int(selected_indices.numel())

    # B1 只用 CAM/bbox 定位局部区域；bbox 是离散裁剪依据，不参与反传。
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_disabled(images):
            cam, logits_ori = generate_cam(model, images.float(), labels, cam_type=cam_type)
            bboxes, bbox_areas, erase_masks = cam_to_bbox(
                cam, images.shape[-2:], quantile=cam_quantile,
                min_area=min_area, max_area=max_area, padding=bbox_padding
            )
            x_part = crop_by_bbox(images.float(), bboxes)
            x_erase = erase_by_mask(images.float(), erase_masks, fill_value=0.0)
            logits_part = _extract_logits(model(x_part))
            logits_erase = _extract_logits(model(x_erase))
            evidence_values = compute_evidence_values(logits_ori, logits_part, logits_erase, labels)
    finally:
        if was_training:
            model.train()

    bbox_areas = torch.tensor(bbox_areas, device=images.device, dtype=images.dtype)
    selected_bbox_areas = bbox_areas[selected_indices] if num_selected > 0 else bbox_areas[:0]
    valid_mask = torch.isfinite(selected_bbox_areas) & (selected_bbox_areas > 0)
    valid_indices = selected_indices[valid_mask]

    return {
        'x_part': x_part[valid_indices],
        'x_erase': x_erase[valid_indices],
        'labels': labels[valid_indices],
        'selected_indices': selected_indices,
        'valid_mask': valid_mask,
        'batch_indices': valid_indices,
        'bbox_area': bbox_areas[valid_indices],
        'p_ori_y': evidence_values['p_ori_y'][valid_indices],
        'p_part_y': evidence_values['p_part_y'][valid_indices],
        'p_erase_y': evidence_values['p_erase_y'][valid_indices],
        'erase_drop': evidence_values['erase_drop'][valid_indices],
        'evidence_score': evidence_values['evidence_score'][valid_indices],
        'num_selected': num_selected,
        'num_valid': int(valid_indices.numel()),
    }


def build_part_ce_log_row(epoch, batch_idx, group, part_batch, josnc_loss,
                          part_ce_loss, part_ce_weight):
    num_selected = part_batch['num_selected']
    num_valid = part_batch['num_valid']
    josnc_loss_value, part_ce_loss_value, weighted_part_ce_loss, loss_ratio = _part_ce_loss_values(
        josnc_loss, part_ce_loss, part_ce_weight
    )
    if num_selected == 0 or num_valid == 0:
        return _empty_part_ce_log_row(
            epoch, batch_idx, group, num_selected, num_valid,
            josnc_loss_value, part_ce_loss_value, part_ce_weight,
            weighted_part_ce_loss, loss_ratio
        )

    # B1/C1 日志使用 build_local_part_batch 中的局部证据，C1 时与实际门控完全一致。
    p_ori_y = part_batch['p_ori_y'].detach()
    p_part_y = part_batch['p_part_y'].detach()
    p_erase_y = part_batch['p_erase_y'].detach()
    erase_drop = part_batch['erase_drop'].detach()
    evidence_score = part_batch['evidence_score'].detach()
    gate_mask = part_batch.get('gate_mask')
    if gate_mask is None:
        gate_mask = torch.ones(num_valid, device=evidence_score.device, dtype=torch.bool)
    else:
        gate_mask = gate_mask.detach().to(device=evidence_score.device, dtype=torch.bool)
    filtered_mask = gate_mask.logical_not()
    num_gated = int(gate_mask.sum().item())
    gate_threshold = _float_value(part_batch.get('gate_threshold', 0.0))

    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'group': group,
        'num_selected': int(num_selected),
        'num_valid': int(num_valid),
        'valid_part_ratio': float(num_valid / max(num_selected, 1)),
        'josnc_loss': josnc_loss_value,
        'part_ce_loss': part_ce_loss_value,
        'part_ce_weight': float(part_ce_weight),
        'weighted_part_ce_loss': weighted_part_ce_loss,
        'loss_ratio': loss_ratio,
        'p_ori_y_mean': float(p_ori_y.mean().item()),
        'p_part_y_mean': float(p_part_y.mean().item()),
        'p_erase_y_mean': float(p_erase_y.mean().item()),
        'erase_drop_mean': float(erase_drop.mean().item()),
        'evidence_score_mean': float(evidence_score.mean().item()),
        'bbox_area_mean': float(part_batch['bbox_area'].detach().mean().item()),
        'num_gated': num_gated,
        'gate_ratio': float(num_gated / max(num_valid, 1)),
        'gate_threshold': gate_threshold,
        'gated_evidence_score_mean': _mean_or_zero(evidence_score[gate_mask]),
        'filtered_evidence_score_mean': _mean_or_zero(evidence_score[filtered_mask]),
        'gated_erase_drop_mean': _mean_or_zero(erase_drop[gate_mask]),
        'filtered_erase_drop_mean': _mean_or_zero(erase_drop[filtered_mask]),
    }


def build_part_ce_gate_sample_rows(epoch, batch_idx, group, part_batch, sample_indices,
                                   student_logits=None):
    if part_batch['num_valid'] == 0:
        return []

    # C1 逐样本日志记录实际 CE gate，便于追踪长期被过滤的样本和证据分布。
    batch_positions = part_batch['batch_indices'].detach().cpu().long()
    if torch.is_tensor(sample_indices):
        sample_ids = sample_indices.detach().cpu().long()[batch_positions]
    else:
        sample_ids = torch.tensor(sample_indices, dtype=torch.long)[batch_positions]

    labels = part_batch['labels'].detach().cpu().long()
    p_ori_y = part_batch['p_ori_y'].detach().cpu()
    p_part_y = part_batch['p_part_y'].detach().cpu()
    p_erase_y = part_batch['p_erase_y'].detach().cpu()
    erase_drop = part_batch['erase_drop'].detach().cpu()
    evidence_score = part_batch['evidence_score'].detach().cpu()
    bbox_area = part_batch['bbox_area'].detach().cpu()
    gate_mask = part_batch.get('gate_mask')
    if gate_mask is None:
        gate_mask = torch.ones(part_batch['num_valid'], dtype=torch.bool)
    else:
        gate_mask = gate_mask.detach().cpu().bool()

    pred_top1 = torch.full((part_batch['num_valid'],), -1, dtype=torch.long)
    pred_conf = torch.zeros(part_batch['num_valid'], dtype=torch.float)
    if student_logits is not None:
        student_probs = student_logits.detach().softmax(dim=1).cpu()
        pred_conf, pred_top1 = student_probs[batch_positions].max(dim=1)

    rows = []
    for i in range(part_batch['num_valid']):
        rows.append({
            'epoch': int(epoch),
            'batch_idx': int(batch_idx),
            'sample_id': int(sample_ids[i].item()),
            'noisy_label': int(labels[i].item()),
            'group': group,
            'evidence_score': float(evidence_score[i].item()),
            'gate': int(gate_mask[i].item()),
            'p_ori_y': float(p_ori_y[i].item()),
            'p_part_y': float(p_part_y[i].item()),
            'p_erase_y': float(p_erase_y[i].item()),
            'erase_drop': float(erase_drop[i].item()),
            'bbox_area': float(bbox_area[i].item()),
            'pred_top1': int(pred_top1[i].item()),
            'pred_conf': float(pred_conf[i].item()),
        })
    return rows


def _part_ce_loss_values(josnc_loss, part_ce_loss, part_ce_weight):
    josnc_loss_value = float(josnc_loss.detach().item())
    part_ce_loss_value = float(part_ce_loss.detach().item())
    weighted_part_ce_loss = float(part_ce_weight) * part_ce_loss_value
    # B1 记录加权局部 CE 相对主 Jo-SNC loss 的比例，用来判断分支实际强度。
    loss_ratio = weighted_part_ce_loss / max(abs(josnc_loss_value), 1e-12)
    return josnc_loss_value, part_ce_loss_value, weighted_part_ce_loss, loss_ratio


def _empty_part_ce_log_row(epoch, batch_idx, group, num_selected, num_valid,
                           josnc_loss, part_ce_loss, part_ce_weight,
                           weighted_part_ce_loss, loss_ratio):
    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'group': group,
        'num_selected': int(num_selected),
        'num_valid': int(num_valid),
        'valid_part_ratio': 0.0,
        'josnc_loss': float(josnc_loss),
        'part_ce_loss': float(part_ce_loss),
        'part_ce_weight': float(part_ce_weight),
        'weighted_part_ce_loss': float(weighted_part_ce_loss),
        'loss_ratio': float(loss_ratio),
        'p_ori_y_mean': 0.0,
        'p_part_y_mean': 0.0,
        'p_erase_y_mean': 0.0,
        'erase_drop_mean': 0.0,
        'evidence_score_mean': 0.0,
        'bbox_area_mean': 0.0,
        'num_gated': 0,
        'gate_ratio': 0.0,
        'gate_threshold': 0.0,
        'gated_evidence_score_mean': 0.0,
        'filtered_evidence_score_mean': 0.0,
        'gated_erase_drop_mean': 0.0,
        'filtered_erase_drop_mean': 0.0,
    }


def _mean_or_zero(values):
    if values.numel() == 0:
        return 0.0
    return float(values.detach().mean().item())


def _float_value(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def save_local_evidence_images(output_dir, rows, images, cam, x_part, x_erase, norm_mean, norm_std, max_samples=8):
    # 图片导出是可选诊断能力，默认关闭，避免每个 batch 产生大量 PNG。
    if output_dir is None or max_samples <= 0:
        return
    os.makedirs(output_dir, exist_ok=True)

    image_h, image_w = images.shape[-2:]
    cam_up = F.interpolate(cam[:, None, :, :], size=(image_h, image_w), mode='bilinear', align_corners=False).squeeze(1)
    for i, row in enumerate(rows[:max_samples]):
        # 每个样本保存一张横向拼图：原图+bbox、CAM 叠加、局部图、擦除图。
        original = _tensor_to_pil(_denormalize_image(images[i], norm_mean, norm_std))
        part = _tensor_to_pil(_denormalize_image(x_part[i], norm_mean, norm_std))
        erase = _tensor_to_pil(_denormalize_image(x_erase[i], norm_mean, norm_std))
        overlay = _tensor_to_pil(_overlay_cam(_denormalize_image(images[i], norm_mean, norm_std), cam_up[i]))

        bbox = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])
        _draw_bbox(original, bbox)
        _draw_bbox(overlay, bbox)

        grid = _concat_images([original, overlay, part, erase])
        filename = (
            f"epoch_{row['epoch']:03d}_batch_{row['batch_idx']:05d}_"
            f"sample_{row['sample_id']}_group_{row['group']}_"
            f"y_{row['noisy_label']}_cam_{row['cam_target']}.png"
        )
        grid.save(os.path.join(output_dir, filename))


def compute_local_evidence(
        model,
        images,
        labels,
        sample_indices,
        idx_clean,
        idx_id,
        idx_ood,
        epoch,
        batch_idx,
        cam_quantile=0.8,
        min_area=0.05,
        max_area=0.7,
        bbox_padding=0.05,
        cam_type='weightcam',
        student_logits=None,
        teacher_logits=None,
        save_images=False,
        image_dir=None,
        image_max_samples=8,
        norm_mean=None,
        norm_std=None):
    # A1 诊断只读：临时固定 BN/dropout 行为，返回前恢复调用方的训练状态。
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_disabled(images):
            cam_targets = labels
            cam, logits_ori = generate_cam(model, images.float(), cam_targets, cam_type=cam_type)
            bboxes, bbox_areas, erase_masks = cam_to_bbox(
                cam, images.shape[-2:], quantile=cam_quantile,
                min_area=min_area, max_area=max_area, padding=bbox_padding
            )
            x_part = crop_by_bbox(images.float(), bboxes)
            x_erase = erase_by_mask(images.float(), erase_masks, fill_value=0.0)

            logits_part = _extract_logits(model(x_part))
            logits_erase = _extract_logits(model(x_erase))
            probs_ori = logits_ori.softmax(dim=1)
            probs_part = logits_part.softmax(dim=1)
            probs_erase = logits_erase.softmax(dim=1)
    finally:
        if was_training:
            model.train()

    rows = _build_rows(
        probs_ori=probs_ori,
        probs_part=probs_part,
        probs_erase=probs_erase,
        labels=labels,
        sample_indices=sample_indices,
        idx_clean=idx_clean,
        idx_id=idx_id,
        idx_ood=idx_ood,
        epoch=epoch,
        batch_idx=batch_idx,
        cam_quantile=cam_quantile,
        cam_type=cam_type,
        bboxes=bboxes,
        bbox_areas=bbox_areas,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
    )
    if save_images:
        save_local_evidence_images(
            image_dir, rows, images.float(), cam, x_part, x_erase,
            norm_mean=norm_mean, norm_std=norm_std, max_samples=image_max_samples
        )
    return rows


def _extract_spatial_features(model, images):
    encoder = model.encoder
    if hasattr(encoder, 'encoder') and isinstance(encoder.encoder, nn.Sequential):
        modules = list(encoder.encoder.children())
        if len(modules) < 1:
            raise ValueError('weightcam requires a CNN encoder with spatial features.')
        if isinstance(modules[-1], (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            # ResNet 路径：最后一层是 GAP，CAM 使用通道权重和空间特征图。
            spatial_features = _forward_modules(images, modules[:-1])
            classifier_features = modules[-1](spatial_features).view(images.size(0), -1)
        else:
            # CIFAR CNN 路径：分类器直接接 flatten 特征，CAM 使用每个空间位置的线性贡献。
            spatial_features = _forward_modules(images, modules)
            classifier_features = spatial_features.view(images.size(0), -1)
    elif hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'feature_encoder'):
        # VGG 路径：显式 feature encoder + avg pool，等价于 GAP-CAM。
        spatial_features = encoder.encoder.feature_encoder(images)
        classifier_features = encoder.encoder.avg_pool(spatial_features).view(images.size(0), -1)
    else:
        raise ValueError('weightcam currently supports torchvision ResNet-style and VGG encoders.')

    if classifier_features.size(1) != encoder.feature_dim:
        raise ValueError(
            f'weightcam needs classifier feature dim {classifier_features.size(1)} to match '
            f'feature dim {encoder.feature_dim}.'
        )
    return spatial_features, classifier_features


def _forward_modules(images, modules):
    features = images
    for module in modules:
        features = module(features)
    return features


def _build_weight_cam(spatial_features, target_weights):
    batch_size, channels, height, width = spatial_features.shape
    if target_weights.size(1) == channels:
        # ResNet/VGG：GAP 后的分类权重只有通道维度，按通道加权得到 CAM。
        cam = (spatial_features * target_weights[:, :, None, None]).sum(dim=1)
    elif target_weights.size(1) == channels * height * width:
        # CIFAR CNN：分类权重对应 flatten 后的 C*H*W，reshape 回空间贡献图。
        spatial_weights = target_weights.view(batch_size, channels, height, width)
        cam = (spatial_features * spatial_weights).sum(dim=1)
    else:
        raise ValueError(
            f'weightcam target weight dim {target_weights.size(1)} does not match '
            f'channel dim {channels} or flattened dim {channels * height * width}.'
        )
    return F.relu(cam)


def _denormalize_image(image, norm_mean, norm_std):
    # 保存图片前反归一化；如果没有均值方差信息，则退化为直接裁剪到 [0, 1]。
    if norm_mean is None or norm_std is None:
        return image.clamp(0, 1)
    mean = torch.tensor(norm_mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(norm_std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def _overlay_cam(image, cam, alpha=0.45):
    # 用轻量红黄热力图叠加 CAM，避免新增 matplotlib 依赖和额外绘图状态。
    cam = cam.float()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    heatmap = torch.zeros_like(image)
    heatmap[0] = cam
    heatmap[1] = cam * 0.35
    return (image * (1 - alpha) + heatmap * alpha).clamp(0, 1)


def _tensor_to_pil(image):
    # PIL 保存需要 HWC uint8，内部仍保持 torch tensor 参与前面的可视化计算。
    image = image.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def _draw_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=(0, 255, 0), width=3)


def _concat_images(images, gap=4):
    # 横向拼接便于人工同时检查原图、CAM、局部图和擦除图。
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    grid = Image.new('RGB', (width, height), color=(255, 255, 255))
    offset = 0
    for image in images:
        grid.paste(image, (offset, 0))
        offset += image.width + gap
    return grid


def _classifier_weights(classifier, pooled_features):
    if isinstance(classifier, nn.Linear):
        logits = classifier(pooled_features)
        weights = classifier.weight.unsqueeze(0).expand(pooled_features.size(0), -1, -1)
        return logits, weights

    if isinstance(classifier, MLPHead):
        modules = list(classifier.mlp_head.children())
        first_linear = modules[0]
        last_linear = modules[-1]
        if not isinstance(first_linear, nn.Linear) or not isinstance(last_linear, nn.Linear):
            raise ValueError('weightcam supports MLPHead with linear input and output layers only.')

        # 默认 MLP head 需要无梯度地推导局部线性权重，避免 A1 依赖 Grad-CAM/backward。
        hidden = first_linear(pooled_features)
        bn_scale = torch.ones(hidden.size(1), device=hidden.device, dtype=hidden.dtype)
        for module in modules[1:-1]:
            if isinstance(module, nn.BatchNorm1d):
                hidden = F.batch_norm(
                    hidden, module.running_mean, module.running_var,
                    module.weight, module.bias, training=False, eps=module.eps
                )
                bn_scale = module.weight / torch.sqrt(module.running_var + module.eps)
                bn_scale = bn_scale.to(device=hidden.device, dtype=hidden.dtype)
            elif isinstance(module, nn.ReLU):
                relu_mask = (hidden > 0).to(hidden.dtype)
                hidden = F.relu(hidden)
            else:
                hidden = module(hidden)

        if 'relu_mask' not in locals():
            relu_mask = torch.ones_like(hidden)
        logits = last_linear(hidden)
        hidden_weights = last_linear.weight.unsqueeze(0) * relu_mask.unsqueeze(1) * bn_scale.view(1, 1, -1)
        weights = torch.matmul(hidden_weights, first_linear.weight)
        return logits, weights

    raise ValueError(f'weightcam does not support classifier type {classifier.__class__.__name__}.')


def _normalize_cam(cam, eps=1e-6):
    flat = cam.flatten(1)
    cam_min = flat.min(dim=1)[0].view(-1, 1, 1)
    cam_max = flat.max(dim=1)[0].view(-1, 1, 1)
    return (cam - cam_min) / (cam_max - cam_min + eps)


def _single_cam_to_bbox(cam, image_h, image_w, quantile, min_area, max_area, padding):
    cam_h, cam_w = cam.shape
    if cam.max() <= 0:
        bbox = _center_bbox(image_h, image_w, min_area)
        return bbox, _bbox_mask(image_h, image_w, bbox, cam.device)

    threshold = torch.quantile(cam.flatten(), quantile)
    cam_mask = (cam >= threshold) & (cam > 0)
    if not cam_mask.any():
        bbox = _center_bbox(image_h, image_w, min_area)
        return bbox, _bbox_mask(image_h, image_w, bbox, cam.device)

    ys, xs = torch.where(cam_mask)
    x1 = math.floor(xs.min().item() * image_w / cam_w)
    x2 = math.ceil((xs.max().item() + 1) * image_w / cam_w)
    y1 = math.floor(ys.min().item() * image_h / cam_h)
    y2 = math.ceil((ys.max().item() + 1) * image_h / cam_h)
    bbox = _pad_bbox((x1, y1, x2, y2), image_h, image_w, padding)
    bbox = _fit_bbox_area(bbox, image_h, image_w, min_area, max_area)

    image_mask = F.interpolate(
        cam_mask.float().view(1, 1, cam_h, cam_w),
        size=(image_h, image_w),
        mode='nearest'
    ).view(image_h, image_w).bool()
    if not image_mask.any():
        image_mask = _bbox_mask(image_h, image_w, bbox, cam.device)
    return bbox, image_mask


def _pad_bbox(bbox, image_h, image_w, padding):
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    pad_x = int(round(width * padding))
    pad_y = int(round(height * padding))
    return _clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_h, image_w)


def _fit_bbox_area(bbox, image_h, image_w, min_area, max_area):
    x1, y1, x2, y2 = bbox
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    area_ratio = width * height / float(image_h * image_w)
    if min_area <= area_ratio <= max_area:
        return _clamp_bbox(bbox, image_h, image_w)

    target_area = min_area if area_ratio < min_area else max_area
    scale = math.sqrt((target_area * image_h * image_w) / float(width * height))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if area_ratio < min_area:
        # bbox 太小时向上取整，避免局部图退化成极小区域。
        new_w = max(1, int(math.ceil(width * scale)))
        new_h = max(1, int(math.ceil(height * scale)))
    else:
        # bbox 太大时向下取整，并在后面用硬约束防止整数取整后仍超上限。
        new_w = max(1, int(math.floor(width * scale)))
        new_h = max(1, int(math.floor(height * scale)))

    fitted = _resize_bbox_around_center(cx, cy, new_w, new_h, image_h, image_w)
    if area_ratio > max_area:
        fitted = _shrink_bbox_to_max_area(fitted, image_h, image_w, max_area)
    return fitted


def _resize_bbox_around_center(cx, cy, width, height, image_h, image_w):
    return _clamp_bbox((
        int(round(cx - width / 2.0)),
        int(round(cy - height / 2.0)),
        int(round(cx + width / 2.0)),
        int(round(cy + height / 2.0)),
    ), image_h, image_w)


def _shrink_bbox_to_max_area(bbox, image_h, image_w, max_area):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = x2 - x1, y2 - y1
    while width * height / float(image_h * image_w) > max_area and width > 1 and height > 1:
        # 取整后的 bbox 仍可能略大于上限，逐步缩小可保证日志面积不越界。
        if width >= height:
            width -= 1
        else:
            height -= 1
    return _resize_bbox_around_center(cx, cy, width, height, image_h, image_w)


def _center_bbox(image_h, image_w, area_ratio):
    side = math.sqrt(max(area_ratio, 1e-6))
    width = max(1, int(round(image_w * side)))
    height = max(1, int(round(image_h * side)))
    x1 = (image_w - width) // 2
    y1 = (image_h - height) // 2
    return _clamp_bbox((x1, y1, x1 + width, y1 + height), image_h, image_w)


def _clamp_bbox(bbox, image_h, image_w):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(x1 + 1, min(image_w, x2))
    y2 = max(y1 + 1, min(image_h, y2))
    return x1, y1, x2, y2


def _bbox_mask(image_h, image_w, bbox, device):
    x1, y1, x2, y2 = bbox
    mask = torch.zeros((image_h, image_w), dtype=torch.bool, device=device)
    mask[y1:y2, x1:x2] = True
    return mask


def _extract_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _build_rows(
        probs_ori,
        probs_part,
        probs_erase,
        labels,
        sample_indices,
        idx_clean,
        idx_id,
        idx_ood,
        epoch,
        batch_idx,
        cam_quantile,
        cam_type,
        bboxes,
        bbox_areas,
        student_logits,
        teacher_logits):
    groups = _group_names(labels.size(0), idx_clean, idx_id, idx_ood)
    label_indices = labels.long()
    p_ori_y = probs_ori.gather(1, label_indices[:, None]).squeeze(1)
    p_part_y = probs_part.gather(1, label_indices[:, None]).squeeze(1)
    p_erase_y = probs_erase.gather(1, label_indices[:, None]).squeeze(1)
    erase_drop = p_ori_y - p_erase_y
    evidence_score = p_ori_y * p_part_y * erase_drop.clamp(min=0)

    student_probs = probs_ori if student_logits is None else student_logits.detach().softmax(dim=1)
    teacher_probs = probs_ori if teacher_logits is None else teacher_logits.detach().softmax(dim=1)
    pred_conf, pred_top1 = student_probs.max(dim=1)
    teacher_conf, teacher_top1 = teacher_probs.max(dim=1)

    rows = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        rows.append({
            'epoch': int(epoch),
            'batch_idx': int(batch_idx),
            'sample_id': int(sample_indices[i].item()),
            'group': groups[i],
            'noisy_label': int(labels[i].item()),
            'cam_target': int(labels[i].item()),
            'pred_top1': int(pred_top1[i].item()),
            'pred_conf': float(pred_conf[i].item()),
            'teacher_top1': int(teacher_top1[i].item()),
            'teacher_conf': float(teacher_conf[i].item()),
            'p_ori_y': float(p_ori_y[i].item()),
            'p_part_y': float(p_part_y[i].item()),
            'p_erase_y': float(p_erase_y[i].item()),
            'erase_drop': float(erase_drop[i].item()),
            'evidence_score': float(evidence_score[i].item()),
            'bbox_area': float(bbox_areas[i]),
            'bbox_x1': int(x1),
            'bbox_y1': int(y1),
            'bbox_x2': int(x2),
            'bbox_y2': int(y2),
            'cam_quantile': float(cam_quantile),
            'cam_type': cam_type,
            'erase_fill': 'norm_zero',
        })
    return rows


def _group_names(batch_size, idx_clean, idx_id, idx_ood):
    groups = ['unknown'] * batch_size
    for name, indices in [('clean', idx_clean), ('ID', idx_id), ('OOD', idx_ood)]:
        for index in indices.detach().cpu().tolist():
            groups[int(index)] = name
    return groups


def _autocast_disabled(images):
    if images.is_cuda:
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()
