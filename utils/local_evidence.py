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

LOCAL_MARGIN_CSV_HEADER = (
    'epoch,batch_idx,group,num_selected,num_valid,num_margin_valid,margin_valid_ratio,'
    'raw_local_margin_loss,weighted_local_margin_loss,weighted_local_margin_loss_ratio,'
    'part_target_logit_mean,part_hardneg_logit_mean,part_margin_mean,'
    'part_margin_p25,part_margin_p50,part_margin_p75,part_margin_violate_ratio,'
    'hardneg_class_eq_pred_ratio'
)

EVIDENCE_CLEAN_REWEIGHT_CSV_HEADER = (
    'epoch,batch_idx,num_clean,num_valid,num_invalid,num_high,num_low,'
    'valid_clean_ratio,invalid_ratio,high_ratio_valid,low_ratio_valid,'
    'high_weight,low_weight,clean_weight_mean,raw_clean_loss_mean,'
    'weighted_clean_loss_mean,loss_reduction_ratio,high_raw_clean_loss_mean,'
    'low_raw_clean_loss_mean,invalid_raw_clean_loss_mean,evidence_score_mean,'
    'high_evidence_score_mean,low_evidence_score_mean,p_part_y_high_mean,'
    'p_part_y_low_mean,erase_drop_high_mean,erase_drop_low_mean,num_id,num_ood'
)

ID_CANDIDATE_CSV_HEADER = (
    'epoch,batch_idx,num_selected,num_valid,id_candidate_loss,'
    'id_candidate_weight,weighted_id_candidate_loss,loss_ratio,'
    'candidate_topk,candidate_entropy_mean,student_candidate_mass_mean,'
    'bbox_area_mean,candidate_size_mean,student_candidate_entropy_mean,'
    'entropy_penalty_mean,candidate_dist_loss_mean,candidate_kl_mean,'
    'student_top1_candidate_mass_mean,student_noisy_label_mass_mean,'
    'target_top1_candidate_mass_mean,target_noisy_label_mass_mean,'
    'effective_id_candidate_weight,num_conf_filtered,conf_filter_ratio,'
    'teacher_top1_prob_mean,teacher_top1_prob_gated_mean,'
    'teacher_top1_prob_filtered_mean'
)

ID_CANDIDATE_SAMPLE_CSV_HEADER = (
    'epoch,batch_idx,sample_id,noisy_label,cam_target,candidate_set,'
    'top1_candidate,candidate_scores,candidate_entropy,pll_loss,bbox_area,'
    'pred_top1,pred_conf,noisy_label_in_candidate,top1_candidate_eq_noisy_label,'
    'candidate_size,student_candidate_entropy,entropy_penalty,'
    'candidate_dist_loss,candidate_kl,student_top1_candidate_mass,'
    'student_noisy_label_mass,target_top1_candidate_mass,target_noisy_label_mass,'
    'teacher_top1_prob,conf_gate,effective_id_candidate_weight,'
    'used_in_id_candidate_loss,skip_reason'
)

MULTI_PART_CSV_HEADER = (
    'epoch,batch_idx,group,part_id,num_selected,num_valid,valid_part_ratio,'
    'num_parts,bbox_area_mean,part_iou_max_mean,part_iou_prev_mean,'
    'p_part_target_mean,p_part_label_mean,part_conf_mean,erase_accum_area_mean,'
    'crop_mode,erase_mode,window_ratio'
)

MULTI_PART_SAMPLE_CSV_HEADER = (
    'epoch,batch_idx,sample_id,group,label,part_id,cam_target,cam_target_source,'
    'bbox_x1,bbox_y1,bbox_x2,bbox_y2,bbox_area,part_pred,part_conf,'
    'p_part_target,p_part_label,part_iou_prev,part_iou_max,erase_accum_area,'
    'part_rank_conf,valid_part,erase_round,crop_mode,erase_mode,window_ratio'
)

D2_DIAG_CSV_HEADER = (
    'epoch,batch_idx,lambda,num_clean,num_c1_gated,num_part_valid_samples,'
    'valid_sample_ratio,valid_part_per_sample,part1_valid_ratio,part2_valid_ratio,part3_valid_ratio,'
    'global_acc_all_c1_gated,global_acc_valid_subset,top1_final_acc,weighted_final_acc,'
    'top1_acc_delta,weighted_acc_delta,changed_correct_top1,changed_wrong_top1,'
    'changed_correct_weighted,changed_wrong_weighted,valid_p_part_target_mean,'
    'filtered_p_part_target_mean,valid_quality_score_mean,filtered_quality_score_mean,'
    'valid_marginal_drop_mean,filtered_marginal_drop_mean,global_logit_norm_mean,'
    'part_logit_norm_mean'
)

D2_DIAG_SAMPLE_CSV_HEADER = (
    'epoch,batch_idx,lambda,sample_id,label,global_pred,global_conf,'
    'global_correct_valid_subset,selected_parts_top1,selected_parts_weighted,'
    'part_quality_scores,part_valid_mask,part_p_targets,part_marginal_drops,part_iou_max,'
    'best_part_idx,best_part_quality_score,top1_final_pred,top1_final_conf,'
    'top1_final_correct,weighted_final_pred,weighted_final_conf,weighted_final_correct,'
    'top1_changed,weighted_changed'
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


def format_local_margin_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['group']},"
        f"{row['num_selected']},{row['num_valid']},{row['num_margin_valid']},"
        f"{row['margin_valid_ratio']:.6f},{row['raw_local_margin_loss']:.6f},"
        f"{row['weighted_local_margin_loss']:.6f},"
        f"{row['weighted_local_margin_loss_ratio']:.6f},"
        f"{row['part_target_logit_mean']:.6f},"
        f"{row['part_hardneg_logit_mean']:.6f},"
        f"{row['part_margin_mean']:.6f},{row['part_margin_p25']:.6f},"
        f"{row['part_margin_p50']:.6f},{row['part_margin_p75']:.6f},"
        f"{row['part_margin_violate_ratio']:.6f},"
        f"{row['hardneg_class_eq_pred_ratio']:.6f}"
    )


def format_evidence_clean_reweight_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['num_clean']},"
        f"{row['num_valid']},{row['num_invalid']},{row['num_high']},"
        f"{row['num_low']},{row['valid_clean_ratio']:.6f},"
        f"{row['invalid_ratio']:.6f},{row['high_ratio_valid']:.6f},"
        f"{row['low_ratio_valid']:.6f},{row['high_weight']:.6f},"
        f"{row['low_weight']:.6f},{row['clean_weight_mean']:.6f},"
        f"{row['raw_clean_loss_mean']:.6f},{row['weighted_clean_loss_mean']:.6f},"
        f"{row['loss_reduction_ratio']:.6f},"
        f"{row['high_raw_clean_loss_mean']:.6f},"
        f"{row['low_raw_clean_loss_mean']:.6f},"
        f"{row['invalid_raw_clean_loss_mean']:.6f},"
        f"{row['evidence_score_mean']:.6f},"
        f"{row['high_evidence_score_mean']:.6f},"
        f"{row['low_evidence_score_mean']:.6f},"
        f"{row['p_part_y_high_mean']:.6f},"
        f"{row['p_part_y_low_mean']:.6f},"
        f"{row['erase_drop_high_mean']:.6f},"
        f"{row['erase_drop_low_mean']:.6f},"
        f"{row['num_id']},{row['num_ood']}"
    )


def format_id_candidate_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['num_selected']},"
        f"{row['num_valid']},{row['id_candidate_loss']:.6f},"
        f"{row['id_candidate_weight']:.6f},"
        f"{row['weighted_id_candidate_loss']:.6f},"
        f"{row['loss_ratio']:.6f},{row['candidate_topk']},"
        f"{row['candidate_entropy_mean']:.6f},"
        f"{row['student_candidate_mass_mean']:.6f},"
        f"{row['bbox_area_mean']:.6f},"
        f"{row['candidate_size_mean']:.6f},"
        f"{row['student_candidate_entropy_mean']:.6f},"
        f"{row['entropy_penalty_mean']:.6f},"
        f"{row['candidate_dist_loss_mean']:.6f},"
        f"{row['candidate_kl_mean']:.6f},"
        f"{row['student_top1_candidate_mass_mean']:.6f},"
        f"{row['student_noisy_label_mass_mean']:.6f},"
        f"{row['target_top1_candidate_mass_mean']:.6f},"
        f"{row['target_noisy_label_mass_mean']:.6f},"
        f"{row['effective_id_candidate_weight']:.6f},"
        f"{row['num_conf_filtered']},{row['conf_filter_ratio']:.6f},"
        f"{row['teacher_top1_prob_mean']:.6f},"
        f"{row['teacher_top1_prob_gated_mean']:.6f},"
        f"{row['teacher_top1_prob_filtered_mean']:.6f}"
    )


def format_id_candidate_sample_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['sample_id']},"
        f"{row['noisy_label']},{row['cam_target']},{row['candidate_set']},"
        f"{row['top1_candidate']},{row['candidate_scores']},"
        f"{row['candidate_entropy']:.6f},{row['pll_loss']:.6f},"
        f"{row['bbox_area']:.6f},{row['pred_top1']},"
        f"{row['pred_conf']:.6f},{row['noisy_label_in_candidate']},"
        f"{row['top1_candidate_eq_noisy_label']},"
        f"{row['candidate_size']},{row['student_candidate_entropy']:.6f},"
        f"{row['entropy_penalty']:.6f},{row['candidate_dist_loss']:.6f},"
        f"{row['candidate_kl']:.6f},{row['student_top1_candidate_mass']:.6f},"
        f"{row['student_noisy_label_mass']:.6f},"
        f"{row['target_top1_candidate_mass']:.6f},"
        f"{row['target_noisy_label_mass']:.6f},"
        f"{row['teacher_top1_prob']:.6f},{row['conf_gate']},"
        f"{row['effective_id_candidate_weight']:.6f},"
        f"{row['used_in_id_candidate_loss']},{row['skip_reason']}"
    )


def format_multi_part_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['group']},"
        f"{row['part_id']},{row['num_selected']},{row['num_valid']},"
        f"{row['valid_part_ratio']:.6f},{row['num_parts']},"
        f"{row['bbox_area_mean']:.6f},{row['part_iou_max_mean']:.6f},"
        f"{row['part_iou_prev_mean']:.6f},{row['p_part_target_mean']:.6f},"
        f"{row['p_part_label_mean']:.6f},{row['part_conf_mean']:.6f},"
        f"{row['erase_accum_area_mean']:.6f},{row['crop_mode']},"
        f"{row['erase_mode']},{row['window_ratio']:.6f}"
    )


def format_multi_part_sample_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['sample_id']},"
        f"{row['group']},{row['label']},{row['part_id']},"
        f"{row['cam_target']},{row['cam_target_source']},"
        f"{row['bbox_x1']},{row['bbox_y1']},{row['bbox_x2']},{row['bbox_y2']},"
        f"{row['bbox_area']:.6f},{row['part_pred']},"
        f"{row['part_conf']:.6f},{row['p_part_target']:.6f},"
        f"{row['p_part_label']:.6f},{row['part_iou_prev']:.6f},"
        f"{row['part_iou_max']:.6f},{row['erase_accum_area']:.6f},"
        f"{row['part_rank_conf']},{row['valid_part']},{row['erase_round']},"
        f"{row['crop_mode']},{row['erase_mode']},{row['window_ratio']:.6f}"
    )


def format_d2_diag_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['lambda']:.6f},"
        f"{row['num_clean']},{row['num_c1_gated']},{row['num_part_valid_samples']},"
        f"{row['valid_sample_ratio']:.6f},{row['valid_part_per_sample']:.6f},"
        f"{row['part1_valid_ratio']:.6f},{row['part2_valid_ratio']:.6f},"
        f"{row['part3_valid_ratio']:.6f},{row['global_acc_all_c1_gated']:.6f},"
        f"{row['global_acc_valid_subset']:.6f},{row['top1_final_acc']:.6f},"
        f"{row['weighted_final_acc']:.6f},{row['top1_acc_delta']:.6f},"
        f"{row['weighted_acc_delta']:.6f},{row['changed_correct_top1']},"
        f"{row['changed_wrong_top1']},{row['changed_correct_weighted']},"
        f"{row['changed_wrong_weighted']},{row['valid_p_part_target_mean']:.6f},"
        f"{row['filtered_p_part_target_mean']:.6f},"
        f"{row['valid_quality_score_mean']:.6f},"
        f"{row['filtered_quality_score_mean']:.6f},"
        f"{row['valid_marginal_drop_mean']:.6f},"
        f"{row['filtered_marginal_drop_mean']:.6f},"
        f"{row['global_logit_norm_mean']:.6f},{row['part_logit_norm_mean']:.6f}"
    )


def format_d2_diag_sample_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['lambda']:.6f},"
        f"{row['sample_id']},{row['label']},{row['global_pred']},"
        f"{row['global_conf']:.6f},{row['global_correct_valid_subset']},"
        f"{row['selected_parts_top1']},{row['selected_parts_weighted']},"
        f"{row['part_quality_scores']},{row['part_valid_mask']},"
        f"{row['part_p_targets']},{row['part_marginal_drops']},"
        f"{row['part_iou_max']},{row['best_part_idx']},"
        f"{row['best_part_quality_score']:.6f},{row['top1_final_pred']},"
        f"{row['top1_final_conf']:.6f},{row['top1_final_correct']},"
        f"{row['weighted_final_pred']},{row['weighted_final_conf']:.6f},"
        f"{row['weighted_final_correct']},{row['top1_changed']},"
        f"{row['weighted_changed']}"
    )


def compute_id_candidate_effective_weight(
        base_weight,
        epoch,
        start_epoch,
        decay_start_epoch=0,
        decay_end_epoch=0,
        min_weight=0.0):
    # C2-v4 在启动前不加 loss，启动后按可选线性衰减控制后期正则强度。
    base_weight = float(base_weight)
    min_weight = float(min_weight)
    epoch = int(epoch)
    start_epoch = int(start_epoch)
    decay_start_epoch = int(decay_start_epoch)
    decay_end_epoch = int(decay_end_epoch)

    if epoch < start_epoch:
        return 0.0
    if decay_end_epoch <= decay_start_epoch or decay_start_epoch <= 0:
        return base_weight
    if epoch < decay_start_epoch:
        return base_weight
    if epoch >= decay_end_epoch:
        return min_weight

    progress = (epoch - decay_start_epoch) / float(decay_end_epoch - decay_start_epoch)
    return base_weight + (min_weight - base_weight) * progress


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


def cam_to_peak_window(cam, image_size, window_ratio=0.35):
    # D1-v2 固定窗口以 CAM peak 为中心，避免多个高响应点的外接框退化成大框。
    image_h, image_w = image_size
    bboxes, areas, masks = [], [], []
    cam = cam.float()
    window_ratio = max(1e-6, min(float(window_ratio), 1.0))
    window_w = max(1, int(round(image_w * window_ratio)))
    window_h = max(1, int(round(image_h * window_ratio)))
    for cam_i in cam:
        cam_h, cam_w = cam_i.shape
        if cam_i.max() <= 0:
            cx = image_w / 2.0
            cy = image_h / 2.0
        else:
            peak_idx = int(cam_i.flatten().argmax().item())
            peak_y, peak_x = divmod(peak_idx, cam_w)
            cx = (peak_x + 0.5) * image_w / float(cam_w)
            cy = (peak_y + 0.5) * image_h / float(cam_h)
        bbox = _fixed_size_bbox_around_center(cx, cy, window_w, window_h, image_h, image_w)
        x1, y1, x2, y2 = bbox
        area = ((x2 - x1) * (y2 - y1)) / float(image_h * image_w)
        bboxes.append(bbox)
        areas.append(area)
        masks.append(_bbox_mask(image_h, image_w, bbox, cam.device))
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


def build_id_candidate_batch(
        model,
        images,
        labels,
        selected_indices,
        candidate_topk=5,
        cam_target='teacher_top1',
        score_type='ori_part_minus_erase',
        include_noisy_label=False,
        cam_quantile=0.8,
        min_area=0.05,
        max_area=0.7,
        bbox_padding=0.05,
        cam_type='weightcam',
        max_top1_prob=1.0):
    selected_indices = selected_indices.detach().long()
    num_selected = int(selected_indices.numel())
    candidate_topk = max(1, int(candidate_topk))
    cam_target = str(cam_target).lower()
    score_type = str(score_type).lower()

    if cam_target != 'teacher_top1':
        raise ValueError(f'id_candidate_cam_target only supports teacher_top1, got {cam_target}.')
    if score_type != 'ori_part_minus_erase':
        raise ValueError(f'id_candidate_score_type only supports ori_part_minus_erase, got {score_type}.')

    # C2 使用单个 teacher-top1 CAM 生成局部图，再用 ori+part-erase 对全类别打分。
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_disabled(images):
            images_float = images.float()
            spatial_features, classifier_features = _extract_spatial_features(model, images_float)
            logits_ori, class_weights = _classifier_weights(model.classifier, classifier_features)
            cam_targets = logits_ori.detach().argmax(dim=1)
            batch_indices = torch.arange(images.size(0), device=images.device)
            target_weights = class_weights[batch_indices, cam_targets]
            cam = _normalize_cam(_build_weight_cam(spatial_features, target_weights))

            bboxes, bbox_areas, erase_masks = cam_to_bbox(
                cam, images.shape[-2:], quantile=cam_quantile,
                min_area=min_area, max_area=max_area, padding=bbox_padding
            )
            x_part = crop_by_bbox(images_float, bboxes)
            x_erase = erase_by_mask(images_float, erase_masks, fill_value=0.0)
            logits_part = _extract_logits(model(x_part))
            logits_erase = _extract_logits(model(x_erase))
    finally:
        if was_training:
            model.train()

    probs_ori = logits_ori.detach().softmax(dim=1)
    probs_part = logits_part.detach().softmax(dim=1)
    probs_erase = logits_erase.detach().softmax(dim=1)
    teacher_top1_prob_all = probs_ori.gather(1, cam_targets.view(-1, 1)).squeeze(1)
    candidate_scores_all = probs_ori + probs_part - probs_erase
    actual_topk = min(candidate_topk, candidate_scores_all.size(1))
    _, topk_indices = candidate_scores_all.topk(actual_topk, dim=1, largest=True, sorted=True)
    candidate_mask_all = _candidate_indices_to_mask(topk_indices, candidate_scores_all.size(1))
    if include_noisy_label:
        # C2-v2 强制保留 noisy label，避免候选监督过早退化成 teacher-top1 自训练。
        noisy_labels = labels.to(device=candidate_mask_all.device, dtype=torch.long).view(-1, 1)
        candidate_mask_all.scatter_(dim=1, index=noisy_labels, value=1.0)
    candidate_indices, candidate_scores, candidate_size = _candidate_mask_to_padded_candidates(
        candidate_mask_all, candidate_scores_all
    )
    candidate_entropy = _candidate_entropy(candidate_scores, candidate_size)

    bbox_areas = torch.tensor(bbox_areas, device=images.device, dtype=images.dtype)
    selected_bbox_areas = bbox_areas[selected_indices] if num_selected > 0 else bbox_areas[:0]
    valid_mask = torch.isfinite(selected_bbox_areas) & (selected_bbox_areas > 0)
    valid_indices = selected_indices[valid_mask]
    candidate_mask = candidate_mask_all[valid_indices]
    teacher_top1_prob = teacher_top1_prob_all[valid_indices]
    # C2-v4 跳过 teacher 原图 top1 过强的 ID 样本，避免候选学习继续强化自训练 top1。
    max_top1_prob = float(max_top1_prob)
    if max_top1_prob < 1.0:
        conf_gate = teacher_top1_prob < max_top1_prob
    else:
        conf_gate = torch.ones_like(teacher_top1_prob, dtype=torch.bool)

    return {
        'labels': labels[valid_indices],
        'selected_indices': selected_indices,
        'valid_mask': valid_mask,
        'batch_indices': valid_indices,
        'bbox_area': bbox_areas[valid_indices],
        'cam_targets': cam_targets[valid_indices],
        'candidate_indices': candidate_indices[valid_indices],
        'candidate_scores': candidate_scores[valid_indices],
        'candidate_size': candidate_size[valid_indices],
        'candidate_mask': candidate_mask,
        'candidate_entropy': candidate_entropy[valid_indices],
        'teacher_top1_prob': teacher_top1_prob,
        'conf_gate': conf_gate,
        'used_in_loss': torch.zeros_like(conf_gate, dtype=torch.bool),
        'num_conf_filtered': int((~conf_gate).sum().item()),
        'candidate_topk': int(actual_topk),
        'num_selected': num_selected,
        'num_valid': int(valid_indices.numel()),
    }


def compute_multi_part_evidence(
        model,
        images,
        labels,
        sample_indices,
        idx_clean,
        idx_id,
        idx_ood,
        epoch,
        batch_idx,
        groups='clean,id',
        num_parts=3,
        use_accum_erase=True,
        top1_source='teacher_top1',
        crop_mode='bbox',
        window_ratio=0.35,
        erase_mode='cam_mask',
        cam_quantile=0.8,
        min_area=0.05,
        max_area=0.7,
        bbox_padding=0.05,
        cam_type='weightcam',
        save_images=False,
        image_dir=None,
        image_max_samples=8,
        image_samples_per_class=1,
        norm_mean=None,
        norm_std=None):
    # D1 只读诊断：CAM 输入使用逐轮擦除图，part crop 始终从原图裁剪，不向训练 loss 传递梯度。
    group_keys = _normalize_multi_part_groups(groups)
    num_parts = max(1, int(num_parts))
    crop_mode = str(crop_mode).lower()
    erase_mode = str(erase_mode).lower()
    window_ratio = float(window_ratio)
    if len(group_keys) == 0:
        return [], []

    idx_clean = idx_clean.detach().long()
    idx_id = idx_id.detach().long()
    idx_ood = idx_ood.detach().long()

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_disabled(images):
            images_float = images.float()
            image_h, image_w = images.shape[-2:]
            labels_long = labels.long()
            logits_ori = _extract_logits(model(images_float))
            top1_targets = logits_ori.detach().argmax(dim=1)
            cam_targets = labels_long.clone()
            cam_target_sources = ['skip'] * images.size(0)
            for index in idx_clean.detach().cpu().tolist():
                cam_target_sources[int(index)] = 'web_label'
            if idx_id.numel() > 0:
                cam_targets[idx_id] = top1_targets[idx_id]
                for index in idx_id.detach().cpu().tolist():
                    # D1 的 ID CAM target 来自调用方传入的诊断模型，显式记录 teacher/student 来源。
                    cam_target_sources[int(index)] = str(top1_source)

            previous_bboxes = []
            part_records = []
            erase_accum_mask = torch.zeros(
                images.size(0), image_h, image_w, device=images.device, dtype=torch.bool
            )
            cam_input = images_float
            for part_idx in range(num_parts):
                cam, _ = generate_cam(model, cam_input, cam_targets, cam_type=cam_type)
                cam_bboxes, cam_bbox_areas, cam_masks = cam_to_bbox(
                    cam, images.shape[-2:], quantile=cam_quantile,
                    min_area=min_area, max_area=max_area, padding=bbox_padding
                )
                peak_bboxes, peak_bbox_areas, peak_masks = cam_to_peak_window(
                    cam, images.shape[-2:], window_ratio=window_ratio
                )
                # D1-v2 支持固定 peak window，避免外接框吞掉整只鸟或多个高响应点。
                if crop_mode == 'bbox':
                    crop_bboxes, crop_bbox_areas = cam_bboxes, cam_bbox_areas
                elif crop_mode == 'peak_window':
                    crop_bboxes, crop_bbox_areas = peak_bboxes, peak_bbox_areas
                else:
                    raise ValueError(f'multi_part_crop_mode only supports bbox or peak_window, got {crop_mode}.')

                if erase_mode == 'cam_mask':
                    erase_masks = cam_masks.bool()
                elif erase_mode == 'bbox':
                    erase_masks = _bboxes_to_masks(image_h, image_w, cam_bboxes, images.device)
                elif erase_mode == 'peak_window':
                    erase_masks = peak_masks.bool()
                else:
                    raise ValueError(
                        f'multi_part_erase_mode only supports cam_mask, bbox, or peak_window, got {erase_mode}.'
                    )
                erase_accum_mask = erase_accum_mask | erase_masks
                x_part = crop_by_bbox(images_float, crop_bboxes)
                logits_part = _extract_logits(model(x_part))
                probs_part = logits_part.detach().softmax(dim=1)
                part_conf, part_pred = probs_part.max(dim=1)
                p_part_target = probs_part.gather(1, cam_targets.view(-1, 1)).squeeze(1)
                p_part_label = probs_part.gather(1, labels_long.view(-1, 1)).squeeze(1)
                bbox_areas_tensor = torch.tensor(crop_bbox_areas, device=images.device, dtype=images.dtype)
                valid_part = torch.isfinite(bbox_areas_tensor) & (bbox_areas_tensor > 0)
                part_iou_prev, part_iou_max = _multi_part_iou_vectors(
                    crop_bboxes, previous_bboxes, images.device, images.dtype
                )
                erase_accum_area = erase_accum_mask.float().mean(dim=(1, 2)).to(dtype=images.dtype)

                part_records.append({
                    'part_id': part_idx + 1,
                    'bboxes': crop_bboxes,
                    'bbox_area': bbox_areas_tensor,
                    'part_pred': part_pred,
                    'part_conf': part_conf,
                    'p_part_target': p_part_target,
                    'p_part_label': p_part_label,
                    'part_iou_prev': part_iou_prev,
                    'part_iou_max': part_iou_max,
                    'erase_accum_area': erase_accum_area,
                    'valid_part': valid_part,
                    'crop_mode': crop_mode,
                    'erase_mode': erase_mode,
                    'window_ratio': window_ratio,
                })
                previous_bboxes.append(crop_bboxes)

                erase_mask_for_next_cam = erase_accum_mask if use_accum_erase else erase_masks
                cam_input = erase_by_mask(images_float, erase_mask_for_next_cam, fill_value=0.0)
            x_erase_accum = erase_by_mask(images_float, erase_accum_mask, fill_value=0.0)
    finally:
        if was_training:
            model.train()

    batch_rows = build_multi_part_log_rows(
        epoch, batch_idx, part_records, idx_clean, idx_id,
        group_keys=group_keys, num_parts=num_parts,
        crop_mode=crop_mode, erase_mode=erase_mode, window_ratio=window_ratio
    )
    sample_rows = build_multi_part_sample_rows(
        epoch, batch_idx, part_records, labels, sample_indices,
        idx_clean, idx_id, group_keys, cam_targets, cam_target_sources
    )
    if save_images:
        save_multi_part_images(
            image_dir, sample_rows, images.float(), x_erase_accum,
            norm_mean=norm_mean, norm_std=norm_std,
            max_samples=image_max_samples,
            samples_per_class=image_samples_per_class,
        )
    return batch_rows, sample_rows


def compute_d2_diag(
        quality_model,
        student_model,
        images,
        labels,
        sample_indices,
        idx_clean,
        c1_gate_mask,
        epoch,
        batch_idx,
        lambdas=(0.1,),
        aggregation_modes=('top1_valid', 'weighted_valid'),
        num_parts=3,
        use_accum_erase=True,
        crop_mode='peak_window',
        window_ratio=0.35,
        erase_mode='peak_window',
        quality_metric='p_target_x_marginal_drop',
        quality_gate_type='per_part_percentile',
        quality_keep_ratio=0.5,
        iou_filter=True,
        iou_thr=0.5,
        cam_quantile=0.8,
        min_area=0.05,
        max_area=0.7,
        bbox_padding=0.05,
        cam_type='weightcam'):
    # D2_diag 只读：teacher/EMA 负责质量评分，student 负责 global/part logits 聚合诊断。
    if str(quality_metric).lower() != 'p_target_x_marginal_drop':
        raise ValueError(f'D2_diag only supports p_target_x_marginal_drop, got {quality_metric}.')
    if str(quality_gate_type).lower() != 'per_part_percentile':
        raise ValueError(f'D2_diag only supports per_part_percentile, got {quality_gate_type}.')

    lambdas = [float(value) for value in lambdas]
    aggregation_modes = [str(value).lower() for value in aggregation_modes]
    use_top1 = 'top1_valid' in aggregation_modes
    use_weighted = 'weighted_valid' in aggregation_modes
    num_parts = max(1, int(num_parts))
    crop_mode = str(crop_mode).lower()
    erase_mode = str(erase_mode).lower()
    window_ratio = float(window_ratio)
    quality_keep_ratio = float(quality_keep_ratio)
    iou_thr = float(iou_thr)

    idx_clean = idx_clean.detach().long()
    c1_gate_mask = c1_gate_mask.detach().to(device=images.device, dtype=torch.bool)
    clean_mask = torch.zeros(images.size(0), device=images.device, dtype=torch.bool)
    clean_mask[idx_clean] = True
    c1_clean_mask = clean_mask & c1_gate_mask
    num_clean = int(idx_clean.numel())
    num_c1_gated = int(c1_clean_mask.sum().item())
    c1_gated_indices = torch.nonzero(c1_clean_mask, as_tuple=False).flatten()
    if num_c1_gated == 0:
        # D2_diag 没有 C1-gated clean 样本时直接写零行，避免整 batch 额外 CAM/ResNet 前向。
        return (
            [
                _empty_d2_diag_row(epoch, batch_idx, lambda_value, num_clean, num_c1_gated)
                for lambda_value in lambdas
            ],
            []
        )

    # D2_diag 的 eligible 子集从 C1-gated clean 开始，后续 CAM/part logits 只在该子 batch 上计算。
    diag_images = images.index_select(0, c1_gated_indices)
    diag_labels = labels.index_select(0, c1_gated_indices)
    if torch.is_tensor(sample_indices):
        diag_sample_indices = sample_indices.detach().cpu().long()[c1_gated_indices.detach().cpu().long()]
    else:
        all_sample_indices = torch.tensor(sample_indices, dtype=torch.long)
        diag_sample_indices = all_sample_indices[c1_gated_indices.detach().cpu().long()]
    diag_c1_clean_mask = torch.ones(num_c1_gated, device=images.device, dtype=torch.bool)

    quality_was_training = quality_model.training
    student_was_training = student_model.training
    quality_model.eval()
    student_model.eval()
    try:
        with torch.no_grad(), _autocast_disabled(diag_images):
            images_float = diag_images.float()
            image_h, image_w = diag_images.shape[-2:]
            labels_long = diag_labels.long()
            previous_bboxes = []
            part_records = []
            student_part_logits = []
            erase_accum_mask = torch.zeros(
                diag_images.size(0), image_h, image_w, device=diag_images.device, dtype=torch.bool
            )
            cam_input = images_float
            for part_idx in range(num_parts):
                cam, logits_before = generate_cam(quality_model, cam_input, labels_long, cam_type=cam_type)
                probs_before = logits_before.detach().softmax(dim=1)
                p_before = probs_before.gather(1, labels_long.view(-1, 1)).squeeze(1)

                cam_bboxes, cam_bbox_areas, cam_masks = cam_to_bbox(
                    cam, diag_images.shape[-2:], quantile=cam_quantile,
                    min_area=min_area, max_area=max_area, padding=bbox_padding
                )
                peak_bboxes, peak_bbox_areas, peak_masks = cam_to_peak_window(
                    cam, diag_images.shape[-2:], window_ratio=window_ratio
                )
                if crop_mode == 'bbox':
                    crop_bboxes, crop_bbox_areas = cam_bboxes, cam_bbox_areas
                elif crop_mode == 'peak_window':
                    crop_bboxes, crop_bbox_areas = peak_bboxes, peak_bbox_areas
                else:
                    raise ValueError(f'multi_part_crop_mode only supports bbox or peak_window, got {crop_mode}.')

                if erase_mode == 'cam_mask':
                    erase_masks = cam_masks.bool()
                elif erase_mode == 'bbox':
                    erase_masks = _bboxes_to_masks(image_h, image_w, cam_bboxes, diag_images.device)
                elif erase_mode == 'peak_window':
                    erase_masks = peak_masks.bool()
                else:
                    raise ValueError(
                        f'multi_part_erase_mode only supports cam_mask, bbox, or peak_window, got {erase_mode}.'
                    )

                next_erase_accum_mask = erase_accum_mask | erase_masks
                x_part = crop_by_bbox(images_float, crop_bboxes)
                quality_logits_part = _extract_logits(quality_model(x_part))
                quality_probs_part = quality_logits_part.detach().softmax(dim=1)
                p_part_target = quality_probs_part.gather(1, labels_long.view(-1, 1)).squeeze(1)
                part_conf, _ = quality_probs_part.max(dim=1)
                cam_after = erase_by_mask(images_float, next_erase_accum_mask, fill_value=0.0)
                logits_after = _extract_logits(quality_model(cam_after))
                p_after = logits_after.detach().softmax(dim=1).gather(1, labels_long.view(-1, 1)).squeeze(1)
                marginal_drop = (p_before - p_after).clamp(min=0)
                quality_score = p_part_target * marginal_drop
                bbox_areas_tensor = torch.tensor(crop_bbox_areas, device=diag_images.device, dtype=diag_images.dtype)
                valid_part = torch.isfinite(bbox_areas_tensor) & (bbox_areas_tensor > 0)
                part_iou_prev, part_iou_max = _multi_part_iou_vectors(
                    crop_bboxes, previous_bboxes, diag_images.device, diag_images.dtype
                )

                part_records.append({
                    'part_id': part_idx + 1,
                    'bboxes': crop_bboxes,
                    'bbox_area': bbox_areas_tensor,
                    'p_part_target': p_part_target,
                    'part_conf': part_conf,
                    'marginal_drop': marginal_drop,
                    'quality_score': quality_score,
                    'part_iou_max': part_iou_max,
                    'valid_part': valid_part,
                })
                student_part_logits.append(_extract_logits(student_model(x_part)).detach())
                previous_bboxes.append(crop_bboxes)
                erase_accum_mask = next_erase_accum_mask
                erase_mask_for_next_cam = erase_accum_mask if use_accum_erase else erase_masks
                cam_input = erase_by_mask(images_float, erase_mask_for_next_cam, fill_value=0.0)

            student_global_logits = _extract_logits(student_model(images_float)).detach()
    finally:
        if quality_was_training:
            quality_model.train()
        if student_was_training:
            student_model.train()

    part_quality = torch.stack([record['quality_score'].detach() for record in part_records], dim=1)
    part_p_target = torch.stack([record['p_part_target'].detach() for record in part_records], dim=1)
    part_marginal_drop = torch.stack([record['marginal_drop'].detach() for record in part_records], dim=1)
    part_iou_max = torch.stack([record['part_iou_max'].detach() for record in part_records], dim=1)
    part_valid_base = torch.stack([record['valid_part'].detach() for record in part_records], dim=1).bool()
    part_logits = torch.stack(student_part_logits, dim=1)

    quality_gate = _build_per_part_percentile_gate(
        part_quality, diag_c1_clean_mask[:, None] & part_valid_base, keep_ratio=quality_keep_ratio
    )
    iou_gate = part_iou_max <= iou_thr if iou_filter else torch.ones_like(part_valid_base, dtype=torch.bool)
    valid_part_mask = part_valid_base & quality_gate & iou_gate & diag_c1_clean_mask[:, None]
    has_valid_part = valid_part_mask.any(dim=1)
    eligible_mask = diag_c1_clean_mask & has_valid_part
    num_eligible = int(eligible_mask.sum().item())

    diag_context = {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'sample_indices': diag_sample_indices,
        'labels': labels_long,
        'num_clean': num_clean,
        'num_c1_gated': num_c1_gated,
        'num_eligible': num_eligible,
        'c1_clean_mask': diag_c1_clean_mask,
        'eligible_mask': eligible_mask,
        'student_global_logits': student_global_logits,
        'part_logits': part_logits,
        'part_quality': part_quality,
        'part_p_target': part_p_target,
        'part_marginal_drop': part_marginal_drop,
        'part_iou_max': part_iou_max,
        'valid_part_mask': valid_part_mask,
        'use_top1': use_top1,
        'use_weighted': use_weighted,
    }
    batch_rows = [
        _build_d2_diag_row(diag_context, lambda_value)
        for lambda_value in lambdas
    ]
    sample_rows = []
    for lambda_value in lambdas:
        sample_rows.extend(_build_d2_diag_sample_rows(diag_context, lambda_value))
    return batch_rows, sample_rows


def compute_id_candidate_loss(
        logits,
        candidate_mask,
        candidate_indices=None,
        candidate_scores=None,
        candidate_size=None,
        labels=None,
        loss_type='pll_entropy',
        entropy_weight=0.0,
        entropy_min_ratio=0.0,
        dist_weight=0.0,
        target_temp=2.0,
        top1_cap=0.5,
        noisy_prior=0.0):
    # C2-v2/v3 统一先计算 PLL 与候选集合内 student 分布，避免 main.py 堆叠 loss 细节。
    loss_type = str(loss_type).lower()
    candidate_mask = candidate_mask.to(device=logits.device, dtype=logits.dtype)
    probs = logits.softmax(dim=1)
    candidate_mass = (probs * candidate_mask).sum(dim=1).clamp(min=1e-12)
    candidate_mass_fp32 = candidate_mass.float().clamp(min=1e-12)
    pll_losses = -torch.log(candidate_mass_fp32)

    candidate_distribution = (probs * candidate_mask) / candidate_mass[:, None]
    candidate_distribution_fp32 = candidate_distribution.float()
    student_candidate_entropy = -(
        candidate_distribution_fp32 * candidate_distribution_fp32.clamp(min=1e-12).log()
    ).sum(dim=1)
    candidate_size_from_mask = candidate_mask.sum(dim=1).float().clamp(min=1.0)
    entropy_floor = float(entropy_min_ratio) * candidate_size_from_mask.log()
    entropy_penalty = torch.relu(entropy_floor - student_candidate_entropy)

    losses = pll_losses
    if loss_type in ['pll_entropy', 'capped_soft']:
        losses = losses + float(entropy_weight) * entropy_penalty

    batch_size = logits.size(0)
    zeros = logits.new_zeros(batch_size).float()
    result = {
        'losses': losses,
        'candidate_mass': candidate_mass,
        'student_candidate_entropy': student_candidate_entropy,
        'entropy_penalty': entropy_penalty,
        'pll_losses': pll_losses,
        'candidate_dist_loss': zeros,
        'candidate_kl': zeros,
        'student_top1_candidate_mass': zeros,
        'student_noisy_label_mass': zeros,
        'target_top1_candidate_mass': zeros,
        'target_noisy_label_mass': zeros,
    }

    if loss_type != 'capped_soft':
        return result

    if candidate_indices is None or candidate_scores is None or labels is None:
        raise ValueError(
            'capped_soft id_candidate loss requires candidate_indices, candidate_scores, and labels.'
        )

    candidate_indices = candidate_indices.to(device=logits.device, dtype=torch.long)
    candidate_scores = candidate_scores.to(device=logits.device, dtype=logits.dtype)
    labels = labels.to(device=logits.device, dtype=torch.long)
    if candidate_size is None:
        candidate_size = candidate_size_from_mask.to(dtype=torch.long)
    else:
        candidate_size = candidate_size.to(device=logits.device, dtype=torch.long)
    valid_positions = (
        torch.arange(candidate_indices.size(1), device=logits.device)[None, :]
        < candidate_size[:, None]
    )

    # C2-v3 构造 capped soft target，显式限制 top1 候选独占目标分布。
    target_distribution = _build_capped_candidate_target(
        candidate_scores, candidate_indices, candidate_size, labels,
        target_temp=target_temp, top1_cap=top1_cap, noisy_prior=noisy_prior
    )
    safe_indices = candidate_indices.clamp(min=0)
    student_candidate_probs = candidate_distribution.gather(1, safe_indices)
    student_candidate_probs = student_candidate_probs * valid_positions.to(dtype=logits.dtype)
    student_candidate_probs_fp32 = student_candidate_probs.float()
    target_distribution_fp32 = target_distribution.float()
    candidate_dist_loss = -(
        target_distribution_fp32 * student_candidate_probs_fp32.clamp(min=1e-12).log()
    ).sum(dim=1)
    candidate_kl = (
        target_distribution_fp32
        * (
            target_distribution_fp32.clamp(min=1e-12).log()
            - student_candidate_probs_fp32.clamp(min=1e-12).log()
        )
    ).sum(dim=1)

    noisy_positions = (candidate_indices == labels[:, None]) & valid_positions
    result['losses'] = losses + float(dist_weight) * candidate_dist_loss
    result['candidate_dist_loss'] = candidate_dist_loss
    result['candidate_kl'] = candidate_kl
    result['student_top1_candidate_mass'] = student_candidate_probs_fp32[:, 0]
    result['student_noisy_label_mass'] = (
        student_candidate_probs_fp32 * noisy_positions.to(dtype=student_candidate_probs_fp32.dtype)
    ).sum(dim=1)
    result['target_top1_candidate_mass'] = target_distribution_fp32[:, 0]
    result['target_noisy_label_mass'] = (
        target_distribution_fp32 * noisy_positions.to(dtype=target_distribution_fp32.dtype)
    ).sum(dim=1)
    return result


def compute_local_margin_loss(logits_part, labels, margin=0.5,
                              hard_negative='max_non_target', use_logits=True):
    hard_negative = str(hard_negative).lower()
    if hard_negative != 'max_non_target':
        raise ValueError(f'local_margin_hard_negative only supports max_non_target, got {hard_negative}.')
    if not use_logits:
        raise ValueError('local_margin first version only supports logits margin.')

    labels = labels.detach().long()
    if logits_part.numel() == 0 or labels.numel() == 0:
        zero_loss = logits_part.sum() * 0.0
        empty = logits_part.new_zeros((0,), dtype=torch.float)
        empty_long = labels.new_zeros((0,), dtype=torch.long)
        return {
            'loss': zero_loss,
            'losses': empty,
            'target_logits': empty,
            'hardneg_logits': empty,
            'part_margin': empty,
            'hardneg_class': empty_long,
            'pred_class': empty_long,
            'violate_mask': empty.to(dtype=torch.bool),
        }

    # D3 使用 logits margin，避免 probability margin 受到 softmax 饱和影响。
    logits = logits_part.float()
    target_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    non_target_logits = logits.clone()
    non_target_logits.scatter_(1, labels.view(-1, 1), torch.finfo(logits.dtype).min)
    hardneg_logits, hardneg_class = non_target_logits.max(dim=1)
    part_margin = target_logits - hardneg_logits
    losses = F.relu(float(margin) - part_margin)
    pred_class = logits.detach().argmax(dim=1)
    return {
        'loss': losses.mean(),
        'losses': losses,
        'target_logits': target_logits.detach(),
        'hardneg_logits': hardneg_logits.detach(),
        'part_margin': part_margin.detach(),
        'hardneg_class': hardneg_class.detach(),
        'pred_class': pred_class.detach(),
        'violate_mask': (part_margin.detach() < float(margin)),
    }


def build_evidence_clean_weights(idx_clean, part_batch, clean_losses,
                                 high_weight=1.0, low_weight=0.5,
                                 batch_size=None):
    device = clean_losses.device
    num_clean = int(clean_losses.numel())
    weights = torch.ones_like(clean_losses)
    valid_clean_mask = torch.zeros(num_clean, device=device, dtype=torch.bool)
    high_clean_mask = torch.zeros(num_clean, device=device, dtype=torch.bool)
    low_clean_mask = torch.zeros(num_clean, device=device, dtype=torch.bool)

    if num_clean == 0:
        return weights, _evidence_clean_context(
            valid_clean_mask, high_clean_mask, low_clean_mask
        )

    gate_mask = part_batch.get('evidence_gate_mask', part_batch.get('gate_mask'))
    if part_batch.get('num_valid', 0) <= 0 or gate_mask is None:
        return weights, _evidence_clean_context(
            valid_clean_mask, high_clean_mask, low_clean_mask
        )

    if batch_size is None:
        batch_size = int(idx_clean.detach().max().item()) + 1 if idx_clean.numel() > 0 else num_clean

    idx_clean = idx_clean.detach().to(device=device, dtype=torch.long)
    batch_indices = part_batch['batch_indices'].detach().to(device=device, dtype=torch.long)
    gate_mask = gate_mask.detach().to(device=device, dtype=torch.bool)
    batch_to_clean_pos = torch.full((int(batch_size),), -1, device=device, dtype=torch.long)
    batch_to_clean_pos[idx_clean] = torch.arange(num_clean, device=device, dtype=torch.long)
    clean_pos = batch_to_clean_pos[batch_indices]
    valid_pos_mask = clean_pos >= 0
    if valid_pos_mask.sum().item() == 0:
        return weights, _evidence_clean_context(
            valid_clean_mask, high_clean_mask, low_clean_mask
        )

    # E1 只在 valid clean part 内按 C1 actual gate 区分 high/low，invalid clean 保持权重 1。
    clean_pos = clean_pos[valid_pos_mask]
    gate_mask = gate_mask[valid_pos_mask]
    valid_clean_mask[clean_pos] = True
    high_clean_mask[clean_pos[gate_mask]] = True
    low_clean_mask[clean_pos[~gate_mask]] = True
    high_values = weights.new_full((clean_pos.numel(),), float(high_weight))
    low_values = weights.new_full((clean_pos.numel(),), float(low_weight))
    weights[clean_pos] = torch.where(gate_mask, high_values, low_values)
    return weights, _evidence_clean_context(
        valid_clean_mask, high_clean_mask, low_clean_mask
    )


def build_evidence_clean_reweight_log_row(
        epoch, batch_idx, part_batch, raw_clean_losses, clean_weights,
        context, high_weight=1.0, low_weight=0.5, num_id=0, num_ood=0):
    raw_clean_losses = raw_clean_losses.detach()
    clean_weights = clean_weights.detach()
    weighted_losses = raw_clean_losses * clean_weights
    num_clean = int(raw_clean_losses.numel())
    if context is None:
        empty_mask = torch.zeros(num_clean, device=raw_clean_losses.device, dtype=torch.bool)
        context = _evidence_clean_context(empty_mask, empty_mask, empty_mask)

    valid_mask = context['valid_clean_mask'].to(device=raw_clean_losses.device, dtype=torch.bool)
    high_mask = context['high_clean_mask'].to(device=raw_clean_losses.device, dtype=torch.bool)
    low_mask = context['low_clean_mask'].to(device=raw_clean_losses.device, dtype=torch.bool)
    invalid_mask = context['invalid_clean_mask'].to(device=raw_clean_losses.device, dtype=torch.bool)
    num_valid = int(valid_mask.sum().item())
    num_high = int(high_mask.sum().item())
    num_low = int(low_mask.sum().item())
    num_invalid = int(invalid_mask.sum().item())
    gate_mask = part_batch.get('evidence_gate_mask', part_batch.get('gate_mask'))

    if part_batch.get('num_valid', 0) > 0 and gate_mask is not None:
        gate_mask = gate_mask.detach().to(device=part_batch['evidence_score'].device, dtype=torch.bool)
        low_part_mask = gate_mask.logical_not()
        evidence_score = part_batch['evidence_score'].detach()
        p_part_y = part_batch['p_part_y'].detach()
        erase_drop = part_batch['erase_drop'].detach()
    else:
        gate_mask = None
        low_part_mask = None
        evidence_score = raw_clean_losses.new_zeros((0,))
        p_part_y = raw_clean_losses.new_zeros((0,))
        erase_drop = raw_clean_losses.new_zeros((0,))

    raw_mean = _mean_or_zero(raw_clean_losses)
    weighted_mean = _mean_or_zero(weighted_losses)
    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'num_clean': num_clean,
        'num_valid': num_valid,
        'num_invalid': num_invalid,
        'num_high': num_high,
        'num_low': num_low,
        'valid_clean_ratio': float(num_valid / max(num_clean, 1)),
        'invalid_ratio': float(num_invalid / max(num_clean, 1)),
        'high_ratio_valid': float(num_high / max(num_valid, 1)),
        'low_ratio_valid': float(num_low / max(num_valid, 1)),
        'high_weight': float(high_weight),
        'low_weight': float(low_weight),
        'clean_weight_mean': _mean_or_zero(clean_weights),
        'raw_clean_loss_mean': raw_mean,
        'weighted_clean_loss_mean': weighted_mean,
        'loss_reduction_ratio': float(weighted_mean / max(abs(raw_mean), 1e-12)),
        'high_raw_clean_loss_mean': _mean_or_zero(raw_clean_losses[high_mask]),
        'low_raw_clean_loss_mean': _mean_or_zero(raw_clean_losses[low_mask]),
        'invalid_raw_clean_loss_mean': _mean_or_zero(raw_clean_losses[invalid_mask]),
        'evidence_score_mean': _mean_or_zero(evidence_score),
        'high_evidence_score_mean': _mean_or_zero(evidence_score[gate_mask]) if gate_mask is not None else 0.0,
        'low_evidence_score_mean': _mean_or_zero(evidence_score[low_part_mask]) if low_part_mask is not None else 0.0,
        'p_part_y_high_mean': _mean_or_zero(p_part_y[gate_mask]) if gate_mask is not None else 0.0,
        'p_part_y_low_mean': _mean_or_zero(p_part_y[low_part_mask]) if low_part_mask is not None else 0.0,
        'erase_drop_high_mean': _mean_or_zero(erase_drop[gate_mask]) if gate_mask is not None else 0.0,
        'erase_drop_low_mean': _mean_or_zero(erase_drop[low_part_mask]) if low_part_mask is not None else 0.0,
        'num_id': int(num_id),
        'num_ood': int(num_ood),
    }


def compute_id_candidate_pll_loss(logits, candidate_mask, entropy_weight=0.0, entropy_min_ratio=0.0):
    result = compute_id_candidate_loss(
        logits, candidate_mask,
        loss_type='pll_entropy',
        entropy_weight=entropy_weight,
        entropy_min_ratio=entropy_min_ratio,
    )
    return (
        result['losses'], result['candidate_mass'],
        result['student_candidate_entropy'], result['entropy_penalty'],
        result['pll_losses']
    )


def attach_id_candidate_loss_results(candidate_batch, result1, result2, used_mask):
    # C2-v4 只把真实参与 loss 的样本写回 loss 指标，被过滤样本保留 0 以便样本级排查。
    metric_keys = [
        'candidate_mass',
        'student_candidate_entropy',
        'entropy_penalty',
        'pll_losses',
        'candidate_dist_loss',
        'candidate_kl',
        'student_top1_candidate_mass',
        'student_noisy_label_mass',
        'target_top1_candidate_mass',
        'target_noisy_label_mass',
    ]
    num_valid = int(candidate_batch['num_valid'])
    for metric_key in metric_keys:
        batch_key = 'pll_loss' if metric_key == 'pll_losses' else metric_key
        if metric_key == 'candidate_mass':
            batch_key = 'student_candidate_mass'
        metric_values = 0.5 * (result1[metric_key] + result2[metric_key])
        full_values = metric_values.new_zeros(num_valid)
        full_values[used_mask.to(device=metric_values.device)] = metric_values
        candidate_batch[batch_key] = full_values
    return candidate_batch


def build_local_margin_log_row(epoch, batch_idx, group, part_batch, total_loss,
                               margin_result, local_margin_weight):
    num_selected = int(part_batch['num_selected'])
    num_valid = int(part_batch['num_valid'])
    if margin_result is None:
        return _empty_local_margin_log_row(epoch, batch_idx, group, num_selected, num_valid)

    part_margin = margin_result['part_margin'].detach()
    num_margin_valid = int(part_margin.numel())
    if num_margin_valid == 0:
        return _empty_local_margin_log_row(epoch, batch_idx, group, num_selected, num_valid)

    raw_loss = float(margin_result['loss'].detach().item())
    weighted_loss = float(local_margin_weight) * raw_loss
    total_loss_value = float(total_loss.detach().item())
    # D3 日志把加权 margin loss 除以当前 total loss，便于判断分支强弱。
    loss_ratio = weighted_loss / max(abs(total_loss_value), 1e-12)
    violate_mask = margin_result['violate_mask'].detach().to(dtype=torch.float)
    hardneg_eq_pred = (
        margin_result['hardneg_class'].detach() == margin_result['pred_class'].detach()
    ).to(dtype=torch.float)

    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'group': group,
        'num_selected': num_selected,
        'num_valid': num_valid,
        'num_margin_valid': num_margin_valid,
        'margin_valid_ratio': float(num_margin_valid / max(num_valid, 1)),
        'raw_local_margin_loss': raw_loss,
        'weighted_local_margin_loss': weighted_loss,
        'weighted_local_margin_loss_ratio': loss_ratio,
        'part_target_logit_mean': _mean_or_zero(margin_result['target_logits']),
        'part_hardneg_logit_mean': _mean_or_zero(margin_result['hardneg_logits']),
        'part_margin_mean': _mean_or_zero(part_margin),
        'part_margin_p25': _quantile_or_zero(part_margin, 0.25),
        'part_margin_p50': _quantile_or_zero(part_margin, 0.50),
        'part_margin_p75': _quantile_or_zero(part_margin, 0.75),
        'part_margin_violate_ratio': _mean_or_zero(violate_mask),
        'hardneg_class_eq_pred_ratio': _mean_or_zero(hardneg_eq_pred),
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


def build_id_candidate_log_row(epoch, batch_idx, candidate_batch, base_loss,
                               id_candidate_loss, id_candidate_weight,
                               effective_id_candidate_weight=None):
    num_selected = candidate_batch['num_selected']
    num_valid = candidate_batch['num_valid']
    base_loss_value = float(base_loss.detach().item())
    id_candidate_loss_value = float(id_candidate_loss.detach().item())
    if effective_id_candidate_weight is None:
        effective_id_candidate_weight = float(id_candidate_weight)
    effective_id_candidate_weight = float(effective_id_candidate_weight)
    weighted_id_candidate_loss = effective_id_candidate_weight * id_candidate_loss_value
    loss_ratio = weighted_id_candidate_loss / max(abs(base_loss_value), 1e-12)

    if num_selected == 0 or num_valid == 0:
        return _empty_id_candidate_log_row(
            epoch, batch_idx, num_selected, num_valid,
            id_candidate_loss_value, id_candidate_weight,
            weighted_id_candidate_loss, loss_ratio,
            candidate_batch.get('candidate_topk', 0),
            effective_id_candidate_weight
        )

    # C2 batch 日志中 loss 相关均值只统计真实参与 C2 loss 的样本，避免高置信过滤样本的 0 占位稀释诊断。
    used_in_loss = candidate_batch.get('used_in_loss')
    used_metric_mask = None if used_in_loss is None else used_in_loss.detach().bool()
    student_candidate_mass = candidate_batch.get('student_candidate_mass')
    student_candidate_mass_mean = _mean_or_zero_or_none(student_candidate_mass, used_metric_mask)
    student_candidate_entropy = candidate_batch.get('student_candidate_entropy')
    student_candidate_entropy_mean = _mean_or_zero_or_none(student_candidate_entropy, used_metric_mask)
    entropy_penalty = candidate_batch.get('entropy_penalty')
    entropy_penalty_mean = _mean_or_zero_or_none(entropy_penalty, used_metric_mask)
    candidate_dist_loss = candidate_batch.get('candidate_dist_loss')
    candidate_kl = candidate_batch.get('candidate_kl')
    student_top1_candidate_mass = candidate_batch.get('student_top1_candidate_mass')
    student_noisy_label_mass = candidate_batch.get('student_noisy_label_mass')
    target_top1_candidate_mass = candidate_batch.get('target_top1_candidate_mass')
    target_noisy_label_mass = candidate_batch.get('target_noisy_label_mass')
    teacher_top1_prob = candidate_batch.get('teacher_top1_prob')
    conf_gate = candidate_batch.get('conf_gate')
    if teacher_top1_prob is None:
        teacher_top1_prob_mean = 0.0
        teacher_top1_prob_gated_mean = 0.0
        teacher_top1_prob_filtered_mean = 0.0
        num_conf_filtered = 0
        conf_filter_ratio = 0.0
    else:
        teacher_top1_prob = teacher_top1_prob.detach()
        if conf_gate is None:
            conf_gate = torch.ones_like(teacher_top1_prob, dtype=torch.bool)
        else:
            conf_gate = conf_gate.detach().to(device=teacher_top1_prob.device, dtype=torch.bool)
        if used_metric_mask is None:
            gated_mask = conf_gate
        else:
            gated_mask = used_metric_mask.to(device=teacher_top1_prob.device, dtype=torch.bool)
        filtered_mask = ~conf_gate
        num_conf_filtered = int(filtered_mask.sum().item())
        conf_filter_ratio = num_conf_filtered / max(int(num_valid), 1)
        teacher_top1_prob_mean = _mean_or_zero(teacher_top1_prob)
        teacher_top1_prob_gated_mean = _mean_or_zero(teacher_top1_prob[gated_mask])
        teacher_top1_prob_filtered_mean = _mean_or_zero(teacher_top1_prob[filtered_mask])

    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'num_selected': int(num_selected),
        'num_valid': int(num_valid),
        'id_candidate_loss': id_candidate_loss_value,
        'id_candidate_weight': float(id_candidate_weight),
        'weighted_id_candidate_loss': weighted_id_candidate_loss,
        'loss_ratio': loss_ratio,
        'candidate_topk': int(candidate_batch['candidate_topk']),
        'candidate_entropy_mean': _mean_or_zero(candidate_batch['candidate_entropy']),
        'student_candidate_mass_mean': student_candidate_mass_mean,
        'bbox_area_mean': _mean_or_zero(candidate_batch['bbox_area']),
        'candidate_size_mean': _mean_or_zero(candidate_batch['candidate_size'].float()),
        'student_candidate_entropy_mean': student_candidate_entropy_mean,
        'entropy_penalty_mean': entropy_penalty_mean,
        'candidate_dist_loss_mean': _mean_or_zero_or_none(candidate_dist_loss, used_metric_mask),
        'candidate_kl_mean': _mean_or_zero_or_none(candidate_kl, used_metric_mask),
        'student_top1_candidate_mass_mean': _mean_or_zero_or_none(student_top1_candidate_mass, used_metric_mask),
        'student_noisy_label_mass_mean': _mean_or_zero_or_none(student_noisy_label_mass, used_metric_mask),
        'target_top1_candidate_mass_mean': _mean_or_zero_or_none(target_top1_candidate_mass, used_metric_mask),
        'target_noisy_label_mass_mean': _mean_or_zero_or_none(target_noisy_label_mass, used_metric_mask),
        'effective_id_candidate_weight': effective_id_candidate_weight,
        'num_conf_filtered': num_conf_filtered,
        'conf_filter_ratio': conf_filter_ratio,
        'teacher_top1_prob_mean': teacher_top1_prob_mean,
        'teacher_top1_prob_gated_mean': teacher_top1_prob_gated_mean,
        'teacher_top1_prob_filtered_mean': teacher_top1_prob_filtered_mean,
    }


def build_id_candidate_sample_rows(epoch, batch_idx, candidate_batch, sample_indices,
                                   student_logits=None):
    if candidate_batch['num_valid'] == 0:
        return []

    # C2 逐样本日志保留候选集合和 noisy label 关系，用于分析是否真正修正闭集错标。
    batch_positions = candidate_batch['batch_indices'].detach().cpu().long()
    if torch.is_tensor(sample_indices):
        sample_ids = sample_indices.detach().cpu().long()[batch_positions]
    else:
        sample_ids = torch.tensor(sample_indices, dtype=torch.long)[batch_positions]

    labels = candidate_batch['labels'].detach().cpu().long()
    cam_targets = candidate_batch['cam_targets'].detach().cpu().long()
    candidate_indices = candidate_batch['candidate_indices'].detach().cpu().long()
    candidate_scores = candidate_batch['candidate_scores'].detach().cpu()
    candidate_size = candidate_batch['candidate_size'].detach().cpu().long()
    candidate_entropy = candidate_batch['candidate_entropy'].detach().cpu()
    bbox_area = candidate_batch['bbox_area'].detach().cpu()
    pll_loss = candidate_batch.get('pll_loss')
    if pll_loss is None:
        pll_loss = torch.zeros(candidate_batch['num_valid'], dtype=torch.float)
    else:
        pll_loss = pll_loss.detach().cpu()
    student_candidate_entropy = candidate_batch.get('student_candidate_entropy')
    if student_candidate_entropy is None:
        student_candidate_entropy = torch.zeros(candidate_batch['num_valid'], dtype=torch.float)
    else:
        student_candidate_entropy = student_candidate_entropy.detach().cpu()
    entropy_penalty = candidate_batch.get('entropy_penalty')
    if entropy_penalty is None:
        entropy_penalty = torch.zeros(candidate_batch['num_valid'], dtype=torch.float)
    else:
        entropy_penalty = entropy_penalty.detach().cpu()
    candidate_dist_loss = _candidate_batch_vector_or_zeros(candidate_batch, 'candidate_dist_loss')
    candidate_kl = _candidate_batch_vector_or_zeros(candidate_batch, 'candidate_kl')
    student_top1_candidate_mass = _candidate_batch_vector_or_zeros(candidate_batch, 'student_top1_candidate_mass')
    student_noisy_label_mass = _candidate_batch_vector_or_zeros(candidate_batch, 'student_noisy_label_mass')
    target_top1_candidate_mass = _candidate_batch_vector_or_zeros(candidate_batch, 'target_top1_candidate_mass')
    target_noisy_label_mass = _candidate_batch_vector_or_zeros(candidate_batch, 'target_noisy_label_mass')
    teacher_top1_prob = _candidate_batch_vector_or_zeros(candidate_batch, 'teacher_top1_prob')
    conf_gate = candidate_batch.get('conf_gate')
    if conf_gate is None:
        conf_gate = torch.ones(candidate_batch['num_valid'], dtype=torch.bool)
    else:
        conf_gate = conf_gate.detach().cpu().bool()
    used_in_loss = candidate_batch.get('used_in_loss')
    if used_in_loss is None:
        used_in_loss = torch.zeros(candidate_batch['num_valid'], dtype=torch.bool)
    else:
        used_in_loss = used_in_loss.detach().cpu().bool()
    effective_id_candidate_weight = float(candidate_batch.get('effective_id_candidate_weight', 0.0))

    pred_top1 = torch.full((candidate_batch['num_valid'],), -1, dtype=torch.long)
    pred_conf = torch.zeros(candidate_batch['num_valid'], dtype=torch.float)
    if student_logits is not None:
        student_probs = student_logits.detach().softmax(dim=1).cpu()
        pred_conf, pred_top1 = student_probs[batch_positions].max(dim=1)

    rows = []
    for i in range(candidate_batch['num_valid']):
        row_candidate_size = int(candidate_size[i].item())
        candidate_set = candidate_indices[i, :row_candidate_size].tolist()
        candidate_score_values = candidate_scores[i, :row_candidate_size].tolist()
        top1_candidate = int(candidate_set[0]) if row_candidate_size > 0 else -1
        noisy_label = int(labels[i].item())
        if bool(used_in_loss[i].item()):
            skip_reason = 'used'
        elif not bool(conf_gate[i].item()):
            skip_reason = 'high_conf'
        elif effective_id_candidate_weight <= 0.0:
            skip_reason = 'zero_weight'
        else:
            skip_reason = 'skipped'
        rows.append({
            'epoch': int(epoch),
            'batch_idx': int(batch_idx),
            'sample_id': int(sample_ids[i].item()),
            'noisy_label': noisy_label,
            'cam_target': int(cam_targets[i].item()),
            'candidate_set': '|'.join(str(int(v)) for v in candidate_set),
            'top1_candidate': top1_candidate,
            'candidate_scores': '|'.join(f'{float(v):.6f}' for v in candidate_score_values),
            'candidate_entropy': float(candidate_entropy[i].item()),
            'pll_loss': float(pll_loss[i].item()),
            'bbox_area': float(bbox_area[i].item()),
            'pred_top1': int(pred_top1[i].item()),
            'pred_conf': float(pred_conf[i].item()),
            'noisy_label_in_candidate': int(noisy_label in candidate_set),
            'top1_candidate_eq_noisy_label': int(top1_candidate == noisy_label),
            'candidate_size': row_candidate_size,
            'student_candidate_entropy': float(student_candidate_entropy[i].item()),
            'entropy_penalty': float(entropy_penalty[i].item()),
            'candidate_dist_loss': float(candidate_dist_loss[i].item()),
            'candidate_kl': float(candidate_kl[i].item()),
            'student_top1_candidate_mass': float(student_top1_candidate_mass[i].item()),
            'student_noisy_label_mass': float(student_noisy_label_mass[i].item()),
            'target_top1_candidate_mass': float(target_top1_candidate_mass[i].item()),
            'target_noisy_label_mass': float(target_noisy_label_mass[i].item()),
            'teacher_top1_prob': float(teacher_top1_prob[i].item()),
            'conf_gate': int(conf_gate[i].item()),
            'effective_id_candidate_weight': effective_id_candidate_weight,
            'used_in_id_candidate_loss': int(used_in_loss[i].item()),
            'skip_reason': skip_reason,
        })
    return rows


def build_multi_part_log_rows(epoch, batch_idx, part_records, idx_clean, idx_id,
                              group_keys=None, num_parts=3,
                              crop_mode='bbox', erase_mode='cam_mask', window_ratio=0.35):
    # D1 batch 日志按 group+part_id 展开，便于直接观察 part2/part3 是否退化或高度重叠。
    if group_keys is None:
        group_keys = ['clean', 'id']
    group_specs = []
    if 'clean' in group_keys:
        group_specs.append(('clean', idx_clean.detach().long()))
    if 'id' in group_keys:
        group_specs.append(('ID', idx_id.detach().long()))

    rows = []
    for group_name, group_indices in group_specs:
        num_selected = int(group_indices.numel())
        for record in part_records:
            if num_selected == 0:
                rows.append(_empty_multi_part_log_row(
                    epoch, batch_idx, group_name, record['part_id'], num_selected, int(num_parts),
                    crop_mode, erase_mode, window_ratio
                ))
                continue

            valid_part = record['valid_part'][group_indices].detach().bool()
            selected_indices = group_indices[valid_part]
            num_valid = int(selected_indices.numel())
            if num_valid == 0:
                rows.append(_empty_multi_part_log_row(
                    epoch, batch_idx, group_name, record['part_id'], num_selected, int(num_parts),
                    crop_mode, erase_mode, window_ratio
                ))
                continue

            rows.append({
                'epoch': int(epoch),
                'batch_idx': int(batch_idx),
                'group': group_name,
                'part_id': int(record['part_id']),
                'num_selected': num_selected,
                'num_valid': num_valid,
                'valid_part_ratio': float(num_valid / max(num_selected, 1)),
                'num_parts': int(num_parts),
                'bbox_area_mean': _mean_or_zero(record['bbox_area'][selected_indices]),
                'part_iou_max_mean': _mean_or_zero(record['part_iou_max'][selected_indices]),
                'part_iou_prev_mean': _mean_or_zero(record['part_iou_prev'][selected_indices]),
                'p_part_target_mean': _mean_or_zero(record['p_part_target'][selected_indices]),
                'p_part_label_mean': _mean_or_zero(record['p_part_label'][selected_indices]),
                'part_conf_mean': _mean_or_zero(record['part_conf'][selected_indices]),
                'erase_accum_area_mean': _mean_or_zero(record['erase_accum_area'][selected_indices]),
                'crop_mode': crop_mode,
                'erase_mode': erase_mode,
                'window_ratio': float(window_ratio),
            })
    return rows


def build_multi_part_sample_rows(epoch, batch_idx, part_records, labels, sample_indices,
                                 idx_clean, idx_id, group_keys, cam_targets, cam_target_sources):
    # D1 逐样本日志保留每个 part 的 bbox、置信度和与历史 part 的 IoU，支撑互补性分析。
    if len(part_records) == 0:
        return []

    selected_specs = []
    if 'clean' in group_keys:
        selected_specs.extend(('clean', int(index)) for index in idx_clean.detach().cpu().tolist())
    if 'id' in group_keys:
        selected_specs.extend(('ID', int(index)) for index in idx_id.detach().cpu().tolist())
    if len(selected_specs) == 0:
        return []

    if torch.is_tensor(sample_indices):
        sample_ids = sample_indices.detach().cpu().long()
    else:
        sample_ids = torch.tensor(sample_indices, dtype=torch.long)
    labels_cpu = labels.detach().cpu().long()
    cam_targets_cpu = cam_targets.detach().cpu().long()

    part_conf_stack = torch.stack([record['part_conf'].detach() for record in part_records], dim=0)
    part_rank_conf = _rank_multi_part_confidence(part_conf_stack).detach().cpu().long()

    rows = []
    for group_name, batch_pos in selected_specs:
        label = int(labels_cpu[batch_pos].item())
        sample_id = int(sample_ids[batch_pos].item())
        cam_target = int(cam_targets_cpu[batch_pos].item())
        cam_target_source = cam_target_sources[batch_pos]
        for part_idx, record in enumerate(part_records):
            bbox = record['bboxes'][batch_pos]
            x1, y1, x2, y2 = bbox
            rows.append({
                'epoch': int(epoch),
                'batch_idx': int(batch_idx),
                'sample_id': sample_id,
                'batch_pos': int(batch_pos),
                'group': group_name,
                'label': label,
                'part_id': int(record['part_id']),
                'cam_target': cam_target,
                'cam_target_source': cam_target_source,
                'bbox_x1': int(x1),
                'bbox_y1': int(y1),
                'bbox_x2': int(x2),
                'bbox_y2': int(y2),
                'bbox_area': float(record['bbox_area'][batch_pos].detach().item()),
                'part_pred': int(record['part_pred'][batch_pos].detach().item()),
                'part_conf': float(record['part_conf'][batch_pos].detach().item()),
                'p_part_target': float(record['p_part_target'][batch_pos].detach().item()),
                'p_part_label': float(record['p_part_label'][batch_pos].detach().item()),
                'part_iou_prev': float(record['part_iou_prev'][batch_pos].detach().item()),
                'part_iou_max': float(record['part_iou_max'][batch_pos].detach().item()),
                'erase_accum_area': float(record['erase_accum_area'][batch_pos].detach().item()),
                'part_rank_conf': int(part_rank_conf[part_idx, batch_pos].item()),
                'valid_part': int(record['valid_part'][batch_pos].detach().item()),
                'erase_round': int(record['part_id']),
                'crop_mode': record['crop_mode'],
                'erase_mode': record['erase_mode'],
                'window_ratio': float(record['window_ratio']),
            })
    return rows


def save_multi_part_images(output_dir, sample_rows, images, x_erase_accum,
                           norm_mean, norm_std, max_samples=8, samples_per_class=1):
    # D1 可视化按类别抽样保存，图像网格展示原图、每个 part 的 bbox 和最终累计擦除图。
    if output_dir is None or max_samples <= 0 or len(sample_rows) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)

    rows_by_sample = {}
    for row in sample_rows:
        rows_by_sample.setdefault(row['sample_id'], []).append(row)

    selected_items = _select_multi_part_image_samples(rows_by_sample, max_samples, samples_per_class)
    for sample_id, rows in selected_items:
        rows = sorted(rows, key=lambda item: item['part_id'])
        batch_pos = int(rows[0]['batch_pos'])
        original = _tensor_to_pil(_denormalize_image(images[batch_pos], norm_mean, norm_std))
        part_bbox_images = []
        for row in rows:
            part_view = original.copy()
            bbox = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])
            _draw_bbox(part_view, bbox)
            part_bbox_images.append(part_view)
        erase_view = _tensor_to_pil(_denormalize_image(x_erase_accum[batch_pos], norm_mean, norm_std))
        grid = _concat_images([original] + part_bbox_images + [erase_view])
        filename = (
            f"epoch_{rows[0]['epoch']:03d}_batch_{rows[0]['batch_idx']:05d}_"
            f"sample_{sample_id}_label_{rows[0]['label']}_group_{rows[0]['group']}_parts.png"
        )
        grid.save(os.path.join(output_dir, filename))


def _empty_d2_diag_row(epoch, batch_idx, lambda_value, num_clean, num_c1_gated):
    # D2_diag 空子集日志保留 batch/lambda/count 字段，其余指标置零，便于 pandas 直接读取。
    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'lambda': float(lambda_value),
        'num_clean': int(num_clean),
        'num_c1_gated': int(num_c1_gated),
        'num_part_valid_samples': 0,
        'valid_sample_ratio': 0.0,
        'valid_part_per_sample': 0.0,
        'part1_valid_ratio': 0.0,
        'part2_valid_ratio': 0.0,
        'part3_valid_ratio': 0.0,
        'global_acc_all_c1_gated': 0.0,
        'global_acc_valid_subset': 0.0,
        'top1_final_acc': 0.0,
        'weighted_final_acc': 0.0,
        'top1_acc_delta': 0.0,
        'weighted_acc_delta': 0.0,
        'changed_correct_top1': 0,
        'changed_wrong_top1': 0,
        'changed_correct_weighted': 0,
        'changed_wrong_weighted': 0,
        'valid_p_part_target_mean': 0.0,
        'filtered_p_part_target_mean': 0.0,
        'valid_quality_score_mean': 0.0,
        'filtered_quality_score_mean': 0.0,
        'valid_marginal_drop_mean': 0.0,
        'filtered_marginal_drop_mean': 0.0,
        'global_logit_norm_mean': 0.0,
        'part_logit_norm_mean': 0.0,
    }


def _build_d2_diag_row(context, lambda_value):
    # D2_diag batch 指标只在 eligible 子集上比较 global 与聚合 logits，避免不同样本集合造成误判。
    labels = context['labels']
    c1_clean_mask = context['c1_clean_mask']
    eligible_mask = context['eligible_mask']
    valid_part_mask = context['valid_part_mask']
    candidate_part_mask = c1_clean_mask[:, None]
    part_quality = context['part_quality']
    part_p_target = context['part_p_target']
    part_marginal_drop = context['part_marginal_drop']
    global_logits = context['student_global_logits']
    part_logits = context['part_logits']
    lambda_value = float(lambda_value)

    global_probs = global_logits.softmax(dim=1)
    global_conf, global_pred = global_probs.max(dim=1)
    global_correct = global_pred.eq(labels)
    global_acc_all_c1_gated = _masked_mean_bool(global_correct, c1_clean_mask)
    global_acc_valid_subset = _masked_mean_bool(global_correct, eligible_mask)

    final = _d2_diag_final_predictions(context, lambda_value)
    top1_correct = final['top1_pred'].eq(labels)
    weighted_correct = final['weighted_pred'].eq(labels)
    top1_final_acc = _masked_mean_bool(top1_correct, eligible_mask) if context['use_top1'] else 0.0
    weighted_final_acc = _masked_mean_bool(weighted_correct, eligible_mask) if context['use_weighted'] else 0.0

    valid_part_counts = valid_part_mask.sum(dim=1).float()
    num_c1_gated = context['num_c1_gated']
    valid_values_mask = valid_part_mask
    filtered_values_mask = candidate_part_mask & valid_part_mask.logical_not()
    part_logit_norm = part_logits.detach().float().norm(dim=2)

    return {
        'epoch': context['epoch'],
        'batch_idx': context['batch_idx'],
        'lambda': lambda_value,
        'num_clean': context['num_clean'],
        'num_c1_gated': num_c1_gated,
        'num_part_valid_samples': context['num_eligible'],
        'valid_sample_ratio': context['num_eligible'] / max(num_c1_gated, 1),
        'valid_part_per_sample': _mean_or_zero(valid_part_counts[eligible_mask]),
        'part1_valid_ratio': _part_valid_ratio(valid_part_mask, 0, num_c1_gated),
        'part2_valid_ratio': _part_valid_ratio(valid_part_mask, 1, num_c1_gated),
        'part3_valid_ratio': _part_valid_ratio(valid_part_mask, 2, num_c1_gated),
        'global_acc_all_c1_gated': global_acc_all_c1_gated,
        'global_acc_valid_subset': global_acc_valid_subset,
        'top1_final_acc': top1_final_acc,
        'weighted_final_acc': weighted_final_acc,
        'top1_acc_delta': top1_final_acc - global_acc_valid_subset if context['use_top1'] else 0.0,
        'weighted_acc_delta': weighted_final_acc - global_acc_valid_subset if context['use_weighted'] else 0.0,
        'changed_correct_top1': _changed_correct_count(global_correct, top1_correct, eligible_mask) if context['use_top1'] else 0,
        'changed_wrong_top1': _changed_wrong_count(global_correct, top1_correct, eligible_mask) if context['use_top1'] else 0,
        'changed_correct_weighted': _changed_correct_count(global_correct, weighted_correct, eligible_mask) if context['use_weighted'] else 0,
        'changed_wrong_weighted': _changed_wrong_count(global_correct, weighted_correct, eligible_mask) if context['use_weighted'] else 0,
        'valid_p_part_target_mean': _mean_or_zero(part_p_target[valid_values_mask]),
        'filtered_p_part_target_mean': _mean_or_zero(part_p_target[filtered_values_mask]),
        'valid_quality_score_mean': _mean_or_zero(part_quality[valid_values_mask]),
        'filtered_quality_score_mean': _mean_or_zero(part_quality[filtered_values_mask]),
        'valid_marginal_drop_mean': _mean_or_zero(part_marginal_drop[valid_values_mask]),
        'filtered_marginal_drop_mean': _mean_or_zero(part_marginal_drop[filtered_values_mask]),
        'global_logit_norm_mean': _mean_or_zero(global_logits.detach().float().norm(dim=1)[eligible_mask]),
        'part_logit_norm_mean': _mean_or_zero(part_logit_norm[valid_part_mask]),
    }


def _build_d2_diag_sample_rows(context, lambda_value):
    # D2_diag 样本日志只写 eligible 子集，便于直接复算 changed_correct / changed_wrong。
    eligible_indices = torch.nonzero(context['eligible_mask'], as_tuple=False).flatten()
    if eligible_indices.numel() == 0:
        return []

    labels = context['labels']
    sample_indices = context['sample_indices']
    if torch.is_tensor(sample_indices):
        sample_ids = sample_indices.detach().cpu().long()
    else:
        sample_ids = torch.tensor(sample_indices, dtype=torch.long)

    global_logits = context['student_global_logits']
    global_probs = global_logits.softmax(dim=1)
    global_conf, global_pred = global_probs.max(dim=1)
    global_correct = global_pred.eq(labels)
    final = _d2_diag_final_predictions(context, lambda_value)
    top1_probs = final['top1_logits'].softmax(dim=1)
    top1_conf, top1_pred = top1_probs.max(dim=1)
    weighted_probs = final['weighted_logits'].softmax(dim=1)
    weighted_conf, weighted_pred = weighted_probs.max(dim=1)
    top1_correct = top1_pred.eq(labels)
    weighted_correct = weighted_pred.eq(labels)

    rows = []
    part_quality = context['part_quality'].detach().cpu()
    part_valid_mask = context['valid_part_mask'].detach().cpu()
    part_p_target = context['part_p_target'].detach().cpu()
    part_marginal_drop = context['part_marginal_drop'].detach().cpu()
    part_iou_max = context['part_iou_max'].detach().cpu()
    best_part_idx = final['best_part_idx'].detach().cpu().long()
    best_part_quality = final['best_part_quality'].detach().cpu()
    for batch_pos in eligible_indices.detach().cpu().long().tolist():
        valid_parts = [
            str(part_idx + 1)
            for part_idx, is_valid in enumerate(part_valid_mask[batch_pos].tolist())
            if bool(is_valid)
        ]
        best_part = int(best_part_idx[batch_pos].item()) + 1
        rows.append({
            'epoch': context['epoch'],
            'batch_idx': context['batch_idx'],
            'lambda': float(lambda_value),
            'sample_id': int(sample_ids[batch_pos].item()),
            'label': int(labels.detach().cpu()[batch_pos].item()),
            'global_pred': int(global_pred.detach().cpu()[batch_pos].item()),
            'global_conf': float(global_conf.detach().cpu()[batch_pos].item()),
            'global_correct_valid_subset': int(global_correct.detach().cpu()[batch_pos].item()),
            'selected_parts_top1': str(best_part),
            'selected_parts_weighted': '|'.join(valid_parts),
            'part_quality_scores': _pipe_float_values(part_quality[batch_pos]),
            'part_valid_mask': _pipe_int_values(part_valid_mask[batch_pos].long()),
            'part_p_targets': _pipe_float_values(part_p_target[batch_pos]),
            'part_marginal_drops': _pipe_float_values(part_marginal_drop[batch_pos]),
            'part_iou_max': _pipe_float_values(part_iou_max[batch_pos]),
            'best_part_idx': best_part,
            'best_part_quality_score': float(best_part_quality[batch_pos].item()),
            'top1_final_pred': int(top1_pred.detach().cpu()[batch_pos].item()) if context['use_top1'] else -1,
            'top1_final_conf': float(top1_conf.detach().cpu()[batch_pos].item()) if context['use_top1'] else 0.0,
            'top1_final_correct': int(top1_correct.detach().cpu()[batch_pos].item()) if context['use_top1'] else 0,
            'weighted_final_pred': int(weighted_pred.detach().cpu()[batch_pos].item()) if context['use_weighted'] else -1,
            'weighted_final_conf': float(weighted_conf.detach().cpu()[batch_pos].item()) if context['use_weighted'] else 0.0,
            'weighted_final_correct': int(weighted_correct.detach().cpu()[batch_pos].item()) if context['use_weighted'] else 0,
            'top1_changed': int(top1_pred.detach().cpu()[batch_pos].item() != global_pred.detach().cpu()[batch_pos].item()) if context['use_top1'] else 0,
            'weighted_changed': int(weighted_pred.detach().cpu()[batch_pos].item() != global_pred.detach().cpu()[batch_pos].item()) if context['use_weighted'] else 0,
        })
    return rows


def _d2_diag_final_predictions(context, lambda_value):
    valid_part_mask = context['valid_part_mask']
    part_quality = context['part_quality'].clamp(min=0)
    part_logits = context['part_logits']
    global_logits = context['student_global_logits']
    masked_quality = part_quality.masked_fill(~valid_part_mask, torch.finfo(part_quality.dtype).min)
    best_part_idx = masked_quality.argmax(dim=1)
    best_part_quality = part_quality.gather(1, best_part_idx.view(-1, 1)).squeeze(1)
    batch_indices = torch.arange(global_logits.size(0), device=global_logits.device)
    best_part_logits = part_logits[batch_indices, best_part_idx]
    top1_logits = global_logits + float(lambda_value) * best_part_logits

    weights = part_quality * valid_part_mask.to(dtype=part_quality.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    weighted_part_logits = (part_logits * weights[:, :, None]).sum(dim=1)
    weighted_logits = global_logits + float(lambda_value) * weighted_part_logits
    return {
        'best_part_idx': best_part_idx,
        'best_part_quality': best_part_quality,
        'top1_logits': top1_logits,
        'top1_pred': top1_logits.argmax(dim=1),
        'weighted_logits': weighted_logits,
        'weighted_pred': weighted_logits.argmax(dim=1),
    }


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


def _empty_local_margin_log_row(epoch, batch_idx, group, num_selected, num_valid):
    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'group': group,
        'num_selected': int(num_selected),
        'num_valid': int(num_valid),
        'num_margin_valid': 0,
        'margin_valid_ratio': 0.0,
        'raw_local_margin_loss': 0.0,
        'weighted_local_margin_loss': 0.0,
        'weighted_local_margin_loss_ratio': 0.0,
        'part_target_logit_mean': 0.0,
        'part_hardneg_logit_mean': 0.0,
        'part_margin_mean': 0.0,
        'part_margin_p25': 0.0,
        'part_margin_p50': 0.0,
        'part_margin_p75': 0.0,
        'part_margin_violate_ratio': 0.0,
        'hardneg_class_eq_pred_ratio': 0.0,
    }


def _evidence_clean_context(valid_clean_mask, high_clean_mask, low_clean_mask):
    invalid_clean_mask = valid_clean_mask.logical_not()
    return {
        'valid_clean_mask': valid_clean_mask,
        'high_clean_mask': high_clean_mask,
        'low_clean_mask': low_clean_mask,
        'invalid_clean_mask': invalid_clean_mask,
    }


def _empty_id_candidate_log_row(epoch, batch_idx, num_selected, num_valid,
                                id_candidate_loss, id_candidate_weight,
                                weighted_id_candidate_loss, loss_ratio,
                                candidate_topk, effective_id_candidate_weight=None):
    if effective_id_candidate_weight is None:
        effective_id_candidate_weight = id_candidate_weight
    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'num_selected': int(num_selected),
        'num_valid': int(num_valid),
        'id_candidate_loss': float(id_candidate_loss),
        'id_candidate_weight': float(id_candidate_weight),
        'weighted_id_candidate_loss': float(weighted_id_candidate_loss),
        'loss_ratio': float(loss_ratio),
        'candidate_topk': int(candidate_topk),
        'candidate_entropy_mean': 0.0,
        'student_candidate_mass_mean': 0.0,
        'bbox_area_mean': 0.0,
        'candidate_size_mean': 0.0,
        'student_candidate_entropy_mean': 0.0,
        'entropy_penalty_mean': 0.0,
        'candidate_dist_loss_mean': 0.0,
        'candidate_kl_mean': 0.0,
        'student_top1_candidate_mass_mean': 0.0,
        'student_noisy_label_mass_mean': 0.0,
        'target_top1_candidate_mass_mean': 0.0,
        'target_noisy_label_mass_mean': 0.0,
        'effective_id_candidate_weight': float(effective_id_candidate_weight),
        'num_conf_filtered': 0,
        'conf_filter_ratio': 0.0,
        'teacher_top1_prob_mean': 0.0,
        'teacher_top1_prob_gated_mean': 0.0,
        'teacher_top1_prob_filtered_mean': 0.0,
    }


def _empty_multi_part_log_row(epoch, batch_idx, group, part_id, num_selected, num_parts,
                              crop_mode='bbox', erase_mode='cam_mask', window_ratio=0.35):
    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'group': group,
        'part_id': int(part_id),
        'num_selected': int(num_selected),
        'num_valid': 0,
        'valid_part_ratio': 0.0,
        'num_parts': int(num_parts),
        'bbox_area_mean': 0.0,
        'part_iou_max_mean': 0.0,
        'part_iou_prev_mean': 0.0,
        'p_part_target_mean': 0.0,
        'p_part_label_mean': 0.0,
        'part_conf_mean': 0.0,
        'erase_accum_area_mean': 0.0,
        'crop_mode': crop_mode,
        'erase_mode': erase_mode,
        'window_ratio': float(window_ratio),
    }


def _mean_or_zero(values):
    if values.numel() == 0:
        return 0.0
    return float(values.detach().mean().item())


def _quantile_or_zero(values, quantile):
    values = values.detach().float()
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, float(quantile)).item())


def _mean_or_zero_or_none(values, mask=None):
    if values is None:
        return 0.0
    values = values.detach()
    if mask is not None and values.numel() == mask.numel():
        values = values[mask.to(device=values.device, dtype=torch.bool)]
    return _mean_or_zero(values)


def _candidate_batch_vector_or_zeros(candidate_batch, key):
    value = candidate_batch.get(key)
    if value is None:
        return torch.zeros(candidate_batch['num_valid'], dtype=torch.float)
    return value.detach().cpu()


def _float_value(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _build_per_part_percentile_gate(scores, base_mask, keep_ratio=0.5):
    # D2_diag 每个 part 独立做 percentile top-k，避免 part1 分数天然压制 part2/part3。
    scores = scores.detach()
    base_mask = base_mask.to(device=scores.device, dtype=torch.bool)
    gate_mask = torch.zeros_like(base_mask, dtype=torch.bool)
    keep_ratio = max(0.0, min(float(keep_ratio), 1.0))
    if scores.numel() == 0 or keep_ratio <= 0.0:
        return gate_mask

    for part_idx in range(scores.size(1)):
        candidate_indices = torch.nonzero(base_mask[:, part_idx], as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        k = max(1, int(candidate_indices.numel() * keep_ratio))
        k = min(k, candidate_indices.numel())
        part_scores = scores[candidate_indices, part_idx]
        _, topk_indices = torch.topk(part_scores, k, largest=True, sorted=False)
        gate_mask[candidate_indices[topk_indices], part_idx] = True
    return gate_mask


def _masked_mean_bool(values, mask):
    # D2_diag accuracy 只在指定子集上统计；空子集返回 0，避免日志出现 NaN。
    values = values.detach().to(dtype=torch.float)
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.sum().item() == 0:
        return 0.0
    return float(values[mask].mean().item())


def _part_valid_ratio(valid_part_mask, part_idx, denominator):
    if part_idx >= valid_part_mask.size(1) or denominator <= 0:
        return 0.0
    return float(valid_part_mask[:, part_idx].sum().item() / max(int(denominator), 1))


def _changed_correct_count(global_correct, final_correct, eligible_mask):
    # 聚合后从错变对的样本数，用于判断 part logits 是否提供正收益。
    mask = eligible_mask.to(device=global_correct.device, dtype=torch.bool)
    changed = mask & global_correct.logical_not() & final_correct
    return int(changed.sum().item())


def _changed_wrong_count(global_correct, final_correct, eligible_mask):
    # 聚合后从对变错的样本数，用于约束 D2_safe 是否可能扰动主干预测。
    mask = eligible_mask.to(device=global_correct.device, dtype=torch.bool)
    changed = mask & global_correct & final_correct.logical_not()
    return int(changed.sum().item())


def _pipe_float_values(values):
    values = values.detach().flatten().cpu().tolist()
    return '|'.join(f'{float(value):.6f}' for value in values)


def _pipe_int_values(values):
    values = values.detach().flatten().cpu().tolist()
    return '|'.join(str(int(value)) for value in values)


def _normalize_multi_part_groups(groups):
    # D1 仅诊断 clean/ID，OOD 默认跳过，避免无目标集合的局部区域污染日志解释。
    if isinstance(groups, str):
        group_items = [item.strip().lower() for item in groups.split(',') if item.strip()]
    else:
        group_items = [str(item).strip().lower() for item in groups if str(item).strip()]
    normalized = []
    for item in group_items:
        if item == 'id':
            key = 'id'
        elif item == 'clean':
            key = 'clean'
        else:
            raise ValueError(f'multi_part_groups only supports clean,id, got {item}.')
        if key not in normalized:
            normalized.append(key)
    return normalized


def _multi_part_iou_vectors(bboxes, previous_bboxes, device, dtype):
    if len(previous_bboxes) == 0:
        zeros = torch.zeros(len(bboxes), device=device, dtype=dtype)
        return zeros, zeros

    prev_values = []
    max_values = []
    for sample_idx, bbox in enumerate(bboxes):
        prev_iou = _bbox_iou(bbox, previous_bboxes[-1][sample_idx])
        max_iou = max(_bbox_iou(bbox, prior_bboxes[sample_idx]) for prior_bboxes in previous_bboxes)
        prev_values.append(prev_iou)
        max_values.append(max_iou)
    return (
        torch.tensor(prev_values, device=device, dtype=dtype),
        torch.tensor(max_values, device=device, dtype=dtype),
    )


def _bbox_iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _rank_multi_part_confidence(part_conf_stack):
    # part_rank_conf=1 表示该样本所有 part 中置信度最高的局部图。
    order = part_conf_stack.argsort(dim=0, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(
        1, part_conf_stack.size(0) + 1,
        device=part_conf_stack.device, dtype=order.dtype
    ).view(-1, 1).expand_as(order)
    ranks.scatter_(0, order, rank_values)
    return ranks


def _select_multi_part_image_samples(rows_by_sample, max_samples, samples_per_class):
    # D1 图片按 label 限制每类保存数量，同时保留总量上限，避免 Web-Bird 200 类时图片爆炸。
    max_samples = max(0, int(max_samples))
    samples_per_class = max(1, int(samples_per_class))
    class_counts = {}
    selected = []
    for sample_id, rows in rows_by_sample.items():
        if len(selected) >= max_samples:
            break
        label = int(rows[0]['label'])
        if class_counts.get(label, 0) >= samples_per_class:
            continue
        class_counts[label] = class_counts.get(label, 0) + 1
        selected.append((sample_id, rows))
    return selected


def _candidate_entropy(candidate_scores, candidate_size=None):
    # 候选分数先在集合内 softmax；C2-v2 允许 noisy label 追加后出现变长候选集合。
    if candidate_scores.numel() == 0:
        return candidate_scores.new_zeros(candidate_scores.size(0))
    output_dtype = candidate_scores.dtype
    candidate_scores_fp32 = candidate_scores.float()
    if candidate_size is None:
        candidate_probs = candidate_scores_fp32.softmax(dim=1)
        entropy = -(candidate_probs * candidate_probs.clamp(min=1e-12).log()).sum(dim=1)
        return entropy.to(dtype=output_dtype)

    candidate_size = candidate_size.to(device=candidate_scores_fp32.device, dtype=torch.long)
    valid_positions = (
        torch.arange(candidate_scores_fp32.size(1), device=candidate_scores_fp32.device)[None, :]
        < candidate_size[:, None]
    )
    # fp16 不能表示 -1e9，masked softmax 临时用 fp32 与 finfo 最小值规避溢出。
    mask_value = torch.finfo(candidate_scores_fp32.dtype).min
    masked_scores = candidate_scores_fp32.masked_fill(~valid_positions, mask_value)
    candidate_probs = masked_scores.softmax(dim=1) * valid_positions.to(dtype=candidate_scores_fp32.dtype)
    candidate_probs = candidate_probs / candidate_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    entropy = -(candidate_probs * candidate_probs.clamp(min=1e-12).log()).sum(dim=1)
    return entropy.to(dtype=output_dtype)


def _build_capped_candidate_target(
        candidate_scores,
        candidate_indices,
        candidate_size,
        labels,
        target_temp=2.0,
        top1_cap=0.5,
        noisy_prior=0.0):
    # C2-v3 目标分布来自 evidence score，但给 top1 设上限，并给 noisy label 一个小保底。
    if candidate_scores.numel() == 0:
        return candidate_scores.new_zeros(candidate_scores.shape)

    output_dtype = candidate_scores.dtype
    candidate_scores_fp32 = candidate_scores.float()
    candidate_indices = candidate_indices.to(device=candidate_scores_fp32.device, dtype=torch.long)
    labels = labels.to(device=candidate_scores_fp32.device, dtype=torch.long)
    candidate_size = candidate_size.to(device=candidate_scores_fp32.device, dtype=torch.long)
    valid_positions = (
        torch.arange(candidate_scores_fp32.size(1), device=candidate_scores_fp32.device)[None, :]
        < candidate_size[:, None]
    )
    target_temp = max(float(target_temp), 1e-6)
    # fp16 下 -1e9 会溢出，target 构造全程用 fp32 与 finfo 最小值，最后再转回原 dtype。
    mask_value = torch.finfo(candidate_scores_fp32.dtype).min
    masked_scores = (candidate_scores_fp32 / target_temp).masked_fill(~valid_positions, mask_value)
    target = masked_scores.softmax(dim=1) * valid_positions.to(dtype=candidate_scores_fp32.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-12)

    top1_cap = max(0.0, min(float(top1_cap), 1.0))
    if top1_cap < 1.0 and candidate_scores.size(1) > 1:
        non_top1_positions = valid_positions.clone()
        non_top1_positions[:, 0] = False
        has_other = non_top1_positions.any(dim=1)
        top1_excess = torch.relu(target[:, 0] - top1_cap)
        top1_excess = torch.where(has_other, top1_excess, torch.zeros_like(top1_excess))
        target[:, 0] = target[:, 0] - top1_excess
        non_top1_mass = (target * non_top1_positions.to(dtype=target.dtype)).sum(dim=1).clamp(min=1e-12)
        target = target + (
            top1_excess[:, None]
            * target
            * non_top1_positions.to(dtype=target.dtype)
            / non_top1_mass[:, None]
        )

    noisy_prior = max(0.0, min(float(noisy_prior), 1.0))
    if noisy_prior > 0.0:
        noisy_positions = (candidate_indices == labels[:, None]) & valid_positions
        non_noisy_positions = valid_positions & noisy_positions.logical_not()
        noisy_mass = (target * noisy_positions.to(dtype=target.dtype)).sum(dim=1)
        non_noisy_mass = (target * non_noisy_positions.to(dtype=target.dtype)).sum(dim=1)
        noisy_deficit = torch.relu(target.new_full(noisy_mass.shape, noisy_prior) - noisy_mass)
        transfer = torch.minimum(noisy_deficit, non_noisy_mass)
        target = target - (
            transfer[:, None]
            * target
            * non_noisy_positions.to(dtype=target.dtype)
            / non_noisy_mass.clamp(min=1e-12)[:, None]
        )
        noisy_count = noisy_positions.sum(dim=1).clamp(min=1).to(dtype=target.dtype)
        target = target + transfer[:, None] * noisy_positions.to(dtype=target.dtype) / noisy_count[:, None]

    target = target * valid_positions.to(dtype=target.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return target.to(dtype=output_dtype)


def _candidate_indices_to_mask(candidate_indices, num_classes):
    candidate_mask = torch.zeros(
        candidate_indices.size(0), num_classes,
        device=candidate_indices.device, dtype=torch.float
    )
    if candidate_indices.numel() > 0:
        candidate_mask.scatter_(dim=1, index=candidate_indices.long(), value=1.0)
    return candidate_mask


def _candidate_mask_to_padded_candidates(candidate_mask, candidate_scores_all):
    # 候选集合用 mask 训练、用按分数排序的 padding 列表写日志，二者解耦避免日志格式影响 loss。
    candidate_size = candidate_mask.sum(dim=1).long()
    max_size = int(candidate_size.max().item()) if candidate_size.numel() > 0 else 0
    candidate_indices = torch.full(
        (candidate_mask.size(0), max_size), -1,
        device=candidate_mask.device, dtype=torch.long
    )
    candidate_scores = candidate_scores_all.new_zeros(candidate_mask.size(0), max_size)

    for row_idx in range(candidate_mask.size(0)):
        selected = torch.nonzero(candidate_mask[row_idx].bool(), as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        selected_scores = candidate_scores_all[row_idx, selected]
        order = selected_scores.argsort(descending=True)
        selected = selected[order]
        selected_scores = selected_scores[order]
        row_size = int(selected.numel())
        candidate_indices[row_idx, :row_size] = selected
        candidate_scores[row_idx, :row_size] = selected_scores

    return candidate_indices, candidate_scores, candidate_size


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


def _fixed_size_bbox_around_center(cx, cy, width, height, image_h, image_w):
    # D1 peak window 保持固定窗口大小；靠近边界时整体平移到图内，而不是压缩窗口。
    width = max(1, min(int(width), image_w))
    height = max(1, min(int(height), image_h))
    left = int(round(cx - width / 2.0))
    top = int(round(cy - height / 2.0))
    left = max(0, min(left, image_w - width))
    top = max(0, min(top, image_h - height))
    return left, top, left + width, top + height


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


def _bboxes_to_masks(image_h, image_w, bboxes, device):
    # D1-v2 bbox 擦除模式使用完整窗口/外接框，强制后续 CAM 避开已选区域。
    return torch.stack([_bbox_mask(image_h, image_w, bbox, device) for bbox in bboxes], dim=0)


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
