# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.local_evidence import (
    _autocast_disabled,
    _bboxes_to_masks,
    _extract_logits,
    cam_to_bbox,
    cam_to_peak_window,
    crop_by_bbox,
    erase_by_mask,
    generate_cam,
)


EG_PSSM_CSV_HEADER = (
    'epoch,batch_idx,actual_backend,lambda_ssm,lambda_ssm_grad_norm,'
    'eg_pssm_grad_norm,enabled_clean_ratio,gate_mean,gate_std,'
    'z_global_norm_mean,z_final_norm_mean,z_ssm_norm_mean,residual_ratio_mean,'
    'global_acc,final_acc,final_minus_global,'
    'pred_evidence_high_global_acc,pred_evidence_high_final_acc,pred_evidence_high_gain,'
    'pred_evidence_mid_gain,pred_evidence_low_gain,changed_correct,changed_wrong'
)


class TorchSSMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 轻量后端只做对角状态递推，表达 token 间关系；它不是官方 Mamba selective scan。
        x_norm = self.in_norm(x)
        drive = torch.tanh(self.in_proj(x_norm))
        input_gate = torch.sigmoid(self.input_gate(x_norm))
        decay = torch.sigmoid(self.log_decay).view(1, -1)
        state = x.new_zeros(x.size(0), x.size(2))
        outputs = []
        for step in range(x.size(1)):
            state = decay * state + (1.0 - decay) * drive[:, step] * input_gate[:, step]
            outputs.append(state)
        y = torch.stack(outputs, dim=1)
        return x + self.out_proj(self.out_norm(y))


def _build_sequence_backend(backend, d_model):
    backend = str(backend).lower()
    if backend in ['auto', 'mamba']:
        try:
            from mamba_ssm import Mamba
            return Mamba(d_model=d_model, d_state=16, d_conv=2, expand=2), 'mamba'
        except Exception as exc:
            if backend == 'mamba':
                raise RuntimeError('eg_pssm_backend=mamba requires a working mamba_ssm installation.') from exc
    if backend in ['auto', 'torch_ssm']:
        return TorchSSMBlock(d_model), 'torch_ssm'
    raise ValueError(f'eg_pssm_backend should be auto, mamba, or torch_ssm, got {backend}.')


class EGPSSMModule(nn.Module):
    def __init__(
            self,
            feature_dim,
            num_parts=3,
            backend='auto',
            use_diff_token=True,
            gate_mode='continuous',
            gate_min=0.05,
            lambda_init=0.001,
            bidirectional=True,
            sort_parts='evidence_desc'):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_parts = int(num_parts)
        self.use_diff_token = bool(use_diff_token)
        self.gate_mode = str(gate_mode).lower()
        self.gate_min = float(gate_min)
        self.bidirectional = bool(bidirectional)
        self.sort_parts = str(sort_parts).lower()

        token_input_dim = self.feature_dim * (2 if self.use_diff_token else 1) + 1
        self.global_token = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        self.part_token = nn.Sequential(
            nn.LayerNorm(token_input_dim),
            nn.Linear(token_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        self.forward_backend, actual_backend = _build_sequence_backend(backend, self.feature_dim)
        self.backward_backend = None
        if self.bidirectional:
            self.backward_backend, backward_backend = _build_sequence_backend(
                actual_backend, self.feature_dim
            )
            if backward_backend != actual_backend:
                raise RuntimeError('EG-PSSM bidirectional backends are inconsistent.')
        self.actual_backend = actual_backend
        self.pool_score = nn.Linear(self.feature_dim, 1)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.lambda_ssm = nn.Parameter(torch.tensor(float(lambda_init)))

    def forward(self, z_global, part_features, erase_features, evidence, valid_part_mask=None):
        if z_global.numel() == 0:
            return self._empty_output(z_global)

        part_features, erase_features, evidence, valid_part_mask = self._prepare_parts(
            part_features, erase_features, evidence, valid_part_mask
        )
        evidence_token = evidence.to(dtype=part_features.dtype).unsqueeze(-1)
        if self.use_diff_token:
            part_input = torch.cat([part_features, part_features - erase_features, evidence_token], dim=-1)
        else:
            part_input = torch.cat([part_features, evidence_token], dim=-1)

        # token 序列固定为 global + 按证据排序后的 part token，避免额外 concat/mean fusion。
        global_token = self.global_token(z_global).unsqueeze(1)
        part_tokens = self.part_token(part_input)
        sequence = torch.cat([global_token, part_tokens], dim=1)
        sequence_out = self.forward_backend(sequence)
        if self.backward_backend is not None:
            backward_out = self.backward_backend(torch.flip(sequence, dims=[1]))
            sequence_out = 0.5 * (sequence_out + torch.flip(backward_out, dims=[1]))

        part_out = sequence_out[:, 1:]
        pool_logits = self.pool_score(part_out).squeeze(-1) + evidence.detach()
        pool_logits = pool_logits.masked_fill(~valid_part_mask, -1e4)
        pool_weights = F.softmax(pool_logits, dim=1)
        z_ssm = (part_out * pool_weights.unsqueeze(-1)).sum(dim=1)
        residual = self.residual_proj(z_ssm)
        gate = self._build_gate(evidence, valid_part_mask).detach()
        z_final = z_global + self.lambda_ssm * gate.view(-1, 1) * residual
        return {
            'z_final': z_final,
            'z_ssm': z_ssm,
            'residual': residual,
            'gate': gate,
            'pool_weights': pool_weights,
            'sorted_evidence': evidence,
        }

    def _prepare_parts(self, part_features, erase_features, evidence, valid_part_mask):
        if valid_part_mask is None:
            valid_part_mask = torch.ones_like(evidence, dtype=torch.bool)
        else:
            valid_part_mask = valid_part_mask.to(device=evidence.device, dtype=torch.bool)
        evidence = evidence.to(device=part_features.device, dtype=part_features.dtype)
        if self.sort_parts == 'evidence_desc':
            order = evidence.argsort(dim=1, descending=True)
            gather_index = order.unsqueeze(-1).expand(-1, -1, part_features.size(-1))
            part_features = part_features.gather(1, gather_index)
            erase_features = erase_features.gather(1, gather_index)
            evidence = evidence.gather(1, order)
            valid_part_mask = valid_part_mask.gather(1, order)
        elif self.sort_parts not in ['none', 'original']:
            raise ValueError(f'eg_pssm_sort_parts should be evidence_desc or none, got {self.sort_parts}.')
        return part_features, erase_features, evidence, valid_part_mask

    def _build_gate(self, evidence, valid_part_mask):
        gate_mode = self.gate_mode
        if gate_mode in ['none', 'off', 'no_gate']:
            return evidence.new_ones(evidence.size(0))
        if gate_mode != 'continuous':
            raise ValueError(f'eg_pssm_gate_mode should be continuous or none, got {gate_mode}.')

        masked_evidence = evidence.masked_fill(~valid_part_mask, -1e4)
        max_evidence = masked_evidence.max(dim=1)[0]
        if max_evidence.numel() <= 1:
            return max_evidence.new_full(max_evidence.shape, 0.5)
        evidence_min = max_evidence.min()
        evidence_max = max_evidence.max()
        if (evidence_max - evidence_min).abs() <= 1e-12:
            return max_evidence.new_full(max_evidence.shape, 0.5)
        gate = (max_evidence - evidence_min) / (evidence_max - evidence_min)
        return gate.clamp(min=self.gate_min, max=1.0)

    def _empty_output(self, z_global):
        empty_scalar = z_global.new_zeros((0,))
        return {
            'z_final': z_global,
            'z_ssm': z_global,
            'residual': z_global,
            'gate': empty_scalar,
            'pool_weights': z_global.new_zeros((0, self.num_parts)),
            'sorted_evidence': z_global.new_zeros((0, self.num_parts)),
        }


def build_eg_pssm_part_batch(
        model,
        images,
        labels,
        selected_indices,
        num_parts=3,
        use_accum_erase=True,
        crop_mode='peak_window',
        window_ratio=0.35,
        erase_mode='peak_window',
        cam_quantile=0.8,
        min_area=0.05,
        max_area=0.7,
        bbox_padding=0.05,
        cam_type='weightcam'):
    selected_indices = selected_indices.detach().long()
    num_selected = int(selected_indices.numel())
    if num_selected == 0:
        return _empty_part_batch(images, selected_indices, int(num_parts))

    num_parts = max(1, int(num_parts))
    crop_mode = str(crop_mode).lower()
    erase_mode = str(erase_mode).lower()
    selected_images = images.index_select(0, selected_indices).float()
    selected_labels = labels.index_select(0, selected_indices).long()

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_disabled(selected_images):
            image_h, image_w = selected_images.shape[-2:]
            logits_ori = _extract_logits(model(selected_images))
            p_ori_y = logits_ori.detach().softmax(dim=1).gather(
                1, selected_labels.view(-1, 1)
            ).squeeze(1)
            erase_accum_mask = torch.zeros(
                selected_images.size(0), image_h, image_w,
                device=selected_images.device, dtype=torch.bool
            )
            cam_input = selected_images
            x_part_list = []
            x_erase_list = []
            evidence_list = []
            valid_list = []
            bbox_area_list = []

            for _ in range(num_parts):
                # clean-only 训练版固定用 noisy/web label 作为 CAM 和 evidence target。
                cam, _ = generate_cam(model, cam_input, selected_labels, cam_type=cam_type)
                cam_bboxes, cam_bbox_areas, cam_masks = cam_to_bbox(
                    cam, selected_images.shape[-2:], quantile=cam_quantile,
                    min_area=min_area, max_area=max_area, padding=bbox_padding
                )
                peak_bboxes, peak_bbox_areas, peak_masks = cam_to_peak_window(
                    cam, selected_images.shape[-2:], window_ratio=window_ratio
                )
                if crop_mode == 'bbox':
                    crop_bboxes, crop_bbox_areas = cam_bboxes, cam_bbox_areas
                elif crop_mode == 'peak_window':
                    crop_bboxes, crop_bbox_areas = peak_bboxes, peak_bbox_areas
                else:
                    raise ValueError(f'eg_pssm crop_mode should be bbox or peak_window, got {crop_mode}.')

                if erase_mode == 'cam_mask':
                    erase_masks = cam_masks.bool()
                elif erase_mode == 'bbox':
                    erase_masks = _bboxes_to_masks(image_h, image_w, cam_bboxes, selected_images.device)
                elif erase_mode == 'peak_window':
                    erase_masks = peak_masks.bool()
                else:
                    raise ValueError(
                        f'eg_pssm erase_mode should be cam_mask, bbox, or peak_window, got {erase_mode}.'
                    )

                x_part = crop_by_bbox(selected_images, crop_bboxes)
                x_erase_single = erase_by_mask(selected_images, erase_masks, fill_value=0.0)
                logits_part = _extract_logits(model(x_part))
                logits_erase = _extract_logits(model(x_erase_single))
                probs_part = logits_part.detach().softmax(dim=1)
                probs_erase = logits_erase.detach().softmax(dim=1)
                p_part_y = probs_part.gather(1, selected_labels.view(-1, 1)).squeeze(1)
                p_erase_y = probs_erase.gather(1, selected_labels.view(-1, 1)).squeeze(1)
                evidence = p_ori_y * p_part_y * (p_ori_y - p_erase_y).clamp(min=0)
                bbox_areas = torch.tensor(
                    crop_bbox_areas, device=selected_images.device, dtype=selected_images.dtype
                )
                valid_part = torch.isfinite(bbox_areas) & (bbox_areas > 0)

                x_part_list.append(x_part)
                x_erase_list.append(x_erase_single)
                evidence_list.append(evidence)
                valid_list.append(valid_part)
                bbox_area_list.append(bbox_areas)

                erase_accum_mask = erase_accum_mask | erase_masks
                erase_mask_for_next_cam = erase_accum_mask if use_accum_erase else erase_masks
                cam_input = erase_by_mask(selected_images, erase_mask_for_next_cam, fill_value=0.0)
    finally:
        if was_training:
            model.train()

    x_part_views = torch.stack(x_part_list, dim=1)
    x_erase_views = torch.stack(x_erase_list, dim=1)
    evidence = torch.stack(evidence_list, dim=1)
    valid_part_mask = torch.stack(valid_list, dim=1).bool()
    bbox_area = torch.stack(bbox_area_list, dim=1)
    valid_sample_mask = valid_part_mask.all(dim=1)
    return {
        'x_part': x_part_views,
        'x_erase': x_erase_views,
        'evidence': evidence,
        'valid_part_mask': valid_part_mask,
        'valid_sample_mask': valid_sample_mask,
        'bbox_area': bbox_area,
        'batch_indices': selected_indices,
        'labels': selected_labels,
        'num_selected': num_selected,
        'num_valid': int(valid_sample_mask.sum().item()),
        'num_parts': num_parts,
    }


def _empty_part_batch(images, selected_indices, num_parts):
    return {
        'x_part': images.new_zeros((0, num_parts, *images.shape[1:])),
        'x_erase': images.new_zeros((0, num_parts, *images.shape[1:])),
        'evidence': images.new_zeros((0, num_parts)),
        'valid_part_mask': torch.zeros((0, num_parts), device=images.device, dtype=torch.bool),
        'valid_sample_mask': torch.zeros((0,), device=images.device, dtype=torch.bool),
        'bbox_area': images.new_zeros((0, num_parts)),
        'batch_indices': selected_indices,
        'labels': torch.zeros((0,), device=images.device, dtype=torch.long),
        'num_selected': int(selected_indices.numel()),
        'num_valid': 0,
        'num_parts': int(num_parts),
    }


def build_eg_pssm_log_row(
        epoch,
        batch_idx,
        actual_backend,
        lambda_ssm,
        num_clean,
        batch_size,
        labels=None,
        global_logits=None,
        final_logits=None,
        gate=None,
        evidence=None,
        z_global=None,
        z_final=None,
        z_ssm=None,
        residual=None):
    enabled = 0 if labels is None else int(labels.numel())
    row = {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'actual_backend': str(actual_backend),
        'lambda_ssm': float(lambda_ssm.detach().item() if torch.is_tensor(lambda_ssm) else lambda_ssm),
        'lambda_ssm_grad_norm': 0.0,
        'eg_pssm_grad_norm': 0.0,
        'enabled_clean_ratio': enabled / max(int(num_clean), 1),
        'gate_mean': _safe_mean(gate),
        'gate_std': _safe_std(gate),
        'z_global_norm_mean': _safe_norm_mean(z_global),
        'z_final_norm_mean': _safe_norm_mean(z_final),
        'z_ssm_norm_mean': _safe_norm_mean(z_ssm),
        'residual_ratio_mean': 0.0,
        'global_acc': 0.0,
        'final_acc': 0.0,
        'final_minus_global': 0.0,
        'pred_evidence_high_global_acc': 0.0,
        'pred_evidence_high_final_acc': 0.0,
        'pred_evidence_high_gain': 0.0,
        'pred_evidence_mid_gain': 0.0,
        'pred_evidence_low_gain': 0.0,
        'changed_correct': 0,
        'changed_wrong': 0,
    }
    if enabled == 0:
        return row

    z_global_norm = z_global.detach().norm(dim=1).clamp_min(1e-12)
    residual_norm = (lambda_ssm.detach() * gate.detach() * residual.detach().norm(dim=1)).abs()
    row['residual_ratio_mean'] = float((residual_norm / z_global_norm).mean().item())

    global_pred = global_logits.detach().argmax(dim=1)
    final_pred = final_logits.detach().argmax(dim=1)
    labels = labels.detach().long()
    global_correct = global_pred.eq(labels)
    final_correct = final_pred.eq(labels)
    row['global_acc'] = _mask_acc(global_correct)
    row['final_acc'] = _mask_acc(final_correct)
    row['final_minus_global'] = row['final_acc'] - row['global_acc']
    changed = global_pred.ne(final_pred)
    row['changed_correct'] = int((changed & ~global_correct & final_correct).sum().item())
    row['changed_wrong'] = int((changed & global_correct & ~final_correct).sum().item())

    evidence_score = evidence.detach().max(dim=1)[0]
    high_mask, mid_mask, low_mask = _evidence_tercile_masks(evidence_score)
    row['pred_evidence_high_global_acc'] = _mask_acc(global_correct, high_mask)
    row['pred_evidence_high_final_acc'] = _mask_acc(final_correct, high_mask)
    row['pred_evidence_high_gain'] = row['pred_evidence_high_final_acc'] - row['pred_evidence_high_global_acc']
    row['pred_evidence_mid_gain'] = _mask_acc(final_correct, mid_mask) - _mask_acc(global_correct, mid_mask)
    row['pred_evidence_low_gain'] = _mask_acc(final_correct, low_mask) - _mask_acc(global_correct, low_mask)
    return row


def update_eg_pssm_grad_fields(row, module):
    lambda_grad = module.lambda_ssm.grad
    row['lambda_ssm_grad_norm'] = 0.0 if lambda_grad is None else float(lambda_grad.detach().abs().item())
    row['eg_pssm_grad_norm'] = _module_grad_norm(module, exclude_names={'lambda_ssm'})
    return row


def format_eg_pssm_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['actual_backend']},"
        f"{row['lambda_ssm']:.8f},{row['lambda_ssm_grad_norm']:.8f},"
        f"{row['eg_pssm_grad_norm']:.8f},{row['enabled_clean_ratio']:.6f},"
        f"{row['gate_mean']:.6f},{row['gate_std']:.6f},"
        f"{row['z_global_norm_mean']:.6f},{row['z_final_norm_mean']:.6f},"
        f"{row['z_ssm_norm_mean']:.6f},{row['residual_ratio_mean']:.6f},"
        f"{row['global_acc']:.6f},{row['final_acc']:.6f},{row['final_minus_global']:.6f},"
        f"{row['pred_evidence_high_global_acc']:.6f},"
        f"{row['pred_evidence_high_final_acc']:.6f},"
        f"{row['pred_evidence_high_gain']:.6f},"
        f"{row['pred_evidence_mid_gain']:.6f},"
        f"{row['pred_evidence_low_gain']:.6f},"
        f"{row['changed_correct']},{row['changed_wrong']}"
    )


def eg_pssm_checkpoint_items(module):
    if module is None:
        return {}
    return {'eg_pssm_state_dict': module.state_dict()}


def _safe_mean(values):
    if values is None or values.numel() == 0:
        return 0.0
    return float(values.detach().float().mean().item())


def _safe_std(values):
    if values is None or values.numel() <= 1:
        return 0.0
    return float(values.detach().float().std(unbiased=False).item())


def _safe_norm_mean(values):
    if values is None or values.numel() == 0:
        return 0.0
    return float(values.detach().float().norm(dim=1).mean().item())


def _mask_acc(correct, mask=None):
    if mask is None:
        denom = correct.numel()
        return 0.0 if denom == 0 else float(correct.float().mean().item() * 100.0)
    mask = mask.to(device=correct.device, dtype=torch.bool)
    denom = int(mask.sum().item())
    if denom == 0:
        return 0.0
    return float(correct[mask].float().mean().item() * 100.0)


def _evidence_tercile_masks(evidence_score):
    n_items = int(evidence_score.numel())
    high = torch.zeros_like(evidence_score, dtype=torch.bool)
    mid = torch.zeros_like(evidence_score, dtype=torch.bool)
    low = torch.zeros_like(evidence_score, dtype=torch.bool)
    if n_items == 0:
        return high, mid, low
    order = evidence_score.argsort(descending=True)
    high_count = max(1, int(math.ceil(n_items / 3.0)))
    mid_count = max(0, int(math.ceil((n_items - high_count) / 2.0)))
    high[order[:high_count]] = True
    if mid_count > 0:
        mid[order[high_count:high_count + mid_count]] = True
    if high_count + mid_count < n_items:
        low[order[high_count + mid_count:]] = True
    return high, mid, low


def _module_grad_norm(module, exclude_names=None):
    exclude_names = set() if exclude_names is None else set(exclude_names)
    total = 0.0
    for name, param in module.named_parameters():
        if name in exclude_names or param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(grad.float().pow(2).sum().item())
    return math.sqrt(total)
