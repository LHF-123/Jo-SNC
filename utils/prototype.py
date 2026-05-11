# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


EVIDENCE_PROTO_CSV_HEADER = (
    'epoch,batch_idx,num_clean,num_valid_part,num_valid_evidence,'
    'num_update_candidate,num_proto_update,num_proto_loss_valid,'
    'proto_update_ratio,proto_loss_valid_ratio,proto_ce_raw,'
    'proto_ce_weighted,proto_temp,proto_loss_ratio,'
    'proto_target_prob_mean,proto_target_logit_mean,'
    'proto_max_non_target_prob_mean,proto_entropy_mean,nearest_proto_acc,'
    'target_proto_available_ratio,prototype_num_initialized,'
    'prototype_initialized_ratio,prototype_update_count_mean,'
    'prototype_update_count_min,prototype_update_count_max,'
    'update_weight_mean,update_weight_std,update_weight_min,'
    'update_weight_max,update_effective_sample_size,evidence_rank_mean,'
    'evidence_rank_std,update_evidence_score_mean,update_p_ori_y_mean,'
    'prototype_drift_mean,prototype_drift_max,skip_reason'
)


def init_prototype_state(num_classes, feature_dim, device):
    # EAPA prototype bank 是训练状态缓冲区，不作为可学习参数参与 optimizer。
    return {
        'bank': torch.zeros(num_classes, feature_dim, device=device),
        'count': torch.zeros(num_classes, device=device),
        'valid': torch.zeros(num_classes, device=device, dtype=torch.bool),
    }


def load_prototype_state(prototype_state, checkpoint, device):
    if prototype_state is None:
        return
    if 'prototype_bank' in checkpoint:
        prototype_state['bank'].copy_(checkpoint['prototype_bank'].to(device))
    if 'prototype_count' in checkpoint:
        prototype_state['count'].copy_(checkpoint['prototype_count'].to(device))
    if 'prototype_valid' in checkpoint:
        prototype_state['valid'].copy_(checkpoint['prototype_valid'].to(device))


def prototype_checkpoint_items(prototype_state):
    if prototype_state is None:
        return {}
    return {
        'prototype_bank': prototype_state['bank'].detach().cpu(),
        'prototype_count': prototype_state['count'].detach().cpu(),
        'prototype_valid': prototype_state['valid'].detach().cpu(),
    }


def _zero_float(tensor):
    return float(tensor.item()) if torch.is_tensor(tensor) and tensor.numel() == 1 else 0.0


def _safe_mean(values):
    if values is None or values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _safe_std(values):
    if values is None or values.numel() <= 1:
        return 0.0
    return float(values.float().std(unbiased=False).item())


def _update_count_stats(prototype_state):
    valid = prototype_state['valid']
    counts = prototype_state['count'][valid]
    if counts.numel() == 0:
        return 0.0, 0.0, 0.0
    return (
        float(counts.float().mean().item()),
        float(counts.float().min().item()),
        float(counts.float().max().item()),
    )


def _prototype_initialized_stats(prototype_state):
    valid = prototype_state['valid']
    num_initialized = int(valid.sum().item())
    initialized_ratio = float(num_initialized / max(valid.numel(), 1))
    return num_initialized, initialized_ratio


def compute_evidence_update_weights(part_batch, conf_thr=0.60, eps=1e-12):
    num_valid = int(part_batch.get('num_valid', 0))
    if num_valid <= 0:
        return {
            'candidate_mask': None,
            'weights': None,
            'rank_norm': None,
            'num_valid_evidence': 0,
            'num_update_candidate': 0,
            'update_weight_mean': 0.0,
            'update_weight_std': 0.0,
            'update_weight_min': 0.0,
            'update_weight_max': 0.0,
            'update_effective_sample_size': 0.0,
            'evidence_rank_mean': 0.0,
            'evidence_rank_std': 0.0,
            'update_evidence_score_mean': 0.0,
            'update_p_ori_y_mean': 0.0,
            'skip_reason': 'no_valid_part',
        }

    p_ori_y = part_batch['p_ori_y'].detach().float()
    evidence_score = part_batch['evidence_score'].detach().float()
    candidate_mask = p_ori_y >= float(conf_thr)
    num_candidates = int(candidate_mask.sum().item())
    rank_norm = torch.zeros_like(evidence_score)
    weights = torch.zeros_like(evidence_score)

    if num_candidates == 0:
        return {
            'candidate_mask': candidate_mask,
            'weights': weights,
            'rank_norm': rank_norm,
            'num_valid_evidence': num_valid,
            'num_update_candidate': 0,
            'update_weight_mean': 0.0,
            'update_weight_std': 0.0,
            'update_weight_min': 0.0,
            'update_weight_max': 0.0,
            'update_effective_sample_size': 0.0,
            'evidence_rank_mean': 0.0,
            'evidence_rank_std': 0.0,
            'update_evidence_score_mean': 0.0,
            'update_p_ori_y_mean': 0.0,
            'skip_reason': 'no_update_candidate',
        }

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    candidate_scores = evidence_score[candidate_indices]
    if num_candidates == 1:
        candidate_rank = torch.ones(1, device=evidence_score.device, dtype=evidence_score.dtype)
    else:
        # rank 只在通过 p_ori_y 门槛的候选集合内计算，避免低置信样本改变有效候选的相对权重。
        order = torch.argsort(candidate_scores, descending=False)
        candidate_rank = torch.zeros_like(candidate_scores)
        candidate_rank[order] = torch.arange(
            1, num_candidates + 1, device=evidence_score.device, dtype=evidence_score.dtype
        ) / float(num_candidates)

    rank_norm[candidate_indices] = candidate_rank
    weights[candidate_indices] = candidate_rank * p_ori_y[candidate_indices]
    candidate_weights = weights[candidate_indices]
    weight_sum = candidate_weights.sum()
    ess = (weight_sum * weight_sum) / (candidate_weights.square().sum() + eps)

    return {
        'candidate_mask': candidate_mask,
        'weights': weights,
        'rank_norm': rank_norm,
        'num_valid_evidence': num_valid,
        'num_update_candidate': num_candidates,
        'update_weight_mean': _safe_mean(candidate_weights),
        'update_weight_std': _safe_std(candidate_weights),
        'update_weight_min': float(candidate_weights.min().item()),
        'update_weight_max': float(candidate_weights.max().item()),
        'update_effective_sample_size': float(ess.item()),
        'evidence_rank_mean': _safe_mean(candidate_rank),
        'evidence_rank_std': _safe_std(candidate_rank),
        'update_evidence_score_mean': _safe_mean(evidence_score[candidate_indices]),
        'update_p_ori_y_mean': _safe_mean(p_ori_y[candidate_indices]),
        'skip_reason': 'none',
    }


def compute_confidence_update_weights(labels, selected_indices, teacher_logits, conf_thr=0.60, eps=1e-12):
    selected_indices = selected_indices.detach().long()
    if selected_indices.numel() == 0:
        return {
            'batch_indices': selected_indices,
            'labels': labels[:0],
            'weights': teacher_logits.new_zeros((0,)),
            'num_update_candidate': 0,
            'update_weight_mean': 0.0,
            'update_weight_std': 0.0,
            'update_weight_min': 0.0,
            'update_weight_max': 0.0,
            'update_effective_sample_size': 0.0,
            'update_p_ori_y_mean': 0.0,
            'skip_reason': 'no_clean',
        }

    probs = teacher_logits.detach().float().softmax(dim=1)
    selected_labels = labels[selected_indices].detach().long()
    p_ori_y = probs[selected_indices].gather(1, selected_labels[:, None]).squeeze(1)
    candidate_mask = p_ori_y >= float(conf_thr)
    candidate_indices = selected_indices[candidate_mask]
    candidate_labels = selected_labels[candidate_mask]
    candidate_weights = p_ori_y[candidate_mask]
    num_candidates = int(candidate_indices.numel())
    if num_candidates == 0:
        return {
            'batch_indices': candidate_indices,
            'labels': candidate_labels,
            'weights': candidate_weights,
            'num_update_candidate': 0,
            'update_weight_mean': 0.0,
            'update_weight_std': 0.0,
            'update_weight_min': 0.0,
            'update_weight_max': 0.0,
            'update_effective_sample_size': 0.0,
            'update_p_ori_y_mean': 0.0,
            'skip_reason': 'no_update_candidate',
        }

    weight_sum = candidate_weights.sum()
    ess = (weight_sum * weight_sum) / (candidate_weights.square().sum() + eps)
    return {
        'batch_indices': candidate_indices,
        'labels': candidate_labels,
        'weights': candidate_weights,
        'num_update_candidate': num_candidates,
        'update_weight_mean': _safe_mean(candidate_weights),
        'update_weight_std': _safe_std(candidate_weights),
        'update_weight_min': float(candidate_weights.min().item()),
        'update_weight_max': float(candidate_weights.max().item()),
        'update_effective_sample_size': float(ess.item()),
        'update_p_ori_y_mean': _safe_mean(candidate_weights),
        'skip_reason': 'none',
    }


def update_prototype_bank_weighted(prototype_state, features, labels, weights, momentum=0.90, eps=1e-12):
    if prototype_state is None or features is None or features.numel() == 0 or weights.numel() == 0:
        return {'num_proto_update': 0, 'prototype_drift_mean': 0.0, 'prototype_drift_max': 0.0}

    with torch.no_grad():
        bank = prototype_state['bank']
        count = prototype_state['count']
        valid = prototype_state['valid']
        features = F.normalize(features.detach().float(), dim=1)
        labels = labels.detach().long()
        weights = weights.detach().float()
        usable = torch.isfinite(weights) & (weights > eps)
        if int(usable.sum().item()) == 0:
            return {'num_proto_update': 0, 'prototype_drift_mean': 0.0, 'prototype_drift_max': 0.0}

        labels = labels[usable]
        features = features[usable]
        weights = weights[usable]
        num_updated_samples = int(labels.numel())
        drifts = []

        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            class_weights = weights[class_mask]
            weight_sum = class_weights.sum()
            if float(weight_sum.item()) <= eps:
                continue
            class_features = features[class_mask]
            weighted_mean = (class_features * class_weights[:, None]).sum(dim=0) / (weight_sum + eps)
            weighted_mean = F.normalize(weighted_mean, dim=0)
            class_idx = int(class_id.item())
            was_valid = bool(valid[class_idx].item())
            if was_valid:
                before = bank[class_idx].detach().clone()
                bank[class_idx] = F.normalize(
                    float(momentum) * bank[class_idx] + (1.0 - float(momentum)) * weighted_mean,
                    dim=0,
                )
                drift = 1.0 - torch.dot(before, bank[class_idx]).clamp(min=-1.0, max=1.0)
                drifts.append(float(drift.item()))
            else:
                bank[class_idx] = weighted_mean
                valid[class_idx] = True
            count[class_idx] += float(class_mask.sum().item())

        if len(drifts) == 0:
            drift_mean, drift_max = 0.0, 0.0
        else:
            drift_mean, drift_max = float(sum(drifts) / len(drifts)), float(max(drifts))
        return {
            'num_proto_update': num_updated_samples,
            'prototype_drift_mean': drift_mean,
            'prototype_drift_max': drift_max,
        }


def compute_prototype_softmax_loss(features, labels, selected_indices, prototype_state, temperature=0.2):
    selected_indices = selected_indices.detach().long()
    device = features.device
    zero_loss = features.new_tensor(0.0)
    empty = {
        'loss': zero_loss,
        'num_proto_loss_valid': 0,
        'proto_ce_raw': 0.0,
        'proto_target_prob_mean': 0.0,
        'proto_target_logit_mean': 0.0,
        'proto_max_non_target_prob_mean': 0.0,
        'proto_entropy_mean': 0.0,
        'nearest_proto_acc': 0.0,
        'target_proto_available_ratio': 0.0,
        'skip_reason': 'not_run',
    }
    if selected_indices.numel() == 0:
        empty['skip_reason'] = 'no_clean'
        return empty

    valid_proto = prototype_state['valid'].to(device=device)
    num_valid_proto = int(valid_proto.sum().item())
    selected_labels = labels[selected_indices].detach().long()
    target_available = valid_proto[selected_labels]
    empty['target_proto_available_ratio'] = _safe_mean(target_available.float())
    if num_valid_proto < 2:
        empty['skip_reason'] = 'prototype_initialized_lt_2'
        return empty

    loss_mask = target_available
    if int(loss_mask.sum().item()) == 0:
        empty['skip_reason'] = 'target_proto_unavailable'
        return empty

    loss_indices = selected_indices[loss_mask]
    loss_labels = selected_labels[loss_mask]
    normalized_features = F.normalize(features[loss_indices].float(), dim=1)
    normalized_bank = F.normalize(prototype_state['bank'].float(), dim=1)
    logits = torch.mm(normalized_features, normalized_bank.t()) / float(temperature)
    logits = logits.masked_fill(~valid_proto[None, :], float('-inf'))
    losses = F.cross_entropy(logits, loss_labels, reduction='none')
    probs = F.softmax(logits, dim=1)
    target_probs = probs.gather(1, loss_labels[:, None]).squeeze(1)
    target_logits = logits.gather(1, loss_labels[:, None]).squeeze(1)
    non_target_probs = probs.clone()
    non_target_probs.scatter_(1, loss_labels[:, None], 0.0)
    max_non_target_probs = non_target_probs.max(dim=1)[0]
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    nearest_proto = logits.argmax(dim=1)
    nearest_acc = (nearest_proto == loss_labels).float().mean()

    return {
        'loss': losses.mean(),
        'num_proto_loss_valid': int(loss_labels.numel()),
        'proto_ce_raw': float(losses.mean().detach().item()),
        'proto_target_prob_mean': _safe_mean(target_probs.detach()),
        'proto_target_logit_mean': _safe_mean(target_logits.detach()),
        'proto_max_non_target_prob_mean': _safe_mean(max_non_target_probs.detach()),
        'proto_entropy_mean': _safe_mean(entropy.detach()),
        'nearest_proto_acc': float(nearest_acc.detach().item()),
        'target_proto_available_ratio': _safe_mean(target_available.float()),
        'skip_reason': 'none',
    }


def build_evidence_proto_log_row(epoch, batch_idx, num_clean, num_valid_part, update_info,
                                 update_result, loss_result, prototype_state, proto_weight,
                                 proto_temp, base_loss):
    update_info = update_info or {}
    update_result = update_result or {}
    loss_result = loss_result or {}
    proto_ce_raw = float(loss_result.get('proto_ce_raw', 0.0))
    proto_ce_weighted = float(proto_weight) * proto_ce_raw
    base_loss_value = _zero_float(base_loss.detach()) if torch.is_tensor(base_loss) else float(base_loss or 0.0)
    proto_loss_ratio = proto_ce_weighted / max(base_loss_value + proto_ce_weighted, 1e-12)
    num_initialized, initialized_ratio = _prototype_initialized_stats(prototype_state)
    count_mean, count_min, count_max = _update_count_stats(prototype_state)
    num_proto_update = int(update_result.get('num_proto_update', 0))
    num_proto_loss_valid = int(loss_result.get('num_proto_loss_valid', 0))
    skip_reasons = [
        str(update_info.get('skip_reason', 'none')),
        str(loss_result.get('skip_reason', 'none')),
    ]
    skip_reason = '|'.join([reason for reason in skip_reasons if reason not in ['', 'none']]) or 'none'

    return {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'num_clean': int(num_clean),
        'num_valid_part': int(num_valid_part),
        'num_valid_evidence': int(update_info.get('num_valid_evidence', num_valid_part)),
        'num_update_candidate': int(update_info.get('num_update_candidate', 0)),
        'num_proto_update': num_proto_update,
        'num_proto_loss_valid': num_proto_loss_valid,
        'proto_update_ratio': float(num_proto_update / max(int(num_clean), 1)),
        'proto_loss_valid_ratio': float(num_proto_loss_valid / max(int(num_clean), 1)),
        'proto_ce_raw': proto_ce_raw,
        'proto_ce_weighted': proto_ce_weighted,
        'proto_temp': float(proto_temp),
        'proto_loss_ratio': proto_loss_ratio,
        'proto_target_prob_mean': float(loss_result.get('proto_target_prob_mean', 0.0)),
        'proto_target_logit_mean': float(loss_result.get('proto_target_logit_mean', 0.0)),
        'proto_max_non_target_prob_mean': float(loss_result.get('proto_max_non_target_prob_mean', 0.0)),
        'proto_entropy_mean': float(loss_result.get('proto_entropy_mean', 0.0)),
        'nearest_proto_acc': float(loss_result.get('nearest_proto_acc', 0.0)),
        'target_proto_available_ratio': float(loss_result.get('target_proto_available_ratio', 0.0)),
        'prototype_num_initialized': int(num_initialized),
        'prototype_initialized_ratio': float(initialized_ratio),
        'prototype_update_count_mean': float(count_mean),
        'prototype_update_count_min': float(count_min),
        'prototype_update_count_max': float(count_max),
        'update_weight_mean': float(update_info.get('update_weight_mean', 0.0)),
        'update_weight_std': float(update_info.get('update_weight_std', 0.0)),
        'update_weight_min': float(update_info.get('update_weight_min', 0.0)),
        'update_weight_max': float(update_info.get('update_weight_max', 0.0)),
        'update_effective_sample_size': float(update_info.get('update_effective_sample_size', 0.0)),
        'evidence_rank_mean': float(update_info.get('evidence_rank_mean', 0.0)),
        'evidence_rank_std': float(update_info.get('evidence_rank_std', 0.0)),
        'update_evidence_score_mean': float(update_info.get('update_evidence_score_mean', 0.0)),
        'update_p_ori_y_mean': float(update_info.get('update_p_ori_y_mean', 0.0)),
        'prototype_drift_mean': float(update_result.get('prototype_drift_mean', 0.0)),
        'prototype_drift_max': float(update_result.get('prototype_drift_max', 0.0)),
        'skip_reason': skip_reason,
    }


def format_evidence_proto_row(row):
    return (
        f"{row['epoch']},{row['batch_idx']},{row['num_clean']},"
        f"{row['num_valid_part']},{row['num_valid_evidence']},"
        f"{row['num_update_candidate']},{row['num_proto_update']},"
        f"{row['num_proto_loss_valid']},{row['proto_update_ratio']:.6f},"
        f"{row['proto_loss_valid_ratio']:.6f},{row['proto_ce_raw']:.6f},"
        f"{row['proto_ce_weighted']:.6f},{row['proto_temp']:.6f},"
        f"{row['proto_loss_ratio']:.6f},{row['proto_target_prob_mean']:.6f},"
        f"{row['proto_target_logit_mean']:.6f},"
        f"{row['proto_max_non_target_prob_mean']:.6f},"
        f"{row['proto_entropy_mean']:.6f},{row['nearest_proto_acc']:.6f},"
        f"{row['target_proto_available_ratio']:.6f},"
        f"{row['prototype_num_initialized']},"
        f"{row['prototype_initialized_ratio']:.6f},"
        f"{row['prototype_update_count_mean']:.6f},"
        f"{row['prototype_update_count_min']:.6f},"
        f"{row['prototype_update_count_max']:.6f},"
        f"{row['update_weight_mean']:.6f},{row['update_weight_std']:.6f},"
        f"{row['update_weight_min']:.6f},{row['update_weight_max']:.6f},"
        f"{row['update_effective_sample_size']:.6f},"
        f"{row['evidence_rank_mean']:.6f},{row['evidence_rank_std']:.6f},"
        f"{row['update_evidence_score_mean']:.6f},"
        f"{row['update_p_ori_y_mean']:.6f},"
        f"{row['prototype_drift_mean']:.6f},{row['prototype_drift_max']:.6f},"
        f"{row['skip_reason']}"
    )
