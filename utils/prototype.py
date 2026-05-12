# -*- coding: utf-8 -*-
import math

import torch
import torch.nn.functional as F


EVIDENCE_PROTO_CSV_HEADER = (
    'epoch,batch_idx,num_clean,num_valid_part,num_valid_evidence,'
    'num_update_candidate,num_proto_update,num_proto_loss_valid,'
    'proto_update_ratio,proto_loss_valid_ratio,proto_ce_raw,'
    'proto_ce_weighted,proto_temp,proto_loss_ratio,'
    'proto_target_prob_mean,proto_target_logit_mean,'
    'proto_max_non_target_prob_mean,proto_entropy_mean,nearest_proto_acc,'
    'nearest_class_acc,target_proto_available_ratio,prototype_num_initialized,'
    'prototype_initialized_ratio,prototype_update_count_mean,'
    'prototype_update_count_min,prototype_update_count_max,'
    'subproto_num_valid,subproto_valid_ratio,subproto_update_count_mean,'
    'subproto_update_count_min,subproto_update_count_max,subproto_dead_count,'
    'subproto_dead_ratio,subproto_intra_class_cos_mean,'
    'subproto_intra_class_cos_std,subproto_intra_class_dist_mean,'
    'subproto_assign_entropy_mean,subproto_assign_entropy_min,'
    'subproto_assign_top1_ratio_mean,target_subproto_similarity_mean,'
    'target_subproto_margin_mean,update_weight_mean,update_weight_std,'
    'update_weight_min,update_weight_max,update_effective_sample_size,'
    'evidence_rank_mean,evidence_rank_std,update_evidence_score_mean,'
    'update_p_ori_y_mean,prototype_drift_mean,prototype_drift_max,skip_reason'
)


def init_prototype_state(num_classes, feature_dim, device, num_subproto=1):
    # prototype bank 是训练状态缓冲区，不作为可学习参数参与 optimizer；K=1 兼容旧 single-prototype。
    num_subproto = int(num_subproto)
    if num_subproto < 1:
        raise ValueError(f'num_subproto should be >= 1, got {num_subproto}.')
    return {
        'bank': torch.zeros(num_classes, num_subproto, feature_dim, device=device),
        'count': torch.zeros(num_classes, num_subproto, device=device),
        'valid': torch.zeros(num_classes, num_subproto, device=device, dtype=torch.bool),
    }


def _load_state_tensor(prototype_state, key, checkpoint_key, checkpoint, device):
    target = prototype_state[key]
    if checkpoint_key not in checkpoint:
        return

    value = checkpoint[checkpoint_key].to(device)
    if key in ['count', 'valid'] and value.dim() == 1 and target.dim() == 2 and target.size(1) == 1:
        # 旧 checkpoint 的 [C] 形状只允许恢复到 K=1，避免 K=2 隐式复制造成实验污染。
        value = value.unsqueeze(1)
    if key == 'bank' and value.dim() == 2 and target.dim() == 3 and target.size(1) == 1:
        # 旧 checkpoint 的 [C,D] 形状只允许恢复到 K=1。
        value = value.unsqueeze(1)
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f'{checkpoint_key} shape mismatch: checkpoint {tuple(value.shape)} vs current {tuple(target.shape)}.'
        )
    target.copy_(value.to(dtype=target.dtype))


def load_prototype_state(prototype_state, checkpoint, device):
    if prototype_state is None:
        return
    _load_state_tensor(prototype_state, 'bank', 'prototype_bank', checkpoint, device)
    _load_state_tensor(prototype_state, 'count', 'prototype_count', checkpoint, device)
    _load_state_tensor(prototype_state, 'valid', 'prototype_valid', checkpoint, device)


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


def _subproto_intra_class_stats(prototype_state):
    bank = prototype_state['bank']
    valid = prototype_state['valid']
    cos_values = []
    with torch.no_grad():
        norm_bank = F.normalize(bank.float(), dim=2)
        for class_idx in range(norm_bank.size(0)):
            class_valid = valid[class_idx]
            if int(class_valid.sum().item()) < 2:
                continue
            class_proto = norm_bank[class_idx, class_valid]
            pair_cos = torch.mm(class_proto, class_proto.t())
            pair_mask = torch.triu(
                torch.ones_like(pair_cos, dtype=torch.bool),
                diagonal=1,
            )
            cos_values.append(pair_cos[pair_mask])
    if len(cos_values) == 0:
        return 0.0, 0.0, 0.0
    cos_values = torch.cat(cos_values)
    cos_mean = _safe_mean(cos_values)
    return cos_mean, _safe_std(cos_values), float(1.0 - cos_mean)


def _subproto_assign_stats(prototype_state, eps=1e-12):
    counts = prototype_state['count'].float()
    num_subproto = counts.size(1)
    totals = counts.sum(dim=1)
    active = totals > 0
    if int(active.sum().item()) == 0:
        return 0.0, 0.0, 0.0

    probs = counts[active] / totals[active, None].clamp_min(eps)
    if num_subproto <= 1:
        entropy = probs.new_zeros((probs.size(0),))
    else:
        # 分配熵按类别内部 K 个中心归一化，便于 K=1/K=2 诊断对比。
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1) / math.log(num_subproto)
    top1_ratio = probs.max(dim=1)[0]
    return _safe_mean(entropy), float(entropy.min().item()), _safe_mean(top1_ratio)


def _subproto_stats(prototype_state):
    valid = prototype_state['valid']
    count_mean, count_min, count_max = _update_count_stats(prototype_state)
    dead = (~valid) | (prototype_state['count'] <= 0)
    cos_mean, cos_std, dist_mean = _subproto_intra_class_stats(prototype_state)
    assign_entropy_mean, assign_entropy_min, assign_top1_ratio_mean = _subproto_assign_stats(prototype_state)
    return {
        'subproto_num_valid': int(valid.sum().item()),
        'subproto_valid_ratio': float(valid.float().mean().item()) if valid.numel() > 0 else 0.0,
        'subproto_update_count_mean': count_mean,
        'subproto_update_count_min': count_min,
        'subproto_update_count_max': count_max,
        'subproto_dead_count': int(dead.sum().item()),
        'subproto_dead_ratio': float(dead.float().mean().item()) if dead.numel() > 0 else 0.0,
        'subproto_intra_class_cos_mean': cos_mean,
        'subproto_intra_class_cos_std': cos_std,
        'subproto_intra_class_dist_mean': dist_mean,
        'subproto_assign_entropy_mean': assign_entropy_mean,
        'subproto_assign_entropy_min': assign_entropy_min,
        'subproto_assign_top1_ratio_mean': assign_top1_ratio_mean,
    }


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
        # rank 只在 update candidates 内计算，避免低置信 clean 样本改变有效候选的相对权重。
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


def _assign_subprototypes(bank, valid, features, labels, update_assign, init_policy):
    if update_assign != 'nearest_same_class':
        raise ValueError(f'unsupported proto update_assign: {update_assign}.')
    if init_policy != 'fill_empty_first':
        raise ValueError(f'unsupported proto init_policy: {init_policy}.')

    # 分配阶段使用临时 bank，保证同一个 batch 内新初始化的中心能参与后续样本的最近中心选择。
    assign_bank = bank.detach().clone()
    assign_valid = valid.detach().clone()
    sub_indices = []
    for feature, label in zip(features, labels):
        class_idx = int(label.item())
        class_valid = assign_valid[class_idx]
        empty_positions = torch.nonzero(~class_valid, as_tuple=False).flatten()
        if empty_positions.numel() > 0:
            sub_idx = int(empty_positions[0].item())
            assign_bank[class_idx, sub_idx] = feature
            assign_valid[class_idx, sub_idx] = True
        else:
            class_bank = F.normalize(assign_bank[class_idx].float(), dim=1)
            sims = torch.mv(class_bank, feature.float())
            sub_idx = int(sims.argmax().item())
        sub_indices.append(sub_idx)
    return torch.tensor(sub_indices, device=features.device, dtype=torch.long)


def update_prototype_bank_weighted(
    prototype_state,
    features,
    labels,
    weights,
    momentum=0.90,
    update_assign='nearest_same_class',
    init_policy='fill_empty_first',
    eps=1e-12,
):
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
        sub_indices = _assign_subprototypes(bank, valid, features, labels, update_assign, init_policy)
        num_updated_samples = int(labels.numel())
        drifts = []

        # 按 (class, sub-prototype) 聚合，K=1 时退化为旧版每类一个 weighted mean。
        pair_keys = labels * bank.size(1) + sub_indices
        for pair_key in pair_keys.unique(sorted=True):
            pair_mask = pair_keys == pair_key
            class_idx = int((pair_key // bank.size(1)).item())
            sub_idx = int((pair_key % bank.size(1)).item())
            pair_weights = weights[pair_mask]
            weight_sum = pair_weights.sum()
            if float(weight_sum.item()) <= eps:
                continue

            pair_features = features[pair_mask]
            weighted_mean = (pair_features * pair_weights[:, None]).sum(dim=0) / (weight_sum + eps)
            weighted_mean = F.normalize(weighted_mean, dim=0)
            was_valid = bool(valid[class_idx, sub_idx].item())
            if was_valid:
                before = bank[class_idx, sub_idx].detach().clone()
                bank[class_idx, sub_idx] = F.normalize(
                    float(momentum) * bank[class_idx, sub_idx] + (1.0 - float(momentum)) * weighted_mean,
                    dim=0,
                )
                drift = 1.0 - torch.dot(before, bank[class_idx, sub_idx]).clamp(min=-1.0, max=1.0)
                drifts.append(float(drift.item()))
            else:
                bank[class_idx, sub_idx] = weighted_mean
                valid[class_idx, sub_idx] = True
            count[class_idx, sub_idx] += float(pair_mask.sum().item())

        if len(drifts) == 0:
            drift_mean, drift_max = 0.0, 0.0
        else:
            drift_mean, drift_max = float(sum(drifts) / len(drifts)), float(max(drifts))
        return {
            'num_proto_update': num_updated_samples,
            'prototype_drift_mean': drift_mean,
            'prototype_drift_max': drift_max,
        }


def compute_prototype_softmax_loss(
    features,
    labels,
    selected_indices,
    prototype_state,
    temperature=0.2,
    class_logit_pool='max',
):
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
        'nearest_class_acc': 0.0,
        'target_proto_available_ratio': 0.0,
        'target_subproto_similarity_mean': 0.0,
        'target_subproto_margin_mean': 0.0,
        'skip_reason': 'not_run',
    }
    if class_logit_pool != 'max':
        raise ValueError(f'unsupported proto class_logit_pool: {class_logit_pool}.')
    if selected_indices.numel() == 0:
        empty['skip_reason'] = 'no_clean'
        return empty

    valid_subproto = prototype_state['valid'].to(device=device)
    valid_class = valid_subproto.any(dim=1)
    num_valid_class = int(valid_class.sum().item())
    selected_labels = labels[selected_indices].detach().long()
    target_available = valid_class[selected_labels]
    empty['target_proto_available_ratio'] = _safe_mean(target_available.float())
    if num_valid_class < 2:
        empty['skip_reason'] = 'prototype_valid_classes_lt_2'
        return empty

    loss_mask = target_available
    if int(loss_mask.sum().item()) == 0:
        empty['skip_reason'] = 'target_proto_unavailable'
        return empty

    loss_indices = selected_indices[loss_mask]
    loss_labels = selected_labels[loss_mask]
    normalized_features = F.normalize(features[loss_indices].float(), dim=1)
    normalized_bank = F.normalize(prototype_state['bank'].float(), dim=2)
    subproto_sim = torch.einsum('nd,ckd->nck', normalized_features, normalized_bank)
    subproto_sim = subproto_sim.masked_fill(~valid_subproto[None, :, :], float('-inf'))
    class_sim = subproto_sim.max(dim=2)[0]
    logits = class_sim / float(temperature)
    losses = F.cross_entropy(logits, loss_labels, reduction='none')
    probs = F.softmax(logits, dim=1)
    target_probs = probs.gather(1, loss_labels[:, None]).squeeze(1)
    target_logits = logits.gather(1, loss_labels[:, None]).squeeze(1)
    target_similarity = class_sim.gather(1, loss_labels[:, None]).squeeze(1)

    non_target_probs = probs.clone()
    non_target_probs.scatter_(1, loss_labels[:, None], 0.0)
    max_non_target_probs = non_target_probs.max(dim=1)[0]
    non_target_sim = class_sim.clone()
    non_target_sim.scatter_(1, loss_labels[:, None], float('-inf'))
    target_margin = target_similarity - non_target_sim.max(dim=1)[0]
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)

    nearest_class = logits.argmax(dim=1)
    nearest_class_acc = (nearest_class == loss_labels).float().mean()
    flat_proto = subproto_sim.reshape(subproto_sim.size(0), -1).argmax(dim=1)
    nearest_proto_class = torch.div(flat_proto, subproto_sim.size(2), rounding_mode='floor')
    nearest_proto_acc = (nearest_proto_class == loss_labels).float().mean()

    return {
        'loss': losses.mean(),
        'num_proto_loss_valid': int(loss_labels.numel()),
        'proto_ce_raw': float(losses.mean().detach().item()),
        'proto_target_prob_mean': _safe_mean(target_probs.detach()),
        'proto_target_logit_mean': _safe_mean(target_logits.detach()),
        'proto_max_non_target_prob_mean': _safe_mean(max_non_target_probs.detach()),
        'proto_entropy_mean': _safe_mean(entropy.detach()),
        'nearest_proto_acc': float(nearest_proto_acc.detach().item()),
        'nearest_class_acc': float(nearest_class_acc.detach().item()),
        'target_proto_available_ratio': _safe_mean(target_available.float()),
        'target_subproto_similarity_mean': _safe_mean(target_similarity.detach()),
        'target_subproto_margin_mean': _safe_mean(target_margin.detach()),
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
    subproto_stats = _subproto_stats(prototype_state)
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
        'nearest_class_acc': float(loss_result.get('nearest_class_acc', 0.0)),
        'target_proto_available_ratio': float(loss_result.get('target_proto_available_ratio', 0.0)),
        'prototype_num_initialized': int(num_initialized),
        'prototype_initialized_ratio': float(initialized_ratio),
        'prototype_update_count_mean': float(count_mean),
        'prototype_update_count_min': float(count_min),
        'prototype_update_count_max': float(count_max),
        'subproto_num_valid': int(subproto_stats['subproto_num_valid']),
        'subproto_valid_ratio': float(subproto_stats['subproto_valid_ratio']),
        'subproto_update_count_mean': float(subproto_stats['subproto_update_count_mean']),
        'subproto_update_count_min': float(subproto_stats['subproto_update_count_min']),
        'subproto_update_count_max': float(subproto_stats['subproto_update_count_max']),
        'subproto_dead_count': int(subproto_stats['subproto_dead_count']),
        'subproto_dead_ratio': float(subproto_stats['subproto_dead_ratio']),
        'subproto_intra_class_cos_mean': float(subproto_stats['subproto_intra_class_cos_mean']),
        'subproto_intra_class_cos_std': float(subproto_stats['subproto_intra_class_cos_std']),
        'subproto_intra_class_dist_mean': float(subproto_stats['subproto_intra_class_dist_mean']),
        'subproto_assign_entropy_mean': float(subproto_stats['subproto_assign_entropy_mean']),
        'subproto_assign_entropy_min': float(subproto_stats['subproto_assign_entropy_min']),
        'subproto_assign_top1_ratio_mean': float(subproto_stats['subproto_assign_top1_ratio_mean']),
        'target_subproto_similarity_mean': float(loss_result.get('target_subproto_similarity_mean', 0.0)),
        'target_subproto_margin_mean': float(loss_result.get('target_subproto_margin_mean', 0.0)),
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
        f"{row['nearest_class_acc']:.6f},"
        f"{row['target_proto_available_ratio']:.6f},"
        f"{row['prototype_num_initialized']},"
        f"{row['prototype_initialized_ratio']:.6f},"
        f"{row['prototype_update_count_mean']:.6f},"
        f"{row['prototype_update_count_min']:.6f},"
        f"{row['prototype_update_count_max']:.6f},"
        f"{row['subproto_num_valid']},"
        f"{row['subproto_valid_ratio']:.6f},"
        f"{row['subproto_update_count_mean']:.6f},"
        f"{row['subproto_update_count_min']:.6f},"
        f"{row['subproto_update_count_max']:.6f},"
        f"{row['subproto_dead_count']},"
        f"{row['subproto_dead_ratio']:.6f},"
        f"{row['subproto_intra_class_cos_mean']:.6f},"
        f"{row['subproto_intra_class_cos_std']:.6f},"
        f"{row['subproto_intra_class_dist_mean']:.6f},"
        f"{row['subproto_assign_entropy_mean']:.6f},"
        f"{row['subproto_assign_entropy_min']:.6f},"
        f"{row['subproto_assign_top1_ratio_mean']:.6f},"
        f"{row['target_subproto_similarity_mean']:.6f},"
        f"{row['target_subproto_margin_mean']:.6f},"
        f"{row['update_weight_mean']:.6f},{row['update_weight_std']:.6f},"
        f"{row['update_weight_min']:.6f},{row['update_weight_max']:.6f},"
        f"{row['update_effective_sample_size']:.6f},"
        f"{row['evidence_rank_mean']:.6f},{row['evidence_rank_std']:.6f},"
        f"{row['update_evidence_score_mean']:.6f},"
        f"{row['update_p_ori_y_mean']:.6f},"
        f"{row['prototype_drift_mean']:.6f},{row['prototype_drift_max']:.6f},"
        f"{row['skip_reason']}"
    )
