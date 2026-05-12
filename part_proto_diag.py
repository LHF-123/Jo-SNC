# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import warnings
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.builder import (
    build_animal10n_dataset,
    build_cifar100n_dataset,
    build_food101n_dataset,
    build_mini_webvision_dataset,
    build_transform,
    build_webfg_dataset,
)
from utils.local_evidence import build_local_part_batch
from utils.model import DualHeadModel
from utils.utils import MultiDataTransform, get_smoothed_label_distribution, js_div, set_seed


SUMMARY_FIELDS = [
    'checkpoint', 'num_total', 'num_clean', 'num_valid_part', 'num_update_candidate',
    'josnc_clean_ratio', 'josnc_id_ratio', 'josnc_ood_ratio',
    'expected_clean_ratio_from_log', 'clean_ratio_abs_diff', 'clean_selection_trust_flag',
    'num_eval_total', 'num_eval_with_target_part_proto', 'num_eval_without_target_part_proto',
    'num_eval_used_for_part_proto', 'part_proto_target_available_ratio',
    'normal_part_proto_valid_classes', 'eapa_part_proto_valid_classes',
    'part_proto_valid_class_ratio', 'global_proto_valid_class_ratio',
    'cls_global_acc',
    'normal_part_proto_acc', 'eapa_part_proto_acc',
    'normal_global_proto_acc', 'eapa_global_proto_acc',
    'normal_part_target_sim', 'eapa_part_target_sim',
    'normal_part_margin', 'eapa_part_margin',
    'normal_part_entropy', 'eapa_part_entropy',
    'normal_part_target_prob', 'eapa_part_target_prob',
    'high_evidence_part_acc', 'mid_evidence_part_acc', 'low_evidence_part_acc',
    'high_evidence_part_margin', 'mid_evidence_part_margin', 'low_evidence_part_margin',
    'cls_wrong_eapa_part_correct', 'cls_correct_eapa_part_wrong',
    'global_proto_wrong_eapa_part_correct', 'global_proto_correct_eapa_part_wrong',
    'part_global_agree_ratio', 'part_correct_global_proto_wrong_ratio',
    'confused_pair_source', 'top_confused_pairs',
    'confused_pair_recover_rate', 'confused_pair_margin_mean',
    'local_e_mean', 'local_e_std', 'local_e_p25', 'local_e_p50', 'local_e_p75',
    'local_e_high_mean', 'local_e_low_mean',
    'corr_local_e_pori',
    'corr_local_e_normal_part_margin', 'corr_local_e_eapa_part_margin',
]

SAMPLE_FIELDS = [
    'sample_index', 'label', 'cls_global_pred',
    'normal_part_pred', 'eapa_part_pred',
    'normal_global_proto_pred', 'eapa_global_proto_pred',
    'evidence_rank_group',
    'p_ori_y', 'p_part_y', 'p_erase_y', 'local_e',
    'normal_part_margin', 'eapa_part_margin',
    'normal_part_target_sim', 'eapa_part_target_sim',
]

REQUIRED_CKPT_FIELDS = [
    'model_state_dict', 'k_model_state_dict', 'queue_keys', 'queue_logits', 'tau_c', 'tau_o',
]


class NullLogger:
    def debug(self, *_args, **_kwargs):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Part-level prototype diagnostic for Jo-SNC full checkpoints.')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output', type=str, default='results/part_proto_diag')
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--proto-build-ratio', type=float, default=0.7)
    parser.add_argument('--proto-conf-thr', type=float, default=None)
    parser.add_argument('--proto-temp', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=1)
    parser.add_argument('--top-confused-pairs', type=int, default=20)
    parser.add_argument('--clean-ratio-trust-tol', type=float, default=0.02)
    return parser.parse_args()


def load_config(cfg_path, data_root=None):
    # 诊断脚本只补齐本脚本会读取的训练配置项，避免依赖 main.py 的命令行解析副作用。
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if data_root is not None:
        cfg['data_root'] = data_root
    cfg.setdefault('seed', 0)
    cfg.setdefault('transform', 'strong')
    cfg.setdefault('eps', 0.3)
    cfg.setdefault('n_neighbors', 10)
    cfg.setdefault('integrate_mode', 'or')
    cfg.setdefault('ood_criterion', 'div')
    cfg.setdefault('fdim', 256)
    cfg.setdefault('hdim', 2)
    cfg.setdefault('batch_size', 64)
    cfg.setdefault('noise_type', 'clean')
    cfg.setdefault('idn_noise_rate', 0.0)
    cfg.setdefault('ood_noise_rate', 0.0)
    cfg.setdefault('local_evidence_cam_quantile', 0.8)
    cfg.setdefault('local_evidence_min_area', 0.05)
    cfg.setdefault('local_evidence_max_area', 0.7)
    cfg.setdefault('local_evidence_bbox_padding', 0.05)
    cfg.setdefault('local_evidence_cam_type', 'weightcam')
    cfg.setdefault('part_ce_use_teacher_cam', True)
    cfg.setdefault('evidence_proto_num_subproto', 1)
    cfg.setdefault('normal_proto_num_subproto', 1)
    return SimpleNamespace(**cfg)


def build_train_dataset(cfg):
    # 和 main.py 的训练集构建保持一致，保留双视图 transform 以复现 Jo-SNC selection 输入形式。
    transform = build_transform(cfg.rescale_size, cfg.crop_size, dataset=cfg.dataset)
    if cfg.dataset.startswith('cifar100n'):
        assert cfg.ood_noise_rate == 0.0, 'ood_noise_rate should be 0.0 in cifar100n-* datasets'
        assert cfg.n_classes == 100, 'number of classes should be 100'
        transform_type = 'train_moco' if cfg.transform == 'moco' else (
            'cifar_train_strong_aug' if cfg.transform == 'strong' else 'cifar_train'
        )
        return build_cifar100n_dataset(
            os.path.join(cfg.data_root, 'cifar100'),
            MultiDataTransform([transform['cifar_train'], transform[transform_type]]),
            transform['cifar_test'],
            cfg.noise_type,
            0.0,
            cfg.idn_noise_rate,
        )
    if cfg.dataset.startswith('cifar80n'):
        assert cfg.ood_noise_rate == 0.2, 'ood_noise_rate should be 0.2 in cifar80n-* datasets'
        assert cfg.n_classes == 80, 'number of classes should be 80'
        transform_type = 'train_moco' if cfg.transform == 'moco' else (
            'cifar_train_strong_aug' if cfg.transform == 'strong' else 'cifar_train'
        )
        return build_cifar100n_dataset(
            os.path.join(cfg.data_root, 'cifar100'),
            MultiDataTransform([transform['cifar_train'], transform[transform_type]]),
            transform['cifar_test'],
            cfg.noise_type,
            0.2,
            cfg.idn_noise_rate,
        )
    if cfg.dataset == 'animal10n':
        transform_type = 'cifar_train_strong_aug' if cfg.transform == 'strong' else 'cifar_train'
        return build_animal10n_dataset(
            os.path.join(cfg.data_root, cfg.dataset),
            MultiDataTransform([transform['cifar_train'], transform[transform_type]]),
            transform['cifar_test'],
        )
    if cfg.dataset in ['web-aircraft', 'web-bird', 'web-car']:
        transform_type = 'train' if cfg.transform == 'weak' else 'train_strong_aug'
        return build_webfg_dataset(
            os.path.join(cfg.data_root, cfg.dataset),
            MultiDataTransform([transform['train'], transform[transform_type]]),
            transform['test'],
        )
    if cfg.dataset == 'food101n':
        transform_type = 'train' if cfg.transform == 'weak' else 'train_strong_aug'
        return build_food101n_dataset(
            os.path.join(cfg.data_root, cfg.dataset),
            MultiDataTransform([transform['train'], transform[transform_type]]),
            transform['test'],
        )
    if cfg.dataset in ['mini-webvision', 'webvision']:
        transform_type = 'train' if cfg.transform == 'weak' else 'train_strong_aug'
        return build_mini_webvision_dataset(
            os.path.join(cfg.data_root, cfg.dataset),
            MultiDataTransform([transform['train'], transform[transform_type]]),
            transform['test'],
            num_class=cfg.n_classes,
        )
    raise NotImplementedError(f'{cfg.dataset} is not supported.')


def run_samples_identification(
        logits1, logits2, ob_labels, features, features_queue, logits_queue,
        threshold_clean, threshold_ood, config, logger):
    # 逐行复现 main.py 的 Jo-SNC clean/ID/OOD selection，不提供 confidence fallback。
    with torch.no_grad():
        probs1, probs2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
        prob_clean = 1 - js_div(probs1, ob_labels)
        cleanness_self = torch.ge(prob_clean, threshold_clean)
        similarity = torch.mm(features, features_queue.t())
        _, neighbor_indices = similarity.topk(config.n_neighbors + 1, dim=1, largest=True, sorted=True)
        neighbor_indices = neighbor_indices[:, 1:].contiguous().view(-1)
        neighbor_probs = logits_queue[neighbor_indices].softmax(dim=1)
        neighbor_ob_labels = ob_labels.repeat(1, config.n_neighbors).view(-1, config.n_classes)
        neighbor_prob_clean = 1 - js_div(neighbor_probs, neighbor_ob_labels).view(-1, config.n_neighbors).mean(dim=1)
        cleanness_neighbor = torch.gt(neighbor_prob_clean, threshold_clean)
        if config.integrate_mode == 'or':
            clean = torch.logical_or(cleanness_self, cleanness_neighbor)
        elif config.integrate_mode == 'and':
            clean = torch.logical_and(cleanness_self, cleanness_neighbor)
        elif config.integrate_mode == 'self-only':
            clean = cleanness_self
        elif config.integrate_mode == 'neighbor-only':
            clean = cleanness_neighbor
        else:
            raise AssertionError(
                'integrate_mode should be within [and, or, self-only, neighbor-only], '
                f'the current value is {config.integrate_mode}'
            )
        unclean = clean.logical_not()
        idx_clean = clean.nonzero(as_tuple=False).squeeze(dim=1)

        prob_ood = js_div(F.softmax(logits1 / 0.1, dim=1), F.softmax(logits2 / 0.1, dim=1))
        pred1, pred2 = probs1.argmax(dim=1), probs2.argmax(dim=1)
        if config.ood_criterion.startswith('div'):
            disagree = prob_ood > threshold_ood
            agree = prob_ood <= threshold_ood
        elif config.ood_criterion.startswith('dis'):
            disagree = pred1 != pred2
            agree = pred1 == pred2
        else:
            raise AssertionError(
                'ood_criterion should be within [div, dis], '
                f'the current value is {config.ood_criterion}'
            )
        idx_ood = (disagree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
        idx_id = (agree * unclean).nonzero(as_tuple=False).squeeze(dim=1)

    logger.debug('selection finished')
    return idx_clean, idx_id, idx_ood, prob_clean, prob_ood


def resolve_device(gpu):
    # 支持 --gpu cpu，便于在没有 CUDA 的机器上先跑输入校验和小规模 smoke。
    gpu_text = str(gpu).strip().lower()
    if gpu_text == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(f'cuda:{gpu_text}')


def build_model(cfg, device):
    # 只构建和训练一致的双头模型；pretrained=False 保证不会触发额外下载。
    return DualHeadModel(
        arch=cfg.arch,
        num_classes=cfg.n_classes,
        mlp_hidden=cfg.hdim,
        feature_dim=cfg.fdim,
        pretrained=False,
        use_bn=True,
    ).to(device)


def strip_module_prefix(state_dict):
    if not any(str(key).startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def load_full_checkpoint(ckpt_path, q_model, k_model, device):
    # 诊断必须复现 Jo-SNC selection，缺少队列或阈值状态时直接报错。
    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError('checkpoint must be a dict full checkpoint, got non-dict object.')
    missing = [field for field in REQUIRED_CKPT_FIELDS if field not in checkpoint]
    if missing:
        raise ValueError(
            '--ckpt-path must be a full Jo-SNC checkpoint. '
            f'Missing fields: {", ".join(missing)}. '
            'Plain model weights such as model_best.pth cannot reproduce samples_identification().'
        )

    q_model.load_state_dict(strip_module_prefix(checkpoint['model_state_dict']), strict=True)
    k_model.load_state_dict(strip_module_prefix(checkpoint['k_model_state_dict']), strict=True)
    # 训练端 selection 在 q_model.train()/k_model.train() 下静态前向；这里保持 BN 行为一致。
    q_model.train()
    k_model.train()

    queue_keys = checkpoint['queue_keys'].to(device).float()
    queue_logits = checkpoint['queue_logits'].to(device).float()
    tau_c = checkpoint['tau_c'].to(device).float()
    tau_o = checkpoint['tau_o'].to(device).float()
    return checkpoint, queue_keys, queue_logits, tau_c, tau_o


def warn_if_k2_checkpoint(cfg, checkpoint):
    # 第一版只做 single prototype 诊断；检测到 K=2 训练痕迹时提示不要做归因结论。
    proto_bank = checkpoint.get('prototype_bank')
    checkpoint_k = int(proto_bank.shape[1]) if torch.is_tensor(proto_bank) and proto_bank.dim() == 3 else 1
    config_k = max(int(getattr(cfg, 'evidence_proto_num_subproto', 1)), int(getattr(cfg, 'normal_proto_num_subproto', 1)))
    if checkpoint_k > 1 or config_k > 1:
        warnings.warn(
            'Detected K=2 prototype config/checkpoint state. '
            'part_proto_diag.py still runs single part prototype diagnostics, '
            'but this checkpoint is not recommended for attribution diagnosis.',
            RuntimeWarning,
        )


def resolve_proto_conf_thr(cfg, cli_value):
    # 默认按当前 active prototype 分支读取 update candidate 门槛；C1 checkpoint 无 prototype 分支时回落到 0.60。
    if cli_value is not None:
        return float(cli_value)

    evidence_active = bool(getattr(cfg, 'evidence_proto_align', False))
    normal_active = bool(getattr(cfg, 'normal_proto_align', False))
    if evidence_active and normal_active:
        warnings.warn(
            'Both evidence_proto_align and normal_proto_align are enabled. '
            'Training normally forbids this; part_proto_diag.py uses evidence_proto_update_conf_thr.',
            RuntimeWarning,
        )

    evidence_thr = getattr(cfg, 'evidence_proto_update_conf_thr', None)
    normal_thr = getattr(cfg, 'normal_proto_update_conf_thr', None)

    if evidence_active and evidence_thr is not None:
        return float(evidence_thr)
    if normal_active and normal_thr is not None:
        return float(normal_thr)
    if evidence_active and evidence_thr is None:
        warnings.warn('evidence_proto_align is active but evidence_proto_update_conf_thr is missing; fallback to 0.60.', RuntimeWarning)
    if normal_active and normal_thr is None:
        warnings.warn('normal_proto_align is active but normal_proto_update_conf_thr is missing; fallback to 0.60.', RuntimeWarning)
    return 0.60


def select_cam_model(cfg, q_model, k_model):
    # C1 默认遵循 part_ce_use_teacher_cam；EAPA 训练会按 update_feature 覆盖 CAM/证据模型。
    if bool(getattr(cfg, 'evidence_proto_align', False)):
        return k_model if getattr(cfg, 'evidence_proto_update_feature', 'teacher_ema') == 'teacher_ema' else q_model
    return k_model if bool(getattr(cfg, 'part_ce_use_teacher_cam', True)) else q_model


def forward_projected_features_without_bn_update(model, images):
    # 诊断额外的 part feature 提取不属于训练 loop，临时 eval 防止污染后续静态重放的 BN running stats。
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(images)
    finally:
        model.train(was_training)
    if not isinstance(output, tuple) or len(output) < 2:
        raise ValueError('DualHeadModel is expected to return (logits, projected_features).')
    return output[0], output[1]


def tensor_to_cpu_list(tensor):
    return tensor.detach().cpu().tolist()


def extract_data_views(sample):
    data = sample['data']
    if isinstance(data, (list, tuple)):
        return data[0], data[1]
    return data, data


def safe_float(value):
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def mean_from_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return safe_float(values.mean())


def std_from_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return safe_float(values.std())


def percentile_from_values(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return safe_float(np.percentile(values, q))


def safe_corr(x_values, y_values):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return 0.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return safe_float(np.corrcoef(x, y)[0, 1])


def split_rank_norm(scores):
    # Evidence 权重的 rank 只在实际 build candidates 内计算，候选集合不变。
    scores = scores.detach().float()
    if scores.numel() == 0:
        return scores.new_zeros((0,))
    if scores.numel() == 1:
        return scores.new_ones((1,))
    order = torch.argsort(scores, descending=False)
    ranks = torch.zeros_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=scores.dtype)
    return ranks / float(scores.numel())


def deterministic_split(num_items, build_ratio, seed):
    # build/eval holdout 只由 split_seed 决定，和 DataLoader 顺序、GPU 非确定性解耦。
    build_ratio = max(0.0, min(float(build_ratio), 1.0))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_items, generator=generator)
    num_build = int(round(num_items * build_ratio))
    num_build = min(max(num_build, 0), num_items)
    build_mask = torch.zeros(num_items, dtype=torch.bool)
    build_mask[perm[:num_build]] = True
    return build_mask, ~build_mask


def build_weighted_prototypes(features, labels, weights, num_classes, eps=1e-12):
    # prototype 用加权均值离线构建，不做动量更新，避免引入训练顺序因素。
    if features.numel() == 0:
        dim = int(features.shape[1]) if features.dim() == 2 else 0
        return torch.zeros(num_classes, dim), torch.zeros(num_classes, dtype=torch.bool)
    features = F.normalize(features.float(), dim=1)
    labels = labels.long()
    weights = weights.float().clamp_min(0.0)
    sums = torch.zeros(num_classes, features.size(1), dtype=torch.float32)
    weight_sums = torch.zeros(num_classes, dtype=torch.float32)
    sums.index_add_(0, labels.cpu(), (features.cpu() * weights.cpu()[:, None]))
    weight_sums.index_add_(0, labels.cpu(), weights.cpu())
    valid = weight_sums > eps
    prototypes = torch.zeros_like(sums)
    prototypes[valid] = F.normalize(sums[valid] / weight_sums[valid, None].clamp_min(eps), dim=1)
    return prototypes, valid


def evaluate_prototypes(features, labels, prototypes, valid_classes, temp=0.2):
    # eval 时 target prototype 不存在的样本不计入 acc 和主指标分母。
    num_samples = int(labels.numel())
    empty = {
        'pred': torch.full((num_samples,), -1, dtype=torch.long),
        'target_available': torch.zeros(num_samples, dtype=torch.bool),
        'target_sim': torch.zeros(num_samples),
        'margin': torch.zeros(num_samples),
        'entropy': torch.zeros(num_samples),
        'target_prob': torch.zeros(num_samples),
        'acc': 0.0,
    }
    if num_samples == 0 or prototypes.numel() == 0 or int(valid_classes.sum().item()) == 0:
        return empty

    features = F.normalize(features.float(), dim=1).cpu()
    labels = labels.long().cpu()
    prototypes = prototypes.float().cpu()
    valid_classes = valid_classes.bool().cpu()

    sims = torch.mm(features, prototypes.t())
    sims[:, ~valid_classes] = -float('inf')
    pred = sims.argmax(dim=1)
    target_available = valid_classes[labels]
    batch_indices = torch.arange(num_samples)

    target_sim = torch.zeros(num_samples)
    target_sim[target_available] = sims[batch_indices[target_available], labels[target_available]]

    non_target = sims.clone()
    non_target[batch_indices, labels] = -float('inf')
    max_non_target = non_target.max(dim=1)[0]
    margin = torch.zeros(num_samples)
    valid_margin = target_available & torch.isfinite(max_non_target)
    margin[valid_margin] = target_sim[valid_margin] - max_non_target[valid_margin]

    logits = sims / max(float(temp), 1e-12)
    probs = torch.softmax(logits, dim=1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    target_prob = torch.zeros(num_samples)
    target_prob[target_available] = probs[batch_indices[target_available], labels[target_available]]

    if int(target_available.sum().item()) > 0:
        acc = float((pred[target_available] == labels[target_available]).float().mean().item())
    else:
        acc = 0.0

    return {
        'pred': pred,
        'target_available': target_available,
        'target_sim': target_sim,
        'margin': margin,
        'entropy': entropy,
        'target_prob': target_prob,
        'acc': acc,
    }


def evidence_groups(local_e):
    # high/mid/low 只用于 holdout 明细和分桶指标，不影响 prototype 构建。
    local_e = np.asarray(local_e, dtype=np.float64)
    groups = ['mid'] * int(local_e.size)
    if local_e.size == 0:
        return groups
    low_thr = np.percentile(local_e, 33.333)
    high_thr = np.percentile(local_e, 66.667)
    for i, value in enumerate(local_e):
        if value >= high_thr:
            groups[i] = 'high'
        elif value <= low_thr:
            groups[i] = 'low'
    return groups


def grouped_metric(groups, group_name, values, correct=None):
    mask = np.asarray([item == group_name for item in groups], dtype=bool)
    if mask.sum() == 0:
        return 0.0
    if correct is not None:
        return safe_float(np.asarray(correct, dtype=np.float64)[mask].mean())
    return safe_float(np.asarray(values, dtype=np.float64)[mask].mean())


def read_expected_clean_ratio(ckpt_path, checkpoint, num_total, max_batches):
    # 只在完整扫描时比较训练日志 clean ratio；max-batches smoke 不做信任判定。
    if max_batches and int(max_batches) > 0:
        return None
    metric_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'prfa_metric.csv')
    if not os.path.isfile(metric_path):
        return None

    target_epoch = None
    checkpoint_epoch = checkpoint.get('epoch')
    if checkpoint_epoch is not None:
        if torch.is_tensor(checkpoint_epoch):
            checkpoint_epoch = int(checkpoint_epoch.item())
        else:
            checkpoint_epoch = int(checkpoint_epoch)
        # checkpoint 内的 tau_c/tau_o 是当前 epoch 结束后更新的阈值，对应下一轮训练日志。
        target_epoch = checkpoint_epoch + 2

    rows = []
    with open(metric_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append((int(row[0]), int(row[1])))
            except ValueError:
                continue
    if not rows:
        return None
    if target_epoch is None:
        return None
    selected = None
    for row in rows:
        if row[0] == target_epoch:
            selected = row
            break
    if selected is None:
        return None
    return safe_float(selected[1] / max(int(num_total), 1))


def collect_diagnostic_records(cfg, args, q_model, k_model, queue_keys, queue_logits, tau_c, tau_o, device):
    # 一次完整前向收集 selection、局部证据、part/global prototype 特征和分类器预测。
    dataset = build_train_dataset(cfg)
    train_dataset = dataset['train']
    batch_size = int(args.batch_size or cfg.batch_size)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available() and device.type == 'cuda',
    )

    records = []
    global_features = []
    part_features = []
    clean_total = 0
    id_total = 0
    ood_total = 0
    null_logger = NullLogger()

    progress = tqdm(loader, total=len(loader), ncols=100, ascii=' >', desc='part-proto-diag')
    with torch.no_grad():
        for batch_idx, sample in enumerate(progress):
            if args.max_batches and batch_idx >= int(args.max_batches):
                break

            x1, x2 = extract_data_views(sample)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            labels = sample['label'].long().to(device, non_blocking=True)
            sample_indices = sample['index'].long().cpu()
            batch_size_actual = int(labels.size(0))
            batch_positions = torch.arange(batch_size_actual, device=device)

            logits1, feat1 = q_model(x1)
            logits2, _feat2 = q_model(x2)
            ema_logits1, ema_feat1 = k_model(x1)
            ob_labels = get_smoothed_label_distribution(labels, cfg.n_classes, cfg.eps)
            idx_clean, idx_id, idx_ood, _batch_p_clean, _batch_p_ood = run_samples_identification(
                logits1, logits2, ob_labels, feat1,
                queue_keys, queue_logits, tau_c[labels], tau_o[labels],
                cfg, null_logger,
            )

            clean_total += int(idx_clean.numel())
            id_total += int(idx_id.numel())
            ood_total += int(idx_ood.numel())
            is_clean = torch.zeros(batch_size_actual, dtype=torch.bool, device=device)
            is_id = torch.zeros(batch_size_actual, dtype=torch.bool, device=device)
            is_ood = torch.zeros(batch_size_actual, dtype=torch.bool, device=device)
            is_clean[idx_clean] = True
            is_id[idx_id] = True
            is_ood[idx_ood] = True

            cam_model = select_cam_model(cfg, q_model, k_model)
            part_batch = build_local_part_batch(
                cam_model, x1, labels, batch_positions,
                cam_quantile=cfg.local_evidence_cam_quantile,
                min_area=cfg.local_evidence_min_area,
                max_area=cfg.local_evidence_max_area,
                bbox_padding=cfg.local_evidence_bbox_padding,
                cam_type=cfg.local_evidence_cam_type,
            )
            valid_positions = part_batch['batch_indices'].detach().long()
            valid_mask = torch.zeros(batch_size_actual, dtype=torch.bool, device=device)
            valid_mask[valid_positions] = True
            valid_lookup = {int(pos.item()): i for i, pos in enumerate(valid_positions)}

            part_feat_batch = torch.zeros(batch_size_actual, cfg.fdim, dtype=torch.float32, device=device)
            p_ori_y = torch.zeros(batch_size_actual, dtype=torch.float32, device=device)
            p_part_y = torch.zeros(batch_size_actual, dtype=torch.float32, device=device)
            p_erase_y = torch.zeros(batch_size_actual, dtype=torch.float32, device=device)
            local_e = torch.zeros(batch_size_actual, dtype=torch.float32, device=device)

            if part_batch['num_valid'] > 0:
                _part_logits, valid_part_feat = forward_projected_features_without_bn_update(k_model, part_batch['x_part'])
                part_feat_batch[valid_positions] = valid_part_feat.float()
                p_ori_y[valid_positions] = part_batch['p_ori_y'].float()
                p_part_y[valid_positions] = part_batch['p_part_y'].float()
                p_erase_y[valid_positions] = part_batch['p_erase_y'].float()
                local_e[valid_positions] = (
                    part_batch['p_part_y'].float() *
                    (part_batch['p_ori_y'].float() - part_batch['p_erase_y'].float()).clamp(min=0.0)
                )

            cls_pred = logits1.argmax(dim=1)
            for pos in range(batch_size_actual):
                valid_part = bool(valid_mask[pos].item())
                feature_pos = len(global_features)
                global_features.append(ema_feat1[pos].detach().cpu().float())
                part_features.append(part_feat_batch[pos].detach().cpu().float())
                records.append({
                    'sample_index': int(sample_indices[pos].item()),
                    'diag_batch_idx': int(batch_idx),
                    'label': int(labels[pos].item()),
                    'cls_global_pred': int(cls_pred[pos].item()),
                    'is_clean': bool(is_clean[pos].item()),
                    'is_id': bool(is_id[pos].item()),
                    'is_ood': bool(is_ood[pos].item()),
                    'valid_part': valid_part,
                    'feature_pos': feature_pos,
                    'p_ori_y': safe_float(p_ori_y[pos].item()),
                    'p_part_y': safe_float(p_part_y[pos].item()),
                    'p_erase_y': safe_float(p_erase_y[pos].item()),
                    'local_e': safe_float(local_e[pos].item()),
                    'valid_lookup_pos': valid_lookup.get(pos, -1),
                })

    if len(global_features) == 0:
        feature_dim = int(cfg.fdim)
        global_tensor = torch.zeros(0, feature_dim)
        part_tensor = torch.zeros(0, feature_dim)
    else:
        global_tensor = torch.stack(global_features, dim=0)
        part_tensor = torch.stack(part_features, dim=0)
    return records, global_tensor, part_tensor, clean_total, id_total, ood_total


def build_summary_and_samples(records, global_features, part_features, cfg, args, checkpoint):
    # holdout 主指标只在 clean ∩ valid_part ∩ p_ori_y>=阈值 的 eval split 上计算。
    num_total = len(records)
    labels = torch.tensor([row['label'] for row in records], dtype=torch.long)
    cls_pred = torch.tensor([row['cls_global_pred'] for row in records], dtype=torch.long)
    is_clean = torch.tensor([row['is_clean'] for row in records], dtype=torch.bool)
    valid_part = torch.tensor([row['valid_part'] for row in records], dtype=torch.bool)
    p_ori = torch.tensor([row['p_ori_y'] for row in records], dtype=torch.float32)
    local_e = torch.tensor([row['local_e'] for row in records], dtype=torch.float32)
    diag_batch_idx = torch.tensor([row.get('diag_batch_idx', 0) for row in records], dtype=torch.long)
    candidate_mask = is_clean & valid_part & (p_ori >= float(args.proto_conf_thr))
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    candidate_eapa_weights = torch.zeros(candidate_indices.numel(), dtype=torch.float32)
    if candidate_indices.numel() > 0:
        # EAPA 训练先在每个 batch 的全部 update candidates 内排序；holdout 只决定哪些样本进入 prototype build。
        candidate_batch_idx = diag_batch_idx[candidate_indices]
        for batch_id in candidate_batch_idx.unique(sorted=True):
            batch_mask = candidate_batch_idx == batch_id
            batch_indices = candidate_indices[batch_mask]
            eapa_rank_score = p_ori[batch_indices] * local_e[batch_indices]
            candidate_eapa_weights[batch_mask] = split_rank_norm(eapa_rank_score) * p_ori[batch_indices]

    build_indices = torch.zeros(0, dtype=torch.long)
    eval_indices = torch.zeros(0, dtype=torch.long)
    build_candidate_weights = torch.zeros(0, dtype=torch.float32)
    if candidate_indices.numel() > 0:
        build_mask_local, eval_mask_local = deterministic_split(
            int(candidate_indices.numel()), args.proto_build_ratio, args.split_seed
        )
        build_indices = candidate_indices[build_mask_local]
        eval_indices = candidate_indices[eval_mask_local]
        build_candidate_weights = candidate_eapa_weights[build_mask_local]

    build_labels = labels[build_indices]
    build_global_features = global_features[build_indices] if build_indices.numel() > 0 else global_features[:0]
    build_part_features = part_features[build_indices] if build_indices.numel() > 0 else part_features[:0]
    normal_weights = p_ori[build_indices] if build_indices.numel() > 0 else torch.zeros(0)
    eapa_weights = build_candidate_weights

    normal_part_proto, normal_part_valid = build_weighted_prototypes(
        build_part_features, build_labels, normal_weights, cfg.n_classes
    )
    eapa_part_proto, eapa_part_valid = build_weighted_prototypes(
        build_part_features, build_labels, eapa_weights, cfg.n_classes
    )
    normal_global_proto, normal_global_valid = build_weighted_prototypes(
        build_global_features, build_labels, normal_weights, cfg.n_classes
    )
    eapa_global_proto, eapa_global_valid = build_weighted_prototypes(
        build_global_features, build_labels, eapa_weights, cfg.n_classes
    )

    eval_labels = labels[eval_indices]
    eval_global_features = global_features[eval_indices] if eval_indices.numel() > 0 else global_features[:0]
    eval_part_features = part_features[eval_indices] if eval_indices.numel() > 0 else part_features[:0]
    eval_cls_pred = cls_pred[eval_indices] if eval_indices.numel() > 0 else cls_pred[:0]
    eval_local_e = local_e[eval_indices] if eval_indices.numel() > 0 else local_e[:0]
    eval_p_ori = p_ori[eval_indices] if eval_indices.numel() > 0 else p_ori[:0]

    normal_part_eval = evaluate_prototypes(
        eval_part_features, eval_labels, normal_part_proto, normal_part_valid, temp=args.proto_temp
    )
    eapa_part_eval = evaluate_prototypes(
        eval_part_features, eval_labels, eapa_part_proto, eapa_part_valid, temp=args.proto_temp
    )
    normal_global_eval = evaluate_prototypes(
        eval_global_features, eval_labels, normal_global_proto, normal_global_valid, temp=args.proto_temp
    )
    eapa_global_eval = evaluate_prototypes(
        eval_global_features, eval_labels, eapa_global_proto, eapa_global_valid, temp=args.proto_temp
    )

    part_available = eapa_part_eval['target_available']
    global_available = eapa_global_eval['target_available']
    used_mask = part_available
    used_global_mask = global_available
    num_eval_total = int(eval_indices.numel())
    num_eval_used = int(used_mask.sum().item())
    num_eval_without = int(num_eval_total - num_eval_used)

    cls_acc = 0.0
    if num_eval_total > 0:
        cls_acc = float((eval_cls_pred == eval_labels).float().mean().item())

    eapa_part_correct = (eapa_part_eval['pred'] == eval_labels) & used_mask
    normal_part_correct = (normal_part_eval['pred'] == eval_labels) & normal_part_eval['target_available']
    eapa_global_correct = (eapa_global_eval['pred'] == eval_labels) & used_global_mask
    cls_correct = eval_cls_pred == eval_labels
    part_global_compare_mask = used_mask & used_global_mask

    groups = evidence_groups(tensor_to_cpu_list(eval_local_e[used_mask]))
    used_eapa_correct = tensor_to_cpu_list(eapa_part_correct[used_mask].float())
    used_eapa_margin = tensor_to_cpu_list(eapa_part_eval['margin'][used_mask])

    pair_counter = Counter(
        (row['label'], row['cls_global_pred'])
        for row in records
        if row['label'] != row['cls_global_pred']
    )
    top_pairs = [
        {'label': int(label), 'cls_global_pred': int(pred), 'count': int(count)}
        for (label, pred), count in pair_counter.most_common(int(args.top_confused_pairs))
    ]
    top_pair_set = {(item['label'], item['cls_global_pred']) for item in top_pairs}
    confused_eval_mask = torch.tensor([
        (int(label.item()), int(pred.item())) in top_pair_set
        for label, pred in zip(eval_labels, eval_cls_pred)
    ], dtype=torch.bool)
    confused_used = confused_eval_mask & used_mask
    if int(confused_used.sum().item()) > 0:
        confused_recover = float(eapa_part_correct[confused_used].float().mean().item())
        confused_margin = mean_from_values(tensor_to_cpu_list(eapa_part_eval['margin'][confused_used]))
    else:
        confused_recover = 0.0
        confused_margin = 0.0

    eligible_local = tensor_to_cpu_list(local_e[candidate_indices])
    if len(eligible_local) > 0:
        eligible_groups = evidence_groups(eligible_local)
        high_local = [v for v, g in zip(eligible_local, eligible_groups) if g == 'high']
        low_local = [v for v, g in zip(eligible_local, eligible_groups) if g == 'low']
    else:
        high_local, low_local = [], []

    expected_clean_ratio = read_expected_clean_ratio(args.ckpt_path, checkpoint, num_total, args.max_batches)
    josnc_clean_ratio = float(sum(row['is_clean'] for row in records) / max(num_total, 1))
    if expected_clean_ratio is None:
        expected_clean_ratio_value = -1.0
        clean_ratio_abs_diff = -1.0
        trust_flag = 'unknown_log'
    else:
        expected_clean_ratio_value = expected_clean_ratio
        clean_ratio_abs_diff = abs(josnc_clean_ratio - expected_clean_ratio)
        # 该标记只描述静态重放和下一轮日志是否接近，不判定 checkpoint 或训练日志本身是否异常。
        if clean_ratio_abs_diff <= float(args.clean_ratio_trust_tol):
            trust_flag = 'replay_close_to_next_epoch_log'
        else:
            trust_flag = 'replay_differs_from_next_epoch_log'

    summary = {
        'checkpoint': os.path.basename(args.ckpt_path),
        'num_total': num_total,
        'num_clean': int(sum(row['is_clean'] for row in records)),
        'num_valid_part': int(sum(row['valid_part'] for row in records)),
        'num_update_candidate': int(candidate_indices.numel()),
        'josnc_clean_ratio': josnc_clean_ratio,
        'josnc_id_ratio': float(sum(row['is_id'] for row in records) / max(num_total, 1)),
        'josnc_ood_ratio': float(sum(row['is_ood'] for row in records) / max(num_total, 1)),
        # Legacy CSV 字段名保留兼容；语义是静态重放与下一轮日志 clean ratio 的参考对比。
        'expected_clean_ratio_from_log': expected_clean_ratio_value,
        'clean_ratio_abs_diff': clean_ratio_abs_diff,
        'clean_selection_trust_flag': trust_flag,
        'num_eval_total': num_eval_total,
        'num_eval_with_target_part_proto': num_eval_used,
        'num_eval_without_target_part_proto': num_eval_without,
        'num_eval_used_for_part_proto': num_eval_used,
        'part_proto_target_available_ratio': float(num_eval_used / max(num_eval_total, 1)),
        'normal_part_proto_valid_classes': int(normal_part_valid.sum().item()),
        'eapa_part_proto_valid_classes': int(eapa_part_valid.sum().item()),
        'part_proto_valid_class_ratio': float(eapa_part_valid.float().mean().item()) if eapa_part_valid.numel() > 0 else 0.0,
        'global_proto_valid_class_ratio': float(eapa_global_valid.float().mean().item()) if eapa_global_valid.numel() > 0 else 0.0,
        'cls_global_acc': cls_acc,
        'normal_part_proto_acc': float(normal_part_eval['acc']),
        'eapa_part_proto_acc': float(eapa_part_eval['acc']),
        'normal_global_proto_acc': float(normal_global_eval['acc']),
        'eapa_global_proto_acc': float(eapa_global_eval['acc']),
        'normal_part_target_sim': mean_from_values(tensor_to_cpu_list(normal_part_eval['target_sim'][normal_part_eval['target_available']])),
        'eapa_part_target_sim': mean_from_values(tensor_to_cpu_list(eapa_part_eval['target_sim'][used_mask])),
        'normal_part_margin': mean_from_values(tensor_to_cpu_list(normal_part_eval['margin'][normal_part_eval['target_available']])),
        'eapa_part_margin': mean_from_values(tensor_to_cpu_list(eapa_part_eval['margin'][used_mask])),
        'normal_part_entropy': mean_from_values(tensor_to_cpu_list(normal_part_eval['entropy'][normal_part_eval['target_available']])),
        'eapa_part_entropy': mean_from_values(tensor_to_cpu_list(eapa_part_eval['entropy'][used_mask])),
        'normal_part_target_prob': mean_from_values(tensor_to_cpu_list(normal_part_eval['target_prob'][normal_part_eval['target_available']])),
        'eapa_part_target_prob': mean_from_values(tensor_to_cpu_list(eapa_part_eval['target_prob'][used_mask])),
        'high_evidence_part_acc': grouped_metric(groups, 'high', [], correct=used_eapa_correct),
        'mid_evidence_part_acc': grouped_metric(groups, 'mid', [], correct=used_eapa_correct),
        'low_evidence_part_acc': grouped_metric(groups, 'low', [], correct=used_eapa_correct),
        'high_evidence_part_margin': grouped_metric(groups, 'high', used_eapa_margin),
        'mid_evidence_part_margin': grouped_metric(groups, 'mid', used_eapa_margin),
        'low_evidence_part_margin': grouped_metric(groups, 'low', used_eapa_margin),
        'cls_wrong_eapa_part_correct': int(((~cls_correct) & eapa_part_correct).sum().item()),
        'cls_correct_eapa_part_wrong': int((cls_correct & used_mask & (~eapa_part_correct)).sum().item()),
        'global_proto_wrong_eapa_part_correct': int(((~eapa_global_correct) & eapa_part_correct & used_global_mask).sum().item()),
        'global_proto_correct_eapa_part_wrong': int((eapa_global_correct & used_mask & (~eapa_part_correct)).sum().item()),
        'part_global_agree_ratio': float(
            ((eapa_part_eval['pred'] == eapa_global_eval['pred']) & part_global_compare_mask).float().sum().item() /
            max(int(part_global_compare_mask.sum().item()), 1)
        ),
        'part_correct_global_proto_wrong_ratio': float((eapa_part_correct & (~eapa_global_correct) & used_global_mask).float().sum().item() / max(num_eval_used, 1)),
        'confused_pair_source': 'label_y_vs_cls_global_pred',
        'top_confused_pairs': json.dumps(top_pairs, ensure_ascii=False),
        'confused_pair_recover_rate': confused_recover,
        'confused_pair_margin_mean': confused_margin,
        'local_e_mean': mean_from_values(eligible_local),
        'local_e_std': std_from_values(eligible_local),
        'local_e_p25': percentile_from_values(eligible_local, 25),
        'local_e_p50': percentile_from_values(eligible_local, 50),
        'local_e_p75': percentile_from_values(eligible_local, 75),
        'local_e_high_mean': mean_from_values(high_local),
        'local_e_low_mean': mean_from_values(low_local),
        'corr_local_e_pori': safe_corr(eligible_local, tensor_to_cpu_list(p_ori[candidate_indices])),
        'corr_local_e_normal_part_margin': safe_corr(
            tensor_to_cpu_list(eval_local_e[normal_part_eval['target_available']]),
            tensor_to_cpu_list(normal_part_eval['margin'][normal_part_eval['target_available']]),
        ),
        'corr_local_e_eapa_part_margin': safe_corr(
            tensor_to_cpu_list(eval_local_e[used_mask]),
            tensor_to_cpu_list(eapa_part_eval['margin'][used_mask]),
        ),
    }

    sample_rows = []
    all_eval_groups = evidence_groups(tensor_to_cpu_list(eval_local_e))
    for local_pos, record_idx in enumerate(tensor_to_cpu_list(eval_indices)):
        record = records[int(record_idx)]
        sample_rows.append({
            'sample_index': record['sample_index'],
            'label': record['label'],
            'cls_global_pred': record['cls_global_pred'],
            'normal_part_pred': int(normal_part_eval['pred'][local_pos].item()) if num_eval_total > 0 else -1,
            'eapa_part_pred': int(eapa_part_eval['pred'][local_pos].item()) if num_eval_total > 0 else -1,
            'normal_global_proto_pred': int(normal_global_eval['pred'][local_pos].item()) if num_eval_total > 0 else -1,
            'eapa_global_proto_pred': int(eapa_global_eval['pred'][local_pos].item()) if num_eval_total > 0 else -1,
            'evidence_rank_group': all_eval_groups[local_pos] if local_pos < len(all_eval_groups) else 'mid',
            'p_ori_y': record['p_ori_y'],
            'p_part_y': record['p_part_y'],
            'p_erase_y': record['p_erase_y'],
            'local_e': record['local_e'],
            'normal_part_margin': safe_float(normal_part_eval['margin'][local_pos].item()) if num_eval_total > 0 else 0.0,
            'eapa_part_margin': safe_float(eapa_part_eval['margin'][local_pos].item()) if num_eval_total > 0 else 0.0,
            'normal_part_target_sim': safe_float(normal_part_eval['target_sim'][local_pos].item()) if num_eval_total > 0 else 0.0,
            'eapa_part_target_sim': safe_float(eapa_part_eval['target_sim'][local_pos].item()) if num_eval_total > 0 else 0.0,
        })

    return summary, sample_rows


def format_csv_value(value):
    if isinstance(value, float):
        return f'{safe_float(value):.6f}'
    return value


def write_outputs(output_dir, summary, sample_rows):
    # 两份 CSV 固定字段顺序，便于后续直接跨实验汇总。
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'part_proto_diag.csv')
    sample_path = os.path.join(output_dir, 'part_proto_diag_samples.csv')
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({field: format_csv_value(summary.get(field, 0)) for field in SUMMARY_FIELDS})
    with open(sample_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({field: format_csv_value(row.get(field, 0)) for field in SAMPLE_FIELDS})
    return summary_path, sample_path


def main():
    args = parse_args()
    cfg = load_config(args.cfg, data_root=args.data_root)
    set_seed(cfg.seed)
    device = resolve_device(args.gpu)

    q_model = build_model(cfg, device)
    k_model = build_model(cfg, device)
    checkpoint, queue_keys, queue_logits, tau_c, tau_o = load_full_checkpoint(
        args.ckpt_path, q_model, k_model, device
    )
    warn_if_k2_checkpoint(cfg, checkpoint)
    args.proto_conf_thr = resolve_proto_conf_thr(cfg, args.proto_conf_thr)

    records, global_features, part_features, clean_total, id_total, ood_total = collect_diagnostic_records(
        cfg, args, q_model, k_model, queue_keys, queue_logits, tau_c, tau_o, device
    )
    summary, sample_rows = build_summary_and_samples(
        records, global_features, part_features, cfg, args, checkpoint
    )
    summary['num_clean'] = clean_total
    summary['josnc_clean_ratio'] = float(clean_total / max(len(records), 1))
    summary['josnc_id_ratio'] = float(id_total / max(len(records), 1))
    summary['josnc_ood_ratio'] = float(ood_total / max(len(records), 1))

    summary_path, sample_path = write_outputs(args.output, summary, sample_rows)
    print(f'Wrote summary CSV: {summary_path}')
    print(f'Wrote sample CSV: {sample_path}')
    print(
        'Key metrics: '
        f"normal_part_acc={summary['normal_part_proto_acc']:.4f}, "
        f"eapa_part_acc={summary['eapa_part_proto_acc']:.4f}, "
        f"high_evidence_acc={summary['high_evidence_part_acc']:.4f}, "
        f"low_evidence_acc={summary['low_evidence_part_acc']:.4f}, "
        f"replay_flag={summary['clean_selection_trust_flag']}"
    )


if __name__ == '__main__':
    main()
