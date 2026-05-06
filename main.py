# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : main.py.py
#   @Author      : Zeren Sun
#   @Created date: 2022/11/18 10:21
#   @Description : Efficient KNN-based Selection Prior + NCR
#
# ================================================================
import os
import sys
import time
import pathlib
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import yaml
import shutil
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from utils.logger import Logger, Writer
from utils.model import Model, DualHeadModel
from utils.builder import build_transform, get_dataset_normalization, build_cifar100n_dataset, build_webfg_dataset, build_food101n_dataset, build_clothing1m_dataset, build_mini_webvision_dataset, build_animal10n_dataset
from utils.eval import accuracy, evaluate, detection_evaluate
from utils.utils import *
from utils.loss import *
from utils.local_evidence import (
    ID_CANDIDATE_CSV_HEADER,
    ID_CANDIDATE_SAMPLE_CSV_HEADER,
    LOCAL_EVIDENCE_CSV_HEADER,
    PART_CE_GATE_SAMPLE_CSV_HEADER,
    PART_CE_CSV_HEADER,
    attach_id_candidate_loss_results,
    build_id_candidate_batch,
    build_id_candidate_log_row,
    build_id_candidate_sample_rows,
    build_gate_mask,
    build_local_part_batch,
    build_part_ce_gate_sample_rows,
    build_part_ce_log_row,
    compute_id_candidate_effective_weight,
    compute_id_candidate_loss,
    compute_local_evidence,
    format_id_candidate_row,
    format_id_candidate_sample_row,
    format_local_evidence_row,
    format_part_ce_gate_sample_row,
    format_part_ce_row,
)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOG_FREQ = 1
SAVE_WARMUP_CKPT = False


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def save_current_config(log_dir, cfg):
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(cfg), f, sort_keys=False)


def save_network_arch(log_dir, net):
    with open(f'{log_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def wrapup_training_statics(result_dir, best_accuracy):
    stats = get_stats(f'{result_dir}/log.txt')
    if len(stats['valid_epoch']) == 0:
        # 短跑或诊断任务可能没有可统计的 epoch 汇总行，此时保留原目录即可。
        with open(f'{result_dir}/stats.txt', 'w') as f:
            f.write('No valid epoch records found; skip mean/std statistics.\n')
        return
    with open(f'{result_dir}/stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
            mean_accuracy = stats['mean']
            std_accuracy = stats['std']
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
            mean_accuracy = stats['mean1']
            std_accuracy = stats['std1']
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.2f}-MeanAcc_{mean_accuracy:.2f}_{std_accuracy:.2f}')


def build_logger(logger_root, dataset_name, project_tag, log_tag, enable_debug_logging=True):
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.now().strftime('%Y%m%d%H%M%S')
    if 'ablation' in project_tag:
        exp_log_dir = os.path.join(logger_root, dataset_name, project_tag, f'{log_tag}-{logtime}')
    elif 'benchmark' in project_tag:
        exp_log_dir = os.path.join(logger_root, project_tag, f'{dataset_name}-{log_tag}-{logtime}')
    else:
        exp_log_dir = os.path.join(logger_root, dataset_name, project_tag, f'{logtime}-{log_tag}')
    exp_logger = Logger(logging_dir=exp_log_dir, DEBUG=enable_debug_logging)
    exp_logger.set_logfile(logfile_name='log.txt')
    return exp_logger, exp_log_dir


def build_optimizer(cfg, net_params):
    if cfg.opt == 'sgd':
        return torch.optim.SGD(net_params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
    elif cfg.opt == 'adam':
        return torch.optim.Adam(net_params, lr=cfg.lr, weight_decay=cfg.weight_decay)  # , betas=(0.9, 0.999), amsgrad=False)
    else:
        raise ValueError(f'{cfg.opt} optimizer is not supported.')


def build_dataset(cfg):
    transform = build_transform(cfg.rescale_size, cfg.crop_size, dataset=cfg.dataset)
    if cfg.dataset.startswith('cifar100n'):
        assert cfg.ood_noise_rate == 0.0, f'ood_noise_rate should be 0.0 in cifar100n-* datasets'
        assert cfg.n_classes == 100, f'number of classes should be 100'
        assert cfg.dataset.split('-')[1][-2:] == str(int(cfg.idn_noise_rate*100))
        if cfg.transform == 'moco':
            transform_type = 'train_moco'
        elif cfg.transform == 'strong':
            transform_type = 'cifar_train_strong_aug'
        else:
            transform_type = 'cifar_train'
        dataset = build_cifar100n_dataset(os.path.join(cfg.data_root, 'cifar100'), MultiDataTransform([transform['cifar_train'], transform[transform_type]]),
                                          transform['cifar_test'], cfg.noise_type, 0.0, cfg.idn_noise_rate)
    elif cfg.dataset.startswith('cifar80n'):
        assert cfg.ood_noise_rate == 0.2, f'ood_noise_rate should be 0.2 in cifar80n-* datasets'
        assert cfg.n_classes == 80, f'number of classes should be 80'
        assert cfg.dataset.split('-')[1][-2:] == str(int(cfg.idn_noise_rate * 100))
        if cfg.transform == 'moco':
            transform_type = 'train_moco'
        elif cfg.transform == 'strong':
            transform_type = 'cifar_train_strong_aug'
        else:
            transform_type = 'cifar_train'
        dataset = build_cifar100n_dataset(os.path.join(cfg.data_root, 'cifar100'), MultiDataTransform([transform['cifar_train'], transform[transform_type]]),
                                          transform['cifar_test'], cfg.noise_type, 0.2, cfg.idn_noise_rate)
    elif cfg.dataset == 'animal10n':
        if cfg.transform == 'strong':
            transform_type = 'cifar_train_strong_aug'
        else:
            transform_type = 'cifar_train'
        dataset = build_animal10n_dataset(os.path.join(cfg.data_root, cfg.dataset),  MultiDataTransform([transform['cifar_train'], transform[transform_type]]), transform['cifar_test'])
    elif cfg.dataset in ['web-aircraft', 'web-bird', 'web-car']:
        if cfg.transform == 'weak':
            transform_type = 'train'
        else:
            transform_type = 'train_strong_aug'
        dataset = build_webfg_dataset(os.path.join(cfg.data_root, cfg.dataset), MultiDataTransform([transform['train'], transform[transform_type]]), transform['test'])
    elif cfg.dataset == 'food101n':
        if cfg.transform == 'weak':
            transform_type = 'train'
        else:
            transform_type = 'train_strong_aug'
        dataset = build_food101n_dataset(os.path.join(cfg.data_root, cfg.dataset), MultiDataTransform([transform['train'], transform[transform_type]]), transform['test'])
    elif cfg.dataset in ['mini-webvision', 'webvision']:
        if cfg.transform == 'weak':
            transform_type = 'train'
        else:
            transform_type = 'train_strong_aug'
        dataset = build_mini_webvision_dataset(os.path.join(cfg.data_root, cfg.dataset), MultiDataTransform([transform['train'], transform[transform_type]]), transform['test'], num_class=cfg.n_classes)
    else:
        raise NotImplementedError(f'{cfg.dataset} is not supported.')
    return dataset


def momentum_update_key_network(qnet, knet, moco_m=0.999):
    with torch.no_grad():
        for param_q, param_k in zip(qnet.parameters(), knet.parameters()):
            param_k.data = param_k.data * moco_m + param_q.data * (1. - moco_m)


def samples_identification(logits1, logits2, ob_labels, features, features_queue, logits_queue, threshold_clean, threshold_ood, config, logger):
    with torch.no_grad():
        probs1, probs2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
        # identify clean samples : self-based
        prob_clean = 1 - js_div(probs1, ob_labels)
        cleanness_self = torch.ge(prob_clean, threshold_clean)
        # identify clean samples : neighbor-based
        similarity = torch.mm(features, features_queue.t())  # (batch_size, queue_length)
        # similarity = F.relu(similarity, inplace=False)
        _, neighbor_indices = similarity.topk(config.n_neighbors + 1, dim=1, largest=True, sorted=True)     # (batch_size, n_neighbors+1)
        neighbor_indices = neighbor_indices[:, 1:].contiguous().view(-1)                                    # (batch_size*n_neighbors,)
        neighbor_probs = logits_queue[neighbor_indices].softmax(dim=1)                                      # (batch_size*n_neighbors, nc)
        neighbor_ob_labels = ob_labels.repeat(1, config.n_neighbors).view(-1, config.n_classes)             # (batch_size*n_neighbors, nc)
        neighbor_prob_clean = 1 - js_div(neighbor_probs, neighbor_ob_labels).view(-1, config.n_neighbors).mean(dim=1)  # (batch_size,)
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
            raise AssertionError(f'integrate_mode should be within [and, or, self-only, neighbor-only], the current value is {config.integrate_mode}')
        unclean = clean.logical_not()
        idx_clean = clean.nonzero(as_tuple=False).squeeze(dim=1)

        # distinguish id and ood noisy samples
        prob_ood = js_div(F.softmax(logits1 / 0.1, dim=1), F.softmax(logits2 / 0.1, dim=1))
        pred1, pred2 = probs1.argmax(dim=1), probs2.argmax(dim=1)
        if config.ood_criterion.startswith('div'):
            disagree = (prob_ood > threshold_ood)
            agree = (prob_ood <= threshold_ood)
        elif config.ood_criterion.startswith('dis'):
            disagree = (pred1 != pred2)
            agree = (pred1 == pred2)
        else:
            raise AssertionError(f'ood_criterion should be within [div, dis], the current value is {config.ood_criterion}')
        idx_ood = (disagree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
        idx_id = (agree * unclean).nonzero(as_tuple=False).squeeze(dim=1)

    logger.debug(f'  |- p_clean mid: {prob_clean.median().item():.3f}, p_clean avg: {prob_clean.mean().item():.3f} || '
                 f'p_clean[clean] avg: {prob_clean[idx_clean].mean().item():.3f}, '
                 f'p_clean[id] avg: {prob_clean[idx_id].mean().item():.3f}, '
                 f'p_clean[ood] avg: {prob_clean[idx_ood].mean().item():.3f}\n'
                 f'  |- p_ood mid: {prob_ood.median().item():.3f}, p_ood avg: {prob_ood.mean().item():.3f} || '
                 f'p_ood[clean] avg: {prob_ood[idx_clean].mean().item():.3f}, '
                 f'p_ood[id] avg: {prob_ood[idx_id].mean().item():.3f}, '
                 f'p_ood[ood] avg: {prob_ood[idx_ood].mean().item():.3f}')
    logger.debug(f'  |- idx_clean: {idx_clean.shape[0]}, idx_id: {idx_id.shape[0]}, idx_ood: {idx_ood.shape[0]}')
    return idx_clean, idx_id, idx_ood, prob_clean, prob_ood


def _should_run_local_evidence(cfg, epoch, batch_idx):
    # A1 诊断按 epoch 间隔和 batch 上限抽样，避免可视化/CSV 记录拖慢完整训练。
    if cfg.local_evidence_every <= 0:
        return False
    if (epoch % cfg.local_evidence_every) != 0:
        return False
    if cfg.local_evidence_max_batches > 0 and batch_idx >= cfg.local_evidence_max_batches:
        return False
    return True


def generate_label_sets(batch_label_sets, nc):
    bs = batch_label_sets.size(0)
    label_sets = torch.zeros(bs, nc).to(batch_label_sets.device)
    label_sets.scatter_(dim=1, index=batch_label_sets, value=1)
    return label_sets


def gmm_based_threshold_generation(value_list, num_classes):
    values = np.array(value_list).reshape(-1, 1)
    gmm_metric = GaussianMixture(2)
    gmm_metric.fit(values)
    v_pred = gmm_metric.predict(values)
    max0 = values[v_pred == 0].max()
    max1 = values[v_pred == 1].max()
    min0 = values[v_pred == 0].min()
    min1 = values[v_pred == 1].min()
    temp = [min0, min1, max0, max1]
    temp.sort()
    ret = (temp[1] + temp[2]) / 2
    # ret = gmm_metric.means_.mean()
    return ret * torch.ones(num_classes)


def mean_based_threshold_generation(value_list, num_classes):
    values = np.array(value_list)
    return values.mean() * torch.ones(num_classes)


def per_class_mean_based_threshold_generation(value_list, label_list, num_classes):
    values_array = np.array(value_list)
    labels_array = np.array(label_list)
    per_class_thresholds = [0.0] * num_classes
    for i in range(num_classes):
        values_of_ith_class = values_array[labels_array == i]
        per_class_thresholds[i] = values_of_ith_class.mean()
    # per_class_thresholds = [values_array[labels_array == i].mean() for i in range(num_classes)]
    # assert check_nan_inf(per_class_thresholds), f'{per_class_thresholds[np.isnan(per_class_thresholds)], per_class_thresholds[np.isinf(per_class_thresholds)]}'
    return torch.tensor(per_class_thresholds)


def main(gpu, cfg):
    cudnn.deterministic = True
    cudnn.benchmark = cfg.benchmark
    torch.cuda.empty_cache()

    set_seed(cfg.seed)
    device = torch.device(f'cuda:{gpu}')

    # model
    q_model = DualHeadModel(arch=cfg.arch, num_classes=cfg.n_classes, mlp_hidden=cfg.hdim, feature_dim=cfg.fdim, pretrained=True, use_bn=True).to(device)
    k_model = DualHeadModel(arch=cfg.arch, num_classes=cfg.n_classes, mlp_hidden=cfg.hdim, feature_dim=cfg.fdim, pretrained=True, use_bn=True).to(device)
    for param_q, param_k in zip(q_model.parameters(), k_model.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False     # not update by gradient

    # optimizer, scheduler
    optim = build_optimizer(cfg, q_model.parameters())
    lr_plan = build_lr_plan(cfg.lr, cfg.epochs, cfg.warmup_epochs, cfg.warmup_lr, cfg.lr_decay)  #, warmup_rampup=(cfg.warmup_lr_plan != 'constant'))

    # dataset, dataloader
    dataset = build_dataset(cfg)
    train_loader = DataLoader(dataset['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if 'webvision' in cfg.dataset:
        valid_loader = DataLoader(dataset['valid'], batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    n_train_samples = dataset['n_train_samples']
    if cfg.eval_det == 1 and cfg.dataset.startswith('cifar'):
        gt_indices_clean, gt_indices_id, gt_indices_ood = dataset['train_indices_clean'], dataset['train_indices_idn'], dataset['train_indices_ood']
        gt_indicator_clean = indices_list_to_indicator_vector(gt_indices_clean, n_train_samples)
        gt_indicator_id = indices_list_to_indicator_vector(gt_indices_id, n_train_samples)
        gt_indicator_ood = indices_list_to_indicator_vector(gt_indices_ood, n_train_samples)
        assert (gt_indicator_clean + gt_indicator_id + gt_indicator_ood == 1).all()
        gt_train_labels = torch.tensor(np.array(dataset['train'].targets)).long()
    else:
        gt_indicator_clean, gt_indicator_id, gt_indicator_ood = None, None, None
        gt_train_labels = torch.zeros(n_train_samples).long()

    # Logging
    logger, result_dir = build_logger(cfg.log_root, cfg.dataset, cfg.log_proj, cfg.log_name)
    save_current_script(result_dir)
    save_current_config(result_dir, cfg)
    save_network_arch(result_dir, q_model)
    logger.msg(f'Result Path   : {result_dir}')
    logger.msg(f"# of training data: {n_train_samples}, # of test data: {dataset['n_test_samples']}")

    threshold_writer = Writer(root_dir=result_dir, filename='threshold.csv', header='epoch,threshold_clean,threshold_ood')
    pr_metric_writer = Writer(root_dir=result_dir, filename='prfa_metric.csv', header='epoch,N,P,R,F1,AUROC,N,P,R,F1,AUROC,N,P,R,F1,AUROC')
    pll_topk_acc_writer = Writer(root_dir=result_dir, filename='pll_topk_acc.csv', header='epoch,top1AccID,topkAccID,top1AccOOD,topkAccOOD')
    if cfg.part_ce and cfg.part_ce_groups != 'clean':
        raise ValueError('B1/C1 currently supports part_ce_groups=clean only.')
    if cfg.part_ce_gate_type not in ['fixed', 'percentile']:
        raise ValueError(f'part_ce_gate_type should be fixed or percentile, got {cfg.part_ce_gate_type}.')
    if not (0.0 <= cfg.part_ce_gate_keep_ratio <= 1.0):
        raise ValueError(f'part_ce_gate_keep_ratio should be within [0, 1], got {cfg.part_ce_gate_keep_ratio}.')
    if not (1 <= cfg.id_candidate_topk <= cfg.n_classes):
        raise ValueError(f'id_candidate_topk should be within [1, {cfg.n_classes}], got {cfg.id_candidate_topk}.')
    if cfg.id_candidate_cam_target not in ['teacher_top1']:
        raise ValueError(f'id_candidate_cam_target should be teacher_top1, got {cfg.id_candidate_cam_target}.')
    if cfg.id_candidate_score_type not in ['ori_part_minus_erase']:
        raise ValueError(f'id_candidate_score_type should be ori_part_minus_erase, got {cfg.id_candidate_score_type}.')
    if cfg.id_candidate_loss_type not in ['pll', 'pll_entropy', 'capped_soft']:
        raise ValueError(f'id_candidate_loss_type should be pll, pll_entropy, or capped_soft, got {cfg.id_candidate_loss_type}.')
    if cfg.id_candidate_weight < 0.0:
        raise ValueError(f'id_candidate_weight should be non-negative, got {cfg.id_candidate_weight}.')
    if cfg.id_candidate_entropy_weight < 0.0:
        raise ValueError(f'id_candidate_entropy_weight should be non-negative, got {cfg.id_candidate_entropy_weight}.')
    if not (0.0 <= cfg.id_candidate_entropy_min_ratio <= 1.0):
        raise ValueError(
            f'id_candidate_entropy_min_ratio should be within [0, 1], got {cfg.id_candidate_entropy_min_ratio}.'
        )
    if cfg.id_candidate_dist_weight < 0.0:
        raise ValueError(f'id_candidate_dist_weight should be non-negative, got {cfg.id_candidate_dist_weight}.')
    if cfg.id_candidate_target_temp <= 0.0:
        raise ValueError(f'id_candidate_target_temp should be positive, got {cfg.id_candidate_target_temp}.')
    if not (0.0 <= cfg.id_candidate_top1_cap <= 1.0):
        raise ValueError(f'id_candidate_top1_cap should be within [0, 1], got {cfg.id_candidate_top1_cap}.')
    if not (0.0 <= cfg.id_candidate_noisy_prior <= 1.0):
        raise ValueError(f'id_candidate_noisy_prior should be within [0, 1], got {cfg.id_candidate_noisy_prior}.')
    if cfg.id_candidate_decay_start_epoch < 0 or cfg.id_candidate_decay_end_epoch < 0:
        raise ValueError('id_candidate decay epochs should be non-negative.')
    if cfg.id_candidate_decay_end_epoch > 0 and cfg.id_candidate_decay_end_epoch < cfg.id_candidate_decay_start_epoch:
        raise ValueError('id_candidate_decay_end_epoch should be >= id_candidate_decay_start_epoch.')
    if not (0.0 <= cfg.id_candidate_min_weight <= cfg.id_candidate_weight):
        raise ValueError(
            f'id_candidate_min_weight should be within [0, id_candidate_weight], got {cfg.id_candidate_min_weight}.'
        )
    if not (0.0 <= cfg.id_candidate_max_top1_prob <= 1.0):
        raise ValueError(
            f'id_candidate_max_top1_prob should be within [0, 1], got {cfg.id_candidate_max_top1_prob}.'
        )
    part_ce_writer = None
    part_ce_gate_sample_writer = None
    if cfg.part_ce and cfg.part_ce_log:
        # B1/C1 单独写局部 CE 诊断日志；不依赖 A1 local_evidence 开关。
        part_ce_writer = Writer(root_dir=result_dir, filename='part_ce.csv', header=PART_CE_CSV_HEADER)
        part_ce_gate_sample_writer = Writer(
            root_dir=result_dir,
            filename='part_ce_gate_samples.csv',
            header=PART_CE_GATE_SAMPLE_CSV_HEADER,
        )
    id_candidate_writer = None
    id_candidate_sample_writer = None
    if cfg.id_candidate and cfg.id_candidate_log:
        # C2 单独记录 ID noisy 候选集合质量和 PLL 强度，避免和 clean 局部 CE 日志混在一起。
        id_candidate_writer = Writer(root_dir=result_dir, filename='id_candidate.csv', header=ID_CANDIDATE_CSV_HEADER)
        id_candidate_sample_writer = Writer(
            root_dir=result_dir,
            filename='id_candidate_samples.csv',
            header=ID_CANDIDATE_SAMPLE_CSV_HEADER,
        )
    local_evidence_writer = None
    local_evidence_image_dir = None
    local_evidence_norm = None
    if cfg.local_evidence:
        local_evidence_writer = Writer(root_dir=result_dir, filename='local_evidence.csv', header=LOCAL_EVIDENCE_CSV_HEADER)
        if cfg.local_evidence_save_images:
            # 可视化图片使用当前数据集的均值方差反归一化，保存到本次实验目录下。
            local_evidence_image_dir = os.path.join(result_dir, 'local_evidence_images')
            local_evidence_norm = get_dataset_normalization(cfg.dataset)
    if 'webvision' in cfg.dataset:
        test_acc_writer = Writer(root_dir=result_dir, filename='test_acc.csv', header='epoch,Top1Acc,Top5Acc,ImagenetTop1Acc,ImagenetTop5Acc')
    else:
        test_acc_writer = Writer(root_dir=result_dir, filename='test_acc.csv', header='epoch,Acc')

    # meters
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    epoch_train_time = AverageMeter()

    # resume from checkpoint
    resume_checkpoint = None
    if 'ckpt_path' in cfg.keys() and cfg.ckpt_path is not None and os.path.isfile(cfg.ckpt_path):
        logger.debug(f'---> loading {cfg.ckpt_path} <---')
        checkpoint = torch.load(cfg.ckpt_path, map_location=f'cuda:{gpu}')
        q_model.load_state_dict(checkpoint['model_state_dict'])
        if 'k_model_state_dict' in checkpoint:
            k_model.load_state_dict(checkpoint['k_model_state_dict'])
        else:
            k_model.load_state_dict(q_model.state_dict())
        optim.load_state_dict(checkpoint['optim_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        best_epoch = checkpoint['best_epoch']
        resume_checkpoint = checkpoint
    else:
        start_epoch = 0
        best_accuracy = 0.0
        best_epoch = None

    # Contrastive Learning - MoCo
    queue_keys = torch.randn(cfg.queue_length, cfg.fdim).to(device)
    queue_keys = F.normalize(queue_keys, dim=0)
    queue_logits = torch.randn(cfg.queue_length, cfg.n_classes).to(device)
    queue_ptr  = 0
    loss_contrastive_func = SupConLoss(temperature=0.1, base_temperature=0.1)

    if cfg.cls4id == 'ce':
        id_loss_func = F.cross_entropy
    elif cfg.cls4id == 'apl':
        id_loss_func = active_passive_loss
    else:
        raise AssertionError(f'{cfg.cls4id} is not supported.')

    tau_c, tau_o = torch.zeros(cfg.n_classes).to(device), torch.zeros(cfg.n_classes).to(device)

    scaler = GradScaler()
    if resume_checkpoint is not None:
        if 'queue_keys' in resume_checkpoint:
            queue_keys = resume_checkpoint['queue_keys'].to(device)
        if 'queue_logits' in resume_checkpoint:
            queue_logits = resume_checkpoint['queue_logits'].to(device)
        queue_ptr = resume_checkpoint.get('queue_ptr', queue_ptr)
        if 'tau_c' in resume_checkpoint:
            tau_c = resume_checkpoint['tau_c'].to(device)
        if 'tau_o' in resume_checkpoint:
            tau_o = resume_checkpoint['tau_o'].to(device)
        if 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.warmup_fc_only:
            if epoch < cfg.warmup_epochs:
                freeze_layer(q_model.encoder)
            elif epoch == cfg.warmup_epochs:
                unfreeze_layer(q_model.encoder)
        set_seed(cfg.seed + epoch)
        epoch_start = time.time()
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        pr_indices_clean, pr_indices_id, pr_indices_ood = [], [], []
        p_clean_metric, p_ood_metric = [], []
        label_recorder = []
        num_pll_top1_match_id, num_pll_topk_match_id = 0, 0
        num_pll_top1_match_ood, num_pll_topk_match_ood = 0, 0

        q_model.train()
        adjust_lr(optim, lr_plan[epoch])
        if epoch < cfg.warmup_epochs and cfg.warmup_lr_plan == 'epoch_linear':
            adjust_lr(optim, min(1, (epoch+1)/cfg.warmup_epochs) * lr_plan[epoch])
        optim.zero_grad()

        curr_lr = [group['lr'] for group in optim.param_groups][0]
        topK = max(1, int(cfg.topK * 0.5 ** ((epoch - cfg.warmup_epochs) // cfg.topK_decay))) if epoch >= cfg.warmup_epochs and cfg.topK_decay > 0 else cfg.topK
        logger.debug(f'----\nEpoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  Lr:[{curr_lr:.3e}]  topK:[{topK}]')
        threshold_writer.write(f'{epoch+1},{tau_c.mean().item():.5f},{tau_o.mean().item():.5f}')
        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        pbar = tqdm(train_loader, ncols=100, ascii=' >', leave=False, desc=f'TRAINING') if cfg.enable_progress_bar else train_loader
        for it, sample in enumerate(pbar):
            iter_start = time.time()
            if cfg.enable_progress_bar: pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')

            optim.zero_grad()
            indices = sample['index']
            x1, x2 = sample['data']
            x1, x2 = x1.to(device), x2.to(device)
            y = sample['label'].to(device)
            ob_labels = get_smoothed_label_distribution(y, cfg.n_classes, cfg.eps)  # > (bs, nc)
            onehot_labels = F.one_hot(y, cfg.n_classes).float()
            bs = x1.size(0)

            with autocast(cfg.use_fp16):
                logits1, feat1 = q_model(x1)
                logits2, feat2 = q_model(x2)
                q = feat1
                probs1, probs2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
                with torch.no_grad():
                    ema_logits1, ema_feat1 = k_model(x1)
                    # ema_logits2, ema_feat2 = k_model(x2)
                    k = ema_feat1

                # >>>>>>>> Warmup Stage <<<<<<<<
                if epoch < cfg.warmup_epochs:
                    if 'warmup_iterations' in cfg.keys() and cfg.warmup_iterations is not None and it > cfg.warmup_iterations: break
                    if 'warmup_iterations' in cfg.keys() and cfg.warmup_iterations is not None and cfg.warmup_lr_plan == 'iter_linear':
                        adjust_lr(optim, min(1, (it+1)/cfg.warmup_iterations) * lr_plan[epoch])
                    with torch.no_grad():
                        probs1, probs2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
                        prob_clean = 1 - js_div(probs1, ob_labels)
                        prob_ood = js_div(F.softmax(logits1 / 0.1, dim=1), F.softmax(logits2 / 0.1, dim=1))

                        p_clean_metric.extend(prob_clean.clone().detach().cpu().numpy().tolist())
                        p_ood_metric.extend(prob_ood.clone().detach().cpu().numpy().tolist())
                        label_recorder.extend(y.clone().detach().cpu().numpy().tolist())

                    loss = F.cross_entropy(logits1, ob_labels, reduction='mean')
                # >>>>>>>> JoSNC Stage <<<<<<<<
                else:
                    batch_tau_c = tau_c[y]  # (bs, )
                    batch_tau_o = tau_o[y]  # (bs, )
                    selection_results = samples_identification(logits1, logits2, ob_labels, q, queue_keys.clone().detach(),
                                                               queue_logits.clone().detach(), batch_tau_c, batch_tau_o, cfg, logger)
                    idx_clean, idx_id, idx_ood, batch_p_clean, batch_p_ood = selection_results

                    p_clean_metric.extend(batch_p_clean.clone().detach().cpu().numpy().tolist())
                    p_ood_metric.extend(batch_p_ood.clone().detach().cpu().numpy().tolist())
                    label_recorder.extend(y.clone().detach().cpu().numpy().tolist())

                    pll_labelsets = ob_labels.clone().detach()
                    with torch.no_grad():
                        soft_labels = F.softmax(ema_logits1, dim=1)
                        if 1 < topK < cfg.n_classes:
                            _, topK_indices1 = soft_labels.topk(1, dim=1, largest=True, sorted=True)     # top1
                            num_pll_top1_match_id += count_topk_label_matches(topK_indices1, idx_id, indices, gt_train_labels)
                            num_pll_top1_match_ood += count_topk_label_matches(topK_indices1, idx_ood, indices, gt_train_labels)
                            topK_probs, topK_indices1 = soft_labels.topk(topK, dim=1, largest=True, sorted=True)  # topK
                            num_pll_topk_match_id += count_topk_label_matches(topK_indices1, idx_id, indices, gt_train_labels)
                            num_pll_topk_match_ood += count_topk_label_matches(topK_indices1, idx_ood, indices, gt_train_labels)
                            topK_conf = topK_probs.sum(dim=1)

                            estimated_labelsets1 = generate_label_sets(topK_indices1, cfg.n_classes)
                            soft_labels1 = soft_labels * estimated_labelsets1 / cfg.temp + soft_labels * torch.logical_not(estimated_labelsets1)
                            # logger.debug(f'  |- {soft_labels1.topk(topK+3, dim=1, largest=True, sorted=True)[0].mean(dim=0).data}')
                            soft_labels1 = F.softmax(soft_labels1, dim=1)
                            # logger.debug(f'  |- {soft_labels1.topk(topK+3, dim=1, largest=True, sorted=True)[0].mean(dim=0).data}')
                            pll_labelsets[idx_id] = soft_labels1[idx_id]
                        else:
                            topK_conf = soft_labels.max(dim=1)[0]
                            pll_labelsets[idx_id] = soft_labels[idx_id]
                        pll_labelsets[idx_ood] = F.softmax(soft_labels[idx_ood] / 10, dim=1)
                        least_scores, false_labels = soft_labels.min(dim=1)  # Last1  (bs, ), (bs, )
                        false_labels = F.one_hot(false_labels, cfg.n_classes)

                    # classification loss
                    # clean samples
                    losses_cls_clean = F.cross_entropy(logits1[idx_clean], pll_labelsets[idx_clean], reduction='none') * 0.5 + \
                                       F.cross_entropy(logits2[idx_clean], pll_labelsets[idx_clean], reduction='none') * 0.5
                    # ID noisy samples
                    losses_cls_id = id_loss_func(logits1[idx_id], pll_labelsets[idx_id], reduction='none') * 0.5 + \
                                    id_loss_func(logits2[idx_id], pll_labelsets[idx_id], reduction='none') * 0.5
                    losses_cls_id = losses_cls_id * torch.sqrt(topK_conf[idx_id])
                    # OOD noisy samples
                    if cfg.cls4ood == 'josrc':
                        losses_cls_ood = id_loss_func(logits1[idx_ood], pll_labelsets[idx_ood], reduction='none') * 0.5 + \
                                         id_loss_func(logits2[idx_ood], pll_labelsets[idx_ood], reduction='none') * 0.5
                        losses_cls_ood = losses_cls_ood * torch.sqrt(topK_conf[idx_ood])
                    elif cfg.cls4ood == 'nl':
                        losses_cls_ood = negative_cross_entropy_loss(logits1[idx_ood], false_labels[idx_ood], reduction='none') * 0.5 + \
                                         negative_cross_entropy_loss(logits2[idx_ood], false_labels[idx_ood], reduction='none') * 0.5
                        losses_cls_ood = losses_cls_ood * torch.square(1-least_scores[idx_ood])
                    else:
                        raise AssertionError(f'cls4ood: {cfg.cls4ood} is not supported!')
                    losses_pll_all = torch.cat((losses_cls_clean, losses_cls_id, losses_cls_ood), dim=0)
                    loss_cls = losses_pll_all.mean()

                    # feature contrastive loss (MoCo)
                    contrastive_embedding_pool = torch.cat((q, k, queue_keys.clone().detach()), dim=0)
                    loss_con_feat = loss_contrastive_func(features=contrastive_embedding_pool, mask=None, batch_size=bs) if cfg.gamma > 0 else torch.tensor(0).to(device)

                    # prediction consistency loss
                    idx_non_ood = torch.cat((idx_clean, idx_id), dim=0)
                    losses_con_pred_all = symmetric_kl_div(probs1, probs2)
                    losses_con_pred_all = losses_con_pred_all[idx_non_ood]
                    loss_con_pred = losses_con_pred_all.mean() if cfg.alpha > 0 else torch.tensor(0).to(device)

                    # NCR loss
                    loss_ncr = ncr_loss(logits1[idx_non_ood], q[idx_non_ood], queue_logits.clone().detach(), queue_keys.clone().detach(), cfg.n_neighbors, loss_func=cfg.ncr_lossfunc) if cfg.beta > 0 else torch.tensor(0).to(device)

                    # assert not check_nan_inf(losses_cls_clean)
                    # assert not check_nan_inf(losses_cls_id)
                    # assert not check_nan_inf(losses_cls_ood)
                    # assert not check_nan_inf(loss_cls), f'loss_cls: {loss_cls.item()}'
                    # assert not check_nan_inf(loss_con_feat), f'loss_con_feat: {loss_con_feat.item()}'
                    # assert not check_nan_inf(loss_con_pred), f'loss_con_pred: {loss_con_pred.item()}'
                    # assert not check_nan_inf(loss_ncr), f'loss_ncr: {loss_ncr.item()}'

                    # final loss
                    loss = loss_cls + cfg.alpha * loss_con_pred + cfg.gamma * loss_con_feat + cfg.beta * loss_ncr
                    josnc_loss = loss

                    if cfg.part_ce:
                        part_ce_group = 'clean'
                        part_ce_loss = loss.new_tensor(0.0)
                        part_ce_batch = {'num_selected': int(idx_clean.numel()), 'num_valid': 0}
                        if idx_clean.numel() > 0:
                            cam_model = k_model if cfg.part_ce_use_teacher_cam else q_model
                            part_ce_batch = build_local_part_batch(
                                cam_model, x1, y, idx_clean,
                                cam_quantile=cfg.local_evidence_cam_quantile,
                                min_area=cfg.local_evidence_min_area,
                                max_area=cfg.local_evidence_max_area,
                                bbox_padding=cfg.local_evidence_bbox_padding,
                                cam_type=cfg.local_evidence_cam_type,
                            )
                            if part_ce_batch['num_valid'] > 0 and part_ce_batch['x_part'].numel() > 0:
                                # B1/C1 共用局部图生成；C1 未到启动 epoch 时不退化成 B1。
                                if cfg.part_ce_gate:
                                    if (epoch + 1) >= cfg.part_ce_gate_start_epoch:
                                        gate_mask, gate_threshold = build_gate_mask(
                                            part_ce_batch['evidence_score'],
                                            gate_type=cfg.part_ce_gate_type,
                                            threshold=cfg.part_ce_gate_threshold,
                                            keep_ratio=cfg.part_ce_gate_keep_ratio,
                                        )
                                    else:
                                        gate_mask = torch.zeros(
                                            part_ce_batch['num_valid'],
                                            device=x1.device,
                                            dtype=torch.bool,
                                        )
                                        gate_threshold = part_ce_batch['evidence_score'].new_tensor(0.0)
                                else:
                                    gate_mask = torch.ones(
                                        part_ce_batch['num_valid'],
                                        device=x1.device,
                                        dtype=torch.bool,
                                    )
                                    gate_threshold = part_ce_batch['evidence_score'].new_tensor(0.0)
                                part_ce_batch['gate_mask'] = gate_mask
                                part_ce_batch['gate_threshold'] = gate_threshold

                                num_gated = int(gate_mask.sum().item())
                                if num_gated > 1:
                                    # 只有实际通过门控的局部图进入 student 前向和 CE 反传。
                                    student_was_training = q_model.training
                                    q_model.train()
                                    try:
                                        logits_part = q_model(part_ce_batch['x_part'][gate_mask])[0]
                                        labels_part = part_ce_batch['labels'][gate_mask]
                                        part_ce_loss = F.cross_entropy(logits_part, labels_part)
                                    finally:
                                        if not student_was_training:
                                            q_model.eval()
                                    loss = loss + cfg.part_ce_weight * part_ce_loss
                                elif num_gated == 1:
                                    # 单样本会触发 BatchNorm1d 训练模式约束，跳过并清空实际 CE gate。
                                    part_ce_batch['gate_mask'] = torch.zeros_like(gate_mask)
                        if part_ce_writer is not None:
                            part_ce_row = build_part_ce_log_row(
                                epoch + 1, it, part_ce_group, part_ce_batch,
                                josnc_loss, part_ce_loss, cfg.part_ce_weight
                            )
                            part_ce_writer.write(format_part_ce_row(part_ce_row))
                        if part_ce_gate_sample_writer is not None and part_ce_batch['num_valid'] > 0:
                            # C1 逐样本 gate 日志记录实际参与 CE 的 gate，用于后续分析长期过滤样本。
                            gate_sample_rows = build_part_ce_gate_sample_rows(
                                epoch + 1, it, part_ce_group, part_ce_batch, indices,
                                student_logits=logits1,
                            )
                            for row in gate_sample_rows:
                                part_ce_gate_sample_writer.write(format_part_ce_gate_sample_row(row))

                    if cfg.id_candidate:
                        id_candidate_loss = loss.new_tensor(0.0)
                        id_candidate_base_loss = loss.detach()
                        id_candidate_effective_weight = compute_id_candidate_effective_weight(
                            cfg.id_candidate_weight,
                            epoch + 1,
                            cfg.id_candidate_start_epoch,
                            decay_start_epoch=cfg.id_candidate_decay_start_epoch,
                            decay_end_epoch=cfg.id_candidate_decay_end_epoch,
                            min_weight=cfg.id_candidate_min_weight,
                        )
                        id_candidate_batch = {
                            'num_selected': int(idx_id.numel()),
                            'num_valid': 0,
                            'candidate_topk': cfg.id_candidate_topk,
                            'effective_id_candidate_weight': id_candidate_effective_weight,
                        }
                        if idx_id.numel() > 0 and (epoch + 1) >= cfg.id_candidate_start_epoch:
                            id_candidate_batch = build_id_candidate_batch(
                                k_model, x1, y, idx_id,
                                candidate_topk=cfg.id_candidate_topk,
                                cam_target=cfg.id_candidate_cam_target,
                                score_type=cfg.id_candidate_score_type,
                                include_noisy_label=cfg.id_candidate_include_noisy_label,
                                cam_quantile=cfg.local_evidence_cam_quantile,
                                min_area=cfg.local_evidence_min_area,
                                max_area=cfg.local_evidence_max_area,
                                bbox_padding=cfg.local_evidence_bbox_padding,
                                cam_type=cfg.local_evidence_cam_type,
                                max_top1_prob=cfg.id_candidate_max_top1_prob,
                            )
                            id_candidate_batch['effective_id_candidate_weight'] = id_candidate_effective_weight
                            if id_candidate_batch['num_valid'] > 0:
                                # C2-v4 只让 teacher top1 不过强且当前权重非零的样本进入 ID candidate loss。
                                loss_mask = id_candidate_batch['conf_gate'].to(device=x1.device, dtype=torch.bool)
                                if id_candidate_effective_weight <= 0.0:
                                    loss_mask = torch.zeros_like(loss_mask)
                                id_candidate_batch['used_in_loss'] = loss_mask
                                if int(loss_mask.sum().item()) > 0:
                                    candidate_positions = id_candidate_batch['batch_indices'][loss_mask]
                                    candidate_mask = id_candidate_batch['candidate_mask'][loss_mask]
                                    candidate_indices = id_candidate_batch['candidate_indices'][loss_mask]
                                    candidate_scores = id_candidate_batch['candidate_scores'][loss_mask]
                                    candidate_size = id_candidate_batch['candidate_size'][loss_mask]
                                    candidate_labels = id_candidate_batch['labels'][loss_mask]
                                    id_candidate_result1 = compute_id_candidate_loss(
                                        logits1[candidate_positions], candidate_mask,
                                        candidate_indices=candidate_indices,
                                        candidate_scores=candidate_scores,
                                        candidate_size=candidate_size,
                                        labels=candidate_labels,
                                        loss_type=cfg.id_candidate_loss_type,
                                        entropy_weight=cfg.id_candidate_entropy_weight,
                                        entropy_min_ratio=cfg.id_candidate_entropy_min_ratio,
                                        dist_weight=cfg.id_candidate_dist_weight,
                                        target_temp=cfg.id_candidate_target_temp,
                                        top1_cap=cfg.id_candidate_top1_cap,
                                        noisy_prior=cfg.id_candidate_noisy_prior,
                                    )
                                    id_candidate_result2 = compute_id_candidate_loss(
                                        logits2[candidate_positions], candidate_mask,
                                        candidate_indices=candidate_indices,
                                        candidate_scores=candidate_scores,
                                        candidate_size=candidate_size,
                                        labels=candidate_labels,
                                        loss_type=cfg.id_candidate_loss_type,
                                        entropy_weight=cfg.id_candidate_entropy_weight,
                                        entropy_min_ratio=cfg.id_candidate_entropy_min_ratio,
                                        dist_weight=cfg.id_candidate_dist_weight,
                                        target_temp=cfg.id_candidate_target_temp,
                                        top1_cap=cfg.id_candidate_top1_cap,
                                        noisy_prior=cfg.id_candidate_noisy_prior,
                                    )
                                    id_candidate_sample_loss = 0.5 * (
                                        id_candidate_result1['losses'] + id_candidate_result2['losses']
                                    )
                                    id_candidate_loss = id_candidate_sample_loss.mean()
                                    attach_id_candidate_loss_results(
                                        id_candidate_batch, id_candidate_result1, id_candidate_result2, loss_mask
                                    )
                                    loss = loss + id_candidate_effective_weight * id_candidate_loss
                        if id_candidate_writer is not None:
                            id_candidate_row = build_id_candidate_log_row(
                                epoch + 1, it, id_candidate_batch,
                                id_candidate_base_loss, id_candidate_loss,
                                cfg.id_candidate_weight,
                                id_candidate_effective_weight
                            )
                            id_candidate_writer.write(format_id_candidate_row(id_candidate_row))
                        if id_candidate_sample_writer is not None and id_candidate_batch['num_valid'] > 0:
                            # C2 样本日志记录候选集合和 noisy label 关系，支撑闭集错标修正分析。
                            id_candidate_sample_rows = build_id_candidate_sample_rows(
                                epoch + 1, it, id_candidate_batch, indices,
                                student_logits=logits1,
                            )
                            for row in id_candidate_sample_rows:
                                id_candidate_sample_writer.write(format_id_candidate_sample_row(row))

                    l1 = losses_cls_clean.mean().clone().detach().item() if idx_clean.size(0) > 0 else 0.000
                    l2 = losses_cls_id.mean().clone().detach().item() if idx_id.size(0) > 0 else 0.000
                    l3 = losses_cls_ood.mean().clone().detach().item() if idx_ood.size(0) > 0 else 0.000
                    logger.debug(f'  |- cls_clean: {l1:.3f}, cls_id: {l2:.6f}, cls_ood: {l3:.3f}, '
                                 f'con_feat: {loss_con_feat.item():.3f}, con_pred: {loss_con_pred.item():.3f}, ncr: {loss_ncr.item():.3f}')

                    pr_indices_clean.extend(select_batch_indices(indices, idx_clean))
                    pr_indices_id.extend(select_batch_indices(indices, idx_id))
                    pr_indices_ood.extend(select_batch_indices(indices, idx_ood))

                    if local_evidence_writer is not None and _should_run_local_evidence(cfg, epoch, it):
                        # A1 只读诊断：使用 student/teacher 的当前输出记录证据，不改变原始 loss。
                        diagnostic_model = k_model if cfg.local_evidence_use_teacher else q_model
                        local_evidence_rows = compute_local_evidence(
                            diagnostic_model, x1, y, indices, idx_clean, idx_id, idx_ood,
                            epoch=epoch + 1,
                            batch_idx=it,
                            cam_quantile=cfg.local_evidence_cam_quantile,
                            min_area=cfg.local_evidence_min_area,
                            max_area=cfg.local_evidence_max_area,
                            bbox_padding=cfg.local_evidence_bbox_padding,
                            cam_type=cfg.local_evidence_cam_type,
                            student_logits=logits1,
                            teacher_logits=ema_logits1,
                            save_images=cfg.local_evidence_save_images,
                            image_dir=os.path.join(local_evidence_image_dir, f'epoch_{epoch + 1:03d}') if local_evidence_image_dir is not None else None,
                            image_max_samples=cfg.local_evidence_image_max_samples,
                            norm_mean=local_evidence_norm[0] if local_evidence_norm is not None else None,
                            norm_std=local_evidence_norm[1] if local_evidence_norm is not None else None,
                        )
                        for row in local_evidence_rows:
                            local_evidence_writer.write(format_local_evidence_row(row))

                # dequeue and enqueue
                if queue_ptr + bs > cfg.queue_length:  # if last interation in each epoch is a small batch
                    n_tailing = cfg.queue_length - queue_ptr
                    n_heading = bs - n_tailing
                    queue_keys[queue_ptr:, :] = k[:n_tailing, :].clone().detach()
                    queue_keys[:n_heading, :] = k[n_tailing:, :].clone().detach()
                    queue_logits[queue_ptr:, :] = logits1[:n_tailing, :].clone().detach()
                    queue_logits[:n_heading, :] = logits1[n_tailing:, :].clone().detach()
                else:
                    queue_keys[queue_ptr: queue_ptr + bs, :] = k.clone().detach()
                    queue_logits[queue_ptr: queue_ptr + bs, :] = logits1.clone().detach()
                queue_ptr = (queue_ptr + bs) % cfg.queue_length

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            momentum_update_key_network(q_model, k_model, cfg.knet_m)

            train_acc = accuracy(logits1, y, topk=(1,))
            train_accuracy_meter.update(train_acc[0], bs)
            train_loss_meter.update(loss.item(), bs)
            epoch_train_time.update((time.time() - iter_start), 1)
            if ((it + 1) % LOG_FREQ == 0) or (it + 1 == len(train_loader)):
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy_meter.avg:6.2f}]  " \
                                  f"Train Loss:[{train_loss_meter.avg:4.4f}]  " \
                                  f"{epoch_train_time.avg:4.0f} sec/iter"
                logger.debug(console_content)

        if cfg.threshold_generator == 'gmm':
            tau_c_tmp = gmm_based_threshold_generation(p_clean_metric, cfg.n_classes).to(device)
            tau_o_tmp = gmm_based_threshold_generation(p_ood_metric, cfg.n_classes).to(device)
        elif cfg.threshold_generator == 'mean':
            tau_c_tmp = mean_based_threshold_generation(p_clean_metric, cfg.n_classes).to(device)
            tau_o_tmp = mean_based_threshold_generation(p_ood_metric, cfg.n_classes).to(device)
        elif cfg.threshold_generator == 'per_class_mean':
            tau_c_tmp = per_class_mean_based_threshold_generation(p_clean_metric, label_recorder, cfg.n_classes).to(device)
            tau_o_tmp = per_class_mean_based_threshold_generation(p_ood_metric, label_recorder, cfg.n_classes).to(device)
        else:
            raise AssertionError(f'threshold_generator')
        if epoch < cfg.warmup_epochs:
            delta = 0.0
            tau_m = 0.75
        else:
            delta = cfg.delta
            tau_m = cfg.tau_m
        tmp_tauc = tau_m * tau_c + (1 - tau_m) * (tau_c_tmp * (1 + delta))
        tmp_tauo = tau_m * tau_o + (1 - tau_m) * (tau_o_tmp * (1 + delta))
        tau_c = torch.where(tmp_tauc > tau_c, tmp_tauc, tau_c)
        tau_o = torch.where(tmp_tauo > tau_o, tmp_tauo, tau_o)
        # if epoch >= 80: tau_c = min(1.001 * tau_c, 0.95)

        if cfg.predefined_tau_clean:
            tau_c_t = make_linear_values(0, cfg.warmup_epochs, 0.75) + make_linear_values(0.75, cfg.epochs-cfg.warmup_epochs, 0.95)
            tau_c_scalar = tau_c_t[epoch+1] if epoch < cfg.epochs-1 else 0.95
            print(f'*** tau_c for next epoch is {tau_c_scalar} (sampled in [0.75, 0.95])')
            tau_c = torch.ones(cfg.n_classes).to(device) * tau_c_scalar

        # save checkpoint
        if cfg.save_ckpt:
            ckpt_file_suffix = f'warmup_{epoch + 1:02d}th_epoch' if epoch < cfg.warmup_epochs and SAVE_WARMUP_CKPT else 'latest'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': q_model.state_dict(),
                'k_model_state_dict': k_model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'queue_keys': queue_keys.detach().cpu(),
                'queue_logits': queue_logits.detach().cpu(),
                'queue_ptr': queue_ptr,
                'tau_c': tau_c.detach().cpu(),
                'tau_o': tau_o.detach().cpu(),
                'best_epoch': best_epoch,
                'best_accuracy': best_accuracy
            }, filename=os.path.join(result_dir, f'checkpoint-{ckpt_file_suffix}.pth'))

        # evaluate this epoch
        if 'webvision' in cfg.dataset:
            imagenet_test_accuracy, imagenet_top5_accuracy  = evaluate(valid_loader, q_model, device, topk=(1, 5), progress_bar=cfg.enable_progress_bar)
            test_accuracy, top5_accuracy = evaluate(test_loader, q_model, device, topk=(1, 5), progress_bar=cfg.enable_progress_bar)
        else:
            test_accuracy = evaluate(test_loader, q_model, device, progress_bar=cfg.enable_progress_bar)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if cfg.save_model:
                torch.save(q_model.state_dict(), f'{result_dir}/model_best.pth')
        if cfg.save_model:
            torch.save(q_model.state_dict(), f'{result_dir}/model_last.pth')

        epoch_runtime = time.time() - epoch_start
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'trainLoss: {train_loss_meter.avg:>6.3f} | '
                    f'trainAcc: {train_accuracy_meter.avg:>6.3f} | '
                    f'testAcc: {test_accuracy:>6.3f} | '
                    f'runtime: {epoch_runtime:4.0f} sec | '
                    f'bestAcc: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt')

        if cfg.eval_det == 1 and epoch >= cfg.warmup_epochs:
            pr_indicator_clean = indices_list_to_indicator_vector(pr_indices_clean, n_train_samples)
            pr_indicator_id = indices_list_to_indicator_vector(pr_indices_id, n_train_samples)
            pr_indicator_ood = indices_list_to_indicator_vector(pr_indices_ood, n_train_samples)
            assert (pr_indicator_clean + pr_indicator_id + pr_indicator_ood == 1).all(), \
                f'{np.intersect1d(pr_indicator_clean, pr_indices_id)}\n{np.intersect1d(pr_indicator_clean, pr_indices_ood)}\n' \
                f'{len(pr_indices_clean)}/{len(pr_indices_id)}/{len(pr_indices_ood)}\n' \
                f'{pr_indices_ood}'
            p_clean, r_clean, f1_clean, auroc_clean = detection_evaluate(pr_indicator_clean, gt_indicator_clean)
            p_id, r_id, f1_id, auroc_id = detection_evaluate(pr_indicator_id, gt_indicator_id)
            p_ood, r_ood, f1_ood, auroc_ood = detection_evaluate(pr_indicator_ood, gt_indicator_ood)
            logger.msg(f'epoch: {epoch + 1:>3d} | '
                       f'clean (N/P/R/F1/AUROC): {len(pr_indices_clean)}/{p_clean:.3f}/{r_clean:.3f}/{f1_clean:.3f}/{auroc_clean:.3f} | '
                       f'id (N/P/R/F1/AUROC): {len(pr_indices_id)}/{p_id:.3f}/{r_id:.3f}/{f1_id:.3f}/{auroc_id:.3f} | '
                       f'ood (N/P/R/F1/AUROC): {len(pr_indices_ood)}/{p_ood:.3f}/{r_ood:.3f}/{f1_ood:.3f}/{auroc_ood:.3f}')
            plot_precision_recall(f'{result_dir}/msg-log.txt')

            pr_metric_writer.write(f'{epoch + 1},'
                                   f'{len(pr_indices_clean)},{p_clean:.3f},{r_clean:.3f},{f1_clean:.3f},{auroc_clean:.3f},'
                                   f'{len(pr_indices_id)},{p_id:.3f},{r_id:.3f},{f1_id:.3f},{auroc_id:.3f},'
                                   f'{len(pr_indices_ood)},{p_ood:.3f},{r_ood:.3f},{f1_ood:.3f},{auroc_ood:.3f}')
        else:
            pr_metric_writer.write(f'{epoch + 1},'
                                   f'{len(pr_indices_clean)},-,-,-,-,'
                                   f'{len(pr_indices_id)},-,-,-,-,'
                                   f'{len(pr_indices_ood)},-,-,-,-')
        if 'webvision' in cfg.dataset:
            test_acc_writer.write(f'{epoch + 1},{test_accuracy:.3f},{top5_accuracy:.3f},{imagenet_test_accuracy:.3f},{imagenet_top5_accuracy:.3f}')
        else:
            test_acc_writer.write(f'{epoch + 1},{test_accuracy:.3f}')
        pll_topk_acc_writer.write(f'{epoch + 1},'
                                  f'{num_pll_top1_match_id/(len(pr_indices_id)+1e-6):.3f},{num_pll_topk_match_id/(len(pr_indices_id)+1e-6):.3f},'
                                  f'{num_pll_top1_match_ood/(len(pr_indices_ood)+1e-6):.3f},{num_pll_topk_match_ood/(len(pr_indices_ood)+1e-6):.3f}')

    wrapup_training_statics(result_dir, best_accuracy)


def check_args(args):
    valid_arg_items = [
        'seed',
        'data_root', 'dataset', 'n_classes', 'rescale_size', 'crop_size', 'noise_type', 'idn_noise_rate', 'ood_noise_rate',
        'arch', 'hdim', 'opt', 'batch_size', 'epochs', 'lr', 'lr_decay', 'warmup_epochs', 'warmup_lr', 'warmup_lr_plan', 'weight_decay',
        'eps', 'alpha', 'beta', 'gamma', 'delta',
        'log_root', 'log_proj', 'log_name', 'ckpt_path', 'enable_progress_bar',
        'warmup_fc_only', 'warmup_iterations',
        'fdim', 'n_neighbors', 'tau_m', 'queue_length', 'knet_m', 'transform', 'topK', 'topK_decay', 'temp',
        'integrate_mode', 'ood_criterion', 'conf_weight', 'threshold_generator',
        'cls4id', 'cls4ood', 'ncr_lossfunc', 'predefined_tau_clean',
        'eval_det', 'use_fp16', 'benchmark', 'ablation', 'save_model', 'save_ckpt',
        'local_evidence', 'local_evidence_every', 'local_evidence_max_batches',
        'local_evidence_cam_quantile', 'local_evidence_use_teacher',
        'local_evidence_min_area', 'local_evidence_max_area',
        'local_evidence_bbox_padding', 'local_evidence_cam_type',
        'local_evidence_save_images', 'local_evidence_image_max_samples',
        'part_ce', 'part_ce_weight', 'part_ce_groups', 'part_ce_use_teacher_cam', 'part_ce_log',
        'part_ce_gate', 'part_ce_gate_type', 'part_ce_gate_threshold',
        'part_ce_gate_keep_ratio', 'part_ce_gate_start_epoch',
        'id_candidate', 'id_candidate_weight', 'id_candidate_topk',
        'id_candidate_start_epoch', 'id_candidate_log',
        'id_candidate_cam_target', 'id_candidate_score_type',
        'id_candidate_include_noisy_label', 'id_candidate_entropy_weight',
        'id_candidate_entropy_min_ratio', 'id_candidate_loss_type',
        'id_candidate_dist_weight', 'id_candidate_target_temp',
        'id_candidate_top1_cap', 'id_candidate_noisy_prior',
        'id_candidate_decay_start_epoch', 'id_candidate_decay_end_epoch',
        'id_candidate_min_weight', 'id_candidate_max_top1_prob'
    ]
    invalid_arg_items = []
    for k in args.keys():
        if k not in valid_arg_items:
            invalid_arg_items.append(k)
    if len(invalid_arg_items) > 0:
        raise AssertionError(f'{invalid_arg_items} is/are not valid arguments!')
    else:
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='configuration file path')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-root', type=str, default=None)
    # Args: network & optimization
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--warmup-fc-only', action='store_true')
    parser.add_argument('--hdim', type=float, default=2)
    parser.add_argument('--fdim', type=int, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr-decay', type=str, default=None)
    parser.add_argument('--warmup-epochs', type=int, default=None)
    parser.add_argument('--warmup-iterations', type=int, default=None)
    parser.add_argument('--warmup-lr', type=float, default=None)
    parser.add_argument('--warmup-lr-plan', type=str, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--use-fp16', type=bool, default=None)
    parser.add_argument('--transform', type=str, default='strong')
    # Args: hyper-params
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.3, help='loss weight for prediction contrastive')
    parser.add_argument('--gamma', type=float, default=0.2, help='loss weight for feature contrastive')
    parser.add_argument('--beta', type=float, default=0.8, help='loss weight for NCR (neighbor consistency regularization)')
    parser.add_argument('--delta', type=float, default=0.0, help='threshold increase factor')
    # Args: logging
    parser.add_argument('--log-proj', type=str, default=None)
    parser.add_argument('--log-name', type=str, default=None)
    parser.add_argument('--enable-progress-bar', type=bool, default=False)
    # Args: checkpoint
    parser.add_argument('--ckpt-path', type=str, default=None)
    # Args: SNC hyper-params
    parser.add_argument('--n-neighbors', type=int, default=10)
    parser.add_argument('--tau-m', type=float, default=0.99)
    # Args: CL hyper-params
    parser.add_argument('--queue-length', type=int, default=32000)
    parser.add_argument('--knet-m', type=float, default=0.99)
    # Args: PLL hyper-params
    parser.add_argument('--topK', type=int, default=5)
    parser.add_argument('--topK-decay', type=int, default=20)
    parser.add_argument('--temp', type=float, default=0.1)

    # Args: Helper
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--save-ckpt', action='store_true')
    parser.add_argument('--eval-det', type=int, default=1)
    parser.add_argument('--cls4ood', type=str, default='nl')
    parser.add_argument('--cls4id', type=str, default='ce')
    parser.add_argument('--ncr-lossfunc', type=str, default='kldiv')
    parser.add_argument('--integrate-mode', type=str, default='or')
    parser.add_argument('--ood-criterion', type=str, default='div')
    parser.add_argument('--threshold-generator', type=str, default='gmm')
    parser.add_argument('--conf-weight', action='store_true')
    parser.add_argument('--predefined-tau-clean', action='store_true')
    # A1 局部证据诊断参数：默认只写 CSV，不额外加入 loss。
    parser.add_argument('--local-evidence', action='store_true', default=None,
                        help='开启 A1 局部证据诊断；只写 CSV/可选图片，不加入额外 loss。')
    parser.add_argument('--local-evidence-every', type=int, default=None,
                        help='A1 每隔多少个 epoch 运行一次；1 表示每轮都运行。')
    parser.add_argument('--local-evidence-max-batches', type=int, default=None,
                        help='每个诊断 epoch 最多处理多少个 batch；0 表示不限制。')
    parser.add_argument('--local-evidence-cam-quantile', type=float, default=None,
                        help='CAM 高响应阈值分位数；0.8 表示取响应最高约 20% 的区域。')
    parser.add_argument('--local-evidence-use-teacher', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用 EMA teacher 生成 CAM 和局部证据；可用 --no-local-evidence-use-teacher 关闭。')
    parser.add_argument('--local-evidence-min-area', type=float, default=None,
                        help='CAM bbox 最小面积占原图比例，防止局部图过小。')
    parser.add_argument('--local-evidence-max-area', type=float, default=None,
                        help='CAM bbox 最大面积占原图比例，防止局部图接近整图。')
    parser.add_argument('--local-evidence-bbox-padding', type=float, default=None,
                        help='CAM bbox 周围扩展比例，用于保留一点上下文。')
    parser.add_argument('--local-evidence-cam-type', type=str, default=None,
                        help='CAM 类型；当前只支持 weightcam。')
    parser.add_argument('--local-evidence-save-images', action='store_true', default=None,
                        help='保存 A1 可视化 PNG：原图+bbox、CAM 叠加、局部图、擦除图。')
    parser.add_argument('--local-evidence-image-max-samples', type=int, default=None,
                        help='每个触发 batch 最多保存多少张 A1 可视化图片。')

    # B1/C1 局部 CE：C1 在 B1 的 clean 局部 CE 前增加 evidence gate。
    parser.add_argument('--part-ce', action='store_true', default=None,
                        help='开启 B1/C1 局部 CE 分支。')
    parser.add_argument('--part-ce-weight', type=float, default=None,
                        help='局部 CE 加到总 loss 的权重。')
    parser.add_argument('--part-ce-groups', type=str, default=None,
                        help='局部 CE 使用哪些 Jo-SNC 分组；当前第一版只支持 clean。')
    parser.add_argument('--part-ce-use-teacher-cam', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用 EMA teacher 生成 CAM/bbox 和 evidence；CE 始终反传到 student。')
    parser.add_argument('--part-ce-log', action=argparse.BooleanOptionalAction, default=None,
                        help='是否写出 part_ce.csv 诊断日志。')

    parser.add_argument('--part-ce-gate', action=argparse.BooleanOptionalAction, default=None,
                        help='开启 C1 clean 局部证据门控；关闭时保持 B1 直接局部 CE。')
    parser.add_argument('--part-ce-gate-type', type=str, default=None,
                        help='C1 门控方式：fixed 或 percentile。')
    parser.add_argument('--part-ce-gate-threshold', type=float, default=None,
                        help='C1 fixed 门控阈值。')
    parser.add_argument('--part-ce-gate-keep-ratio', type=float, default=None,
                        help='C1 percentile 门控保留比例。')
    parser.add_argument('--part-ce-gate-start-epoch', type=int, default=None,
                        help='从第几个用户可见 epoch 开始启用 C1 门控局部 CE。')

    # C2 ID noisy 候选标签：使用 teacher-top1 单 CAM 构造候选集合，再按配置选择 PLL 或 capped soft target。
    parser.add_argument('--id-candidate', action=argparse.BooleanOptionalAction, default=None,
                        help='开启 C2 ID noisy 局部证据候选标签学习分支。')
    parser.add_argument('--id-candidate-weight', type=float, default=None,
                        help='C2 ID candidate loss 加到总 loss 的权重。')
    parser.add_argument('--id-candidate-topk', type=int, default=None,
                        help='C2 每个 ID noisy 样本保留多少个候选类别。')
    parser.add_argument('--id-candidate-start-epoch', type=int, default=None,
                        help='从第几个用户可见 epoch 开始启用 C2 ID candidate loss。')
    parser.add_argument('--id-candidate-log', action=argparse.BooleanOptionalAction, default=None,
                        help='是否写出 id_candidate.csv 和 id_candidate_samples.csv 诊断日志。')
    parser.add_argument('--id-candidate-cam-target', type=str, default=None,
                        help='C2 生成 CAM 的目标；第一版只支持 teacher_top1。')
    parser.add_argument('--id-candidate-score-type', type=str, default=None,
                        help='C2 候选打分方式；第一版只支持 ori_part_minus_erase。')
    parser.add_argument('--id-candidate-include-noisy-label', action=argparse.BooleanOptionalAction, default=None,
                        help='C2-v2 是否把 noisy label 强制并入候选集合，降低早期监督漂移。')
    parser.add_argument('--id-candidate-entropy-weight', type=float, default=None,
                        help='C2-v2 候选集合内熵下界惩罚权重。')
    parser.add_argument('--id-candidate-entropy-min-ratio', type=float, default=None,
                        help='C2-v2 熵下界占 log(|S|) 的比例。')

    parser.add_argument('--id-candidate-loss-type', type=str, default=None,
                        help='C2 loss 类型：pll、pll_entropy 或 capped_soft。')
    parser.add_argument('--id-candidate-dist-weight', type=float, default=None,
                        help='C2 capped soft target 分布约束权重。')
    parser.add_argument('--id-candidate-target-temp', type=float, default=None,
                        help='C2 evidence score 构造目标分布时的 softmax 温度。')
    parser.add_argument('--id-candidate-top1-cap', type=float, default=None,
                        help='C2 目标分布中 candidate top1 的最大质量。')
    parser.add_argument('--id-candidate-noisy-prior', type=float, default=None,
                        help='C2 目标分布中 noisy label 的最小保底质量。')
    parser.add_argument('--id-candidate-decay-start-epoch', type=int, default=None,
                        help='C2-v4 从第几个用户可见 epoch 开始线性衰减 ID candidate loss 权重。')
    parser.add_argument('--id-candidate-decay-end-epoch', type=int, default=None,
                        help='C2-v4 到第几个用户可见 epoch 衰减到 id_candidate_min_weight。')
    parser.add_argument('--id-candidate-min-weight', type=float, default=None,
                        help='C2-v4 后期衰减后的最小 ID candidate loss 权重。')
    parser.add_argument('--id-candidate-max-top1-prob', type=float, default=None,
                        help='C2-v4 teacher 原图 top1 prob 高于等于该阈值时跳过 ID candidate loss。')

    parsed_args = parser.parse_args()
    cfg_path = parsed_args.cfg
    gpu = parsed_args.gpu
    parsed_args = {k: v for k, v in vars(parsed_args).items() if v is not None and k not in ['cfg', 'gpu']}
    # 配置文件按 UTF-8 读取，避免 Windows 默认 GBK 环境遇到中文注释时报解码错误。
    with open(cfg_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args.update(parsed_args)
    # fdim 优先使用 YAML 或命令行显式值，仅在配置缺省时补默认值。
    if args.get('fdim') is None:
        args['fdim'] = 256
    # A1 参数同样先尊重 YAML/命令行，缺省时再统一补默认值。
    args.setdefault('local_evidence', False)
    args.setdefault('local_evidence_every', 1)
    args.setdefault('local_evidence_max_batches', 0)
    args.setdefault('local_evidence_cam_quantile', 0.8)
    args.setdefault('local_evidence_use_teacher', True)
    args.setdefault('local_evidence_min_area', 0.05)
    args.setdefault('local_evidence_max_area', 0.7)
    args.setdefault('local_evidence_bbox_padding', 0.05)
    args.setdefault('local_evidence_cam_type', 'weightcam')
    args.setdefault('local_evidence_save_images', False)
    args.setdefault('local_evidence_image_max_samples', 8)
    # B1/C1 默认关闭；开启后无需同时开启 A1 local_evidence，也会独立写 part_ce.csv。
    args.setdefault('part_ce', False)
    args.setdefault('part_ce_weight', 0.5)
    args.setdefault('part_ce_groups', 'clean')
    args.setdefault('part_ce_use_teacher_cam', True)
    args.setdefault('part_ce_log', True)
    args.setdefault('part_ce_gate', False)
    args.setdefault('part_ce_gate_type', 'percentile')
    args.setdefault('part_ce_gate_threshold', 0.10)
    args.setdefault('part_ce_gate_keep_ratio', 0.50)
    args.setdefault('part_ce_gate_start_epoch', 20)
    # C2 默认关闭；启用后可在 PLL、熵约束、capped soft target 和 v4 衰减过滤间组合。
    args.setdefault('id_candidate', False)
    args.setdefault('id_candidate_weight', 0.3)
    args.setdefault('id_candidate_topk', 5)
    args.setdefault('id_candidate_start_epoch', 20)
    args.setdefault('id_candidate_log', True)
    args.setdefault('id_candidate_cam_target', 'teacher_top1')
    args.setdefault('id_candidate_score_type', 'ori_part_minus_erase')
    args.setdefault('id_candidate_include_noisy_label', True)
    args.setdefault('id_candidate_entropy_weight', 0.02)
    args.setdefault('id_candidate_entropy_min_ratio', 0.50)
    args.setdefault('id_candidate_loss_type', 'pll_entropy')
    args.setdefault('id_candidate_dist_weight', 0.5)
    args.setdefault('id_candidate_target_temp', 2.0)
    args.setdefault('id_candidate_top1_cap', 0.5)
    args.setdefault('id_candidate_noisy_prior', 0.05)
    args.setdefault('id_candidate_decay_start_epoch', 0)
    args.setdefault('id_candidate_decay_end_epoch', 0)
    args.setdefault('id_candidate_min_weight', 0.0)
    args.setdefault('id_candidate_max_top1_prob', 1.0)
    # C1/C2 字符串参数统一小写，避免 YAML/命令行大小写差异导致误判。
    args['part_ce_gate_type'] = str(args['part_ce_gate_type']).lower()
    args['id_candidate_cam_target'] = str(args['id_candidate_cam_target']).lower()
    args['id_candidate_score_type'] = str(args['id_candidate_score_type']).lower()
    args['id_candidate_loss_type'] = str(args['id_candidate_loss_type']).lower()
    assert check_args(args)
    return gpu, edict(args)


if __name__ == '__main__':
    igpu, params = parse_args()
    script_start_time = time.time()
    print(params)
    main(igpu, params)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime // 3600:.0f} hours {script_runtime % 3600 / 60:.0f} minutes')
