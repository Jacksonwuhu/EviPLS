from __future__ import print_function
import argparse
import pandas as pd
import os
import os.path as osp
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataloader import LLP_dataset, ToTensor, categories
from nets.net_audiovisual import NREP, LabelSmoothingNCELoss
from utils.eval_metrics import segment_level, event_level, print_overall_metric

from utils.logger import get_logger
from openpyxl import load_workbook
from nets.criterion import EvidenceLoss, MutualLearningLoss
from nrep.data import build_llp_dataloaders
from nrep.checkpoints import build_training_state
from nrep.ema import ModelEMA
from nrep.paths import project_path
from nrep.runtime import capture_random_state, seed_everything
from nrep.uncertainty import compute_uncertainty, hard_negative_mining
from tqdm import tqdm
import datetime
import pickle as pkl
import pdb

exp_logger = get_logger()


def risk_proxy_from_logits(
    logits: torch.Tensor,
    proxy: str = "edl",
    *,
    evidence_type: str = "exp",
    prior_strength: float = 1.0,
) -> torch.Tensor:
    """Return a bounded high-value-is-risk score for binary logits.

    The legacy EDL/vacuity path remains available through ``proxy='edl'``.
    Entropy, margin and random controls are exposed only for the frozen
    risk-proxy ablations; none of them changes the predictive target ``p``.
    """
    if logits.ndim == 0 or logits.shape[-1] != 2:
        raise ValueError(f"expected binary logits[..., 2], got {tuple(logits.shape)}")
    if proxy in {"edl", "vacuity"}:
        return compute_uncertainty(
            logits, evidence_type, prior_strength=prior_strength
        ).clamp(0.0, 1.0)
    probabilities = torch.softmax(logits, dim=-1)
    if proxy in {"entropy", "softmax_entropy"}:
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        return (entropy / np.log(2.0)).clamp(0.0, 1.0)
    if proxy in {"margin", "softmax_margin"}:
        return (1.0 - (probabilities[..., 0] - probabilities[..., 1]).abs()).clamp(
            0.0, 1.0
        )
    if proxy == "random":
        return torch.rand(logits.shape[:-1], device=logits.device, dtype=logits.dtype)
    raise ValueError(f"unsupported risk proxy: {proxy!r}")


def get_LLP_dataloader(args):
    return build_llp_dataloaders(args)


def set_random_seed(seed):
    seed_everything(seed)


def get_random_state():
    return capture_random_state()


def train_with_pseudo_labels(args, models, train_loader, optimizers, criterion, epoch, logger):
    args.device = getattr(args, "device", str(next(models[0].parameters()).device))
    exp_logger(f"begin train_with_pseudo_labels.")
    for model in models:
        model.train()

    criterion2 = LabelSmoothingNCELoss(classes=10, smoothing=args.nce_smooth)
    criterion3 = MutualLearningLoss()
    
    # 依然使用我们修正好的 EvidenceLoss
    use_evidence_loss = (
        getattr(args, 'loss_semantics', 'legacy') == 'paper'
        or args.uncertain
    )
    criterion_edl = EvidenceLoss(
        num_classes=2,
        evidence=args.evidence,
        loss_type=args.loss_type,
        uncertain=use_evidence_loss,
        annealing_step=args.epochs,
        brier_weight=args.brier_weight,
        temperature=args.edl_temperature,
        legacy_normalization=getattr(args, 'loss_semantics', 'legacy') == 'legacy',
    )
    criterion_risk = EvidenceLoss(
        num_classes=2,
        evidence=args.evidence,
        loss_type=args.loss_type,
        uncertain=use_evidence_loss,
        annealing_step=args.epochs,
        temperature=args.edl_temperature,
        legacy_normalization=getattr(args, 'loss_semantics', 'legacy') == 'legacy',
        risk_weighting=args.risk_loss,
        risk_weight_floor=args.risk_weight_floor,
        risk_weight_gamma=args.risk_weight_gamma,
        risk_weight_warmup=args.risk_weight_warmup,
        risk_smooth_max=args.risk_smooth_max,
    )
    ps_criterion = criterion_risk if args.risk_loss != 'none' else criterion_edl
    mel_criterion = (
        criterion_risk
        if args.risk_loss != 'none' and args.risk_loss_scope == 'ps_mel'
        else criterion_edl
    )

    for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training with pseudo labels epoch {}".format(epoch)):
        # 1. 获取原始数据 (Original Data)
        audio, video, video_st, target = sample['audio'].to(args.device), \
                                         sample['video_s'].to(args.device), \
                                         sample['video_st'].to(args.device), \
                                         sample['label'].type(torch.FloatTensor).to(args.device)
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to(args.device), sample['Pv'].type(torch.FloatTensor).to(args.device)
        
        audio_pseudo_labels = sample['audio_pseudo_labels'].float().to(args.device)
        visual_pseudo_labels = sample['visual_pseudo_labels'].float().to(args.device)

        # =======================================================
        # ������ 创新点：跨模态混合增强 (Cross-Modal Mixing)
        # 参考论文 Section 3.2 "Exchanging Audio and Visual Tracks"
        # =======================================================
        
        # U-HNM: 不确定性驱动的难例挖掘 (Pass 1后选择)
        bsz = audio.size(0)
        idx = torch.randperm(bsz).to(args.device)
        
        # 构造“混合”音频 (Audio from Video B)
        audio_mixed = audio[idx]
        
        # 构造“混合”后的目标标签
        # Visual分支 应该依然预测原始 target (因为 Video 没变)
        # Audio分支 应该预测 target[idx] (因为 Audio 换成了 Video B 的)
        target_mixed_audio = target[idx] 
        Pa_mixed = Pa[idx]
        audio_pseudo_labels_mixed = audio_pseudo_labels[idx]
        
        # =======================================================

        for optimizer in optimizers:
            optimizer.zero_grad()

        if epoch <= args.epochs:
            # === Pass 1: 原始数据训练 (保持不变) ===
            global_logits, a_logits, v_logits, frame_logits, temporal_logits, frame_att, temporal_att, sims_after, mask_after = models[0](audio, video, video_st, with_ca=True, epoch_num=epoch-1)
            
            # --- 原始 Loss 计算 ---
            a_frame_logits = frame_logits[:,:,0,:,:]
            v_frame_logits = frame_logits[:,:,1,:,:]
            fbcl_loss = args.lamda_1*criterion2(sims_after, mask_after) + args.lamda_2*(criterion3(temporal_att[..., 0], frame_att[..., 0].max(dim=-2)[0]) + criterion3(temporal_att[..., 1], frame_att[..., 1].max(dim=-2)[0]))

            frame_target = audio_pseudo_labels * visual_pseudo_labels * target.unsqueeze(dim=1).expand(Pa.shape[0], 10, 25)
            
            global_sl_loss = criterion_edl(global_logits, target, epoch_num=epoch)
            temporal_sl_loss = criterion_edl(temporal_logits, frame_target, epoch_num=epoch) 
            tel_loss = global_sl_loss + temporal_sl_loss

            a_frame_sl_loss = ps_criterion(a_frame_logits, audio_pseudo_labels, fore_only=True, epoch_num=epoch)
            v_frame_sl_loss = ps_criterion(v_frame_logits, visual_pseudo_labels, fore_only=True, epoch_num=epoch)
            ps_loss = args.audio_factor * a_frame_sl_loss + v_frame_sl_loss

            audio_sl_loss = mel_criterion(a_logits, Pa, epoch_num=epoch)
            video_sl_loss = mel_criterion(v_logits, Pv, epoch_num=epoch)
            mel_loss = args.audio_factor * audio_sl_loss + video_sl_loss
            
            loss_original = ps_loss + args.lamda_3*mel_loss + args.lamda_4*tel_loss + fbcl_loss


            # === Cross-modal mixing scheduler: random / similarity / U-HNM ===
            # ``random`` preserves the historical quality-agnostic control.
            # ``similarity`` isolates feature similarity, and ``uhnm`` adds the
            # U-HNM adds the evidence-uncertainty factor.
            mixing = getattr(args, 'mixing', 'uhnm')
            if mixing == 'similarity':
                idx = hard_negative_mining(audio, torch.ones(bsz, device=audio.device))
            elif mixing == 'uhnm':
                prior_strength = 2.0 if getattr(args, 'loss_semantics', 'legacy') == 'paper' else 1.0
                u_a = compute_uncertainty(
                    a_logits.detach(), args.evidence, prior_strength=prior_strength
                )
                idx = hard_negative_mining(audio, u_a.mean(dim=-1))
            elif mixing != 'random':
                raise ValueError(f"Unsupported mixing strategy: {mixing!r}")
            audio_mixed = audio[idx]
            Pa_mixed = Pa[idx]
            audio_pseudo_labels_mixed = audio_pseudo_labels[idx]

            # === Pass 2: ������ 混合数据增强训练 (Mixing Augmentation) ===
            # 将 (Video_A, Audio_B) 输入模型
            # 注意：with_ca=True 意味着让 Cross-Attention 去处理“不匹配”的特征，这是最好的训练！
            _, a_logits_mix, v_logits_mix, frame_logits_mix, _, _, _, _, _ = models[0](audio_mixed, video, video_st, with_ca=True, epoch_num=epoch-1)
            
            # 1. Audio 分支必须预测 mixed 标签 (target[idx])
            # 这强迫 Audio 分支完全忽略 Video (因为 Video 是错的)
            loss_audio_mix = criterion_edl(a_logits_mix, Pa_mixed, epoch_num=epoch)
            
            # 2. Video 分支必须预测原始标签 (target)
            # 这强迫 Video 分支完全忽略 Audio (因为 Audio 是错的)
            loss_video_mix = criterion_edl(v_logits_mix, Pv, epoch_num=epoch)
            
            # 3. 总 Loss = 原始 Loss + 0.5 * 混合 Loss
            # 系数 0.5 可以调整，避免主任务被干扰
            loss = loss_original + 0.5 * loss_video_mix + 0.0 * loss_audio_mix

            loss.backward()
            optimizers[0].step()
            if hasattr(args, 'ema_audio') and args.ema_audio is not None:
                args.ema_audio.update()

            if batch_idx % args.log_interval == 0:
                log_str = 'Train Audio Model Epoch: {} \t Loss: {:.3f} (Mix added)'.format(epoch, loss.item())
                exp_logger(log_str)
                
        else: 
            # === Video Model 训练 (同理添加 Mix 逻辑) ===
            # Pass 1: Original
            global_logits, a_logits, v_logits, frame_logits, temporal_logits, frame_att, temporal_att, sims_after, mask_after = models[1](audio, video, video_st, with_ca=True, epoch_num=epoch-1)

            a_frame_logits = frame_logits[:,:,0,:,:]
            v_frame_logits = frame_logits[:,:,1,:,:]

            fbcl_loss = args.lamda_1*criterion2(sims_after, mask_after) + args.lamda_2*(criterion3(temporal_att[..., 0], frame_att[..., 0].max(dim=-2)[0]) + criterion3(temporal_att[..., 1], frame_att[..., 1].max(dim=-2)[0]))

            frame_target = audio_pseudo_labels * visual_pseudo_labels * target.unsqueeze(dim=1).expand(Pa.shape[0], 10, 25)
            
            global_sl_loss = criterion_edl(global_logits, target, epoch_num=epoch)
            temporal_sl_loss = criterion_edl(temporal_logits, frame_target, epoch_num=epoch) 
            tel_loss = global_sl_loss + temporal_sl_loss

            a_frame_sl_loss = ps_criterion(a_frame_logits, audio_pseudo_labels, fore_only=True, epoch_num=epoch)
            v_frame_sl_loss = ps_criterion(v_frame_logits, visual_pseudo_labels, fore_only=True, epoch_num=epoch)
            ps_loss = args.audio_factor * a_frame_sl_loss + v_frame_sl_loss

            audio_sl_loss = mel_criterion(a_logits, Pa, epoch_num=epoch)
            video_sl_loss = mel_criterion(v_logits, Pv, epoch_num=epoch)
            mel_loss = audio_sl_loss + args.video_factor * video_sl_loss

            loss_original = ps_loss + args.lamda_3*mel_loss + args.lamda_4*tel_loss + fbcl_loss


            # See the audio phase above for the shared mixing contract.
            mixing = getattr(args, 'mixing', 'uhnm')
            if mixing == 'similarity':
                idx = hard_negative_mining(audio, torch.ones(bsz, device=audio.device))
            elif mixing == 'uhnm':
                prior_strength = 2.0 if getattr(args, 'loss_semantics', 'legacy') == 'paper' else 1.0
                u_a = compute_uncertainty(
                    a_logits.detach(), args.evidence, prior_strength=prior_strength
                )
                idx = hard_negative_mining(audio, u_a.mean(dim=-1))
            elif mixing != 'random':
                raise ValueError(f"Unsupported mixing strategy: {mixing!r}")
            audio_mixed = audio[idx]
            Pa_mixed = Pa[idx]
            audio_pseudo_labels_mixed = audio_pseudo_labels[idx]

            # Pass 2: ������ Mixing Augmentation
            _, a_logits_mix, v_logits_mix, frame_logits_mix, _, _, _, _, _ = models[1](audio_mixed, video, video_st, with_ca=True, epoch_num=epoch-1)
            
            # 强制解耦
            loss_audio_mix = criterion_edl(a_logits_mix, Pa_mixed, epoch_num=epoch)
            loss_video_mix = criterion_edl(v_logits_mix, Pv, epoch_num=epoch)
            
            loss = loss_original + 0.5 * (loss_audio_mix + args.video_factor * loss_video_mix)

            loss.backward()
            optimizers[1].step()
            if hasattr(args, 'ema_video') and args.ema_video is not None:
                args.ema_video.update()

            if batch_idx % args.log_interval == 0:
                log_str = 'Train Video Model Epoch: {} \t Loss: {:.3f} (Mix added)'.format(epoch, loss.item())
                exp_logger(log_str)

def eval(args, model, val_loader, set, epoch=0, dual_eval=False):
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    if isinstance(model, list):
        eval_device = next(model[0].parameters()).device
    else:
        eval_device = next(model.parameters()).device
    exp_logger("begin evaluate.")
    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv(project_path("data", "AVVP_eval_audio.csv"), header=0, sep='\t')
    df_v = pd.read_csv(project_path("data", "AVVP_eval_visual.csv"), header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():

        criterion_edl_modality = EvidenceLoss(
            num_classes=2,
            evidence='exp',
            legacy_normalization=getattr(args, 'loss_semantics', 'legacy') == 'legacy',
        )
        for batch_idx, sample in tqdm(enumerate(val_loader), total=len(val_loader), desc="Eval epoch {}".format(epoch)):
            audio, video, video_st, target = sample['audio'].to(eval_device), \
                sample['video_s'].to(eval_device), \
                sample['video_st'].to(eval_device), \
                sample['label'].to(eval_device)

            if isinstance(model, list):
                _, a_logits, _, a_frame_logits, _, _, _, _, _ = model[0](audio, video, video_st)
                _, _, v_logits, v_frame_logits, _, _, _, _, _ = model[1](audio, video, video_st)
                a_sl_prob, _ = criterion_edl_modality.get_predictions(a_logits)
                v_sl_prob, _ = criterion_edl_modality.get_predictions(v_logits)
                a_frame_sl_prob, _ = criterion_edl_modality.get_predictions(a_frame_logits)
                v_frame_sl_prob, _ = criterion_edl_modality.get_predictions(v_frame_logits)
                frame_sl_prob = torch.stack((a_frame_sl_prob[:, :, 0, :, :], v_frame_sl_prob[:, :, 1, :, :]), dim=2)

            else:
                _, a_logits, v_logits, frame_logits, _, _, _, _, _ = model(audio, video, video_st)
                a_sl_prob, _ = criterion_edl_modality.get_predictions(a_logits)
                v_sl_prob, _ = criterion_edl_modality.get_predictions(v_logits)
                frame_sl_prob, _ = criterion_edl_modality.get_predictions(frame_logits)

            a_prob = torch.clamp(a_sl_prob[:, :, 0], min=1e-7, max=1 - 1e-7)
            v_prob = torch.clamp(v_sl_prob[:, :, 0], min=1e-7, max=1 - 1e-7)
            a_temporal_prob = frame_sl_prob[:, :, 0, :, 0]
            v_temporal_prob = frame_sl_prob[:, :, 1, :, 0]

            Pa = a_temporal_prob.cpu().detach().numpy()[0, :, :]
            Pv = v_temporal_prob.cpu().detach().numpy()[0, :, :]
            a_prob = a_prob.cpu().detach().numpy()
            v_prob = v_prob.cpu().detach().numpy()
            a_sl_prob = a_sl_prob.cpu().detach().numpy()
            v_sl_prob = v_sl_prob.cpu().detach().numpy()

            v_thresh_list = np.full((1, 25), 0.45)
            a_thresh_list = np.full((1, 25), 0.36)

            if args.selected_thre:
                excel_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint) + '_thresh_a.xlsx'
                wb = load_workbook(excel_path)
                ws = wb['Thres']
                for idx in range(25):
                    a_thresh_list[0, idx] = ws['A'][idx].value

                excel_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint) + '_thresh_v.xlsx'
                wb = load_workbook(excel_path)
                ws = wb['Thres']
                for idx in range(25):
                    v_thresh_list[0, idx] = ws['A'][idx].value

            oa = (a_prob >= a_thresh_list).astype(np.int_)
            ov = (v_prob >= v_thresh_list).astype(np.int_)

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.25).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.25).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level \
        = print_overall_metric(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av)
    return audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level


# ============================================================
# IPR: 迭代伪标签精炼 (v2 — Neighbor-Consensus 增强)
# ============================================================
def _neighbor_consensus(pred, conf, feat, k=5, gamma=0.3, temperature=0.1):
    """Batch-level neighbor-consensus prediction blending.

    For each sample, aggregates predictions from its top-k feature-space
    neighbors, weighted by similarity × neighbor confidence.  Blends the
    neighbor-consensus prediction back with the self-prediction.

    Args:
        pred: (B, T, C) self model predictions
        conf: (B, T, C) self confidence = 1 - u
        feat: (B, D) pooled features for neighbor search
        k: number of neighbors to aggregate
        gamma: neighbor influence (0 = self only, 1 = neighbors only)
        temperature: softmax temperature for similarity weighting

    Returns:
        consensus_pred: (B, T, C) blended predictions
    """
    B = feat.size(0)
    if B < 2:
        return pred  # singleton batch: no neighbors

    k_eff = min(k, B - 1)
    feat_norm = F.normalize(feat, p=2, dim=-1)
    sim = feat_norm @ feat_norm.T  # (B, B)
    sim.fill_diagonal_(-float('inf'))

    topk_sim, topk_idx = sim.topk(k_eff, dim=-1)  # (B, k)
    topk_w = F.softmax(topk_sim / temperature, dim=-1)  # (B, k)

    # Gather neighbor predictions and confidences
    nbr_pred = pred[topk_idx]   # (B, k, T, C)
    nbr_conf = conf[topk_idx]   # (B, k, T, C)

    # Weighted average: similarity × confidence
    w = topk_w.unsqueeze(-1).unsqueeze(-1) * nbr_conf  # (B, k, T, C)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    nbr_consensus = (nbr_pred * w).sum(dim=1)  # (B, T, C)

    # Blend self and neighbor consensus
    consensus = (1.0 - gamma) * pred + gamma * nbr_consensus
    return torch.clamp(consensus, 0.0, 1.0)


@torch.no_grad()
def refine_pseudo_labels(args, models, train_loader, round_id=2):
    """用训练好的 NREP 模型生成精炼后的伪标签。

    v2: 支持 Neighbor-Consensus 增强 (--ipr_nc_gamma > 0 时启用)。
    """
    args.device = getattr(args, "device", str(next(models[0].parameters()).device))
    for m in models:
        m.eval()

    T = getattr(args, 'ipr_temperature', 1.0)
    sparsity_th = getattr(args, 'ipr_sparsity', 0.0)
    blend_w = getattr(args, 'ipr_blend', 0.3)
    nc_k = getattr(args, 'ipr_nc_k', 5)
    nc_gamma = getattr(args, 'ipr_nc_gamma', 0.0)
    nc_temperature = getattr(args, 'ipr_nc_temperature', 0.1)

    refined_output_root = getattr(args, 'refined_output_root', None)
    if refined_output_root:
        round_root = os.path.join(refined_output_root, f'round_{round_id}')
        a_saved = os.path.join(round_root, 'audio')
        v_saved = os.path.join(round_root, 'visual')
    else:
        a_saved = args.a_pseudo_data_dir + f'_refined_r{round_id}'
        v_saved = args.v_pseudo_data_dir + f'_refined_r{round_id}'
    a_source = args.a_pseudo_data_dir
    v_source = args.v_pseudo_data_dir
    os.makedirs(a_saved, exist_ok=True)
    os.makedirs(v_saved, exist_ok=True)

    nc_tag = f', nc_k={nc_k}, nc_γ={nc_gamma}' if nc_gamma > 0 else ''
    exp_logger(f"Refining R{round_id}: T={T}, sparsity={sparsity_th}, blend={blend_w}{nc_tag}")

    for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Refine R{round_id}"):
        audio = sample['audio'].to(args.device, non_blocking=True)
        video = sample['video_s'].to(args.device, non_blocking=True)
        video_st = sample['video_st'].to(args.device, non_blocking=True)
        video_names = sample.get('video_name')
        if isinstance(video_names, str):
            video_names = [video_names]

        # 双模型推理
        _, a_logits, _, a_frame_logits, _, _, _, _, _ = models[0](audio, video, video_st)
        _, _, v_logits, v_frame_logits, _, _, _, _, _ = models[1](audio, video, video_st)

        def calibrated_pu(logits, T_val):
            evidence = torch.exp(torch.clamp(logits / T_val, -10, 10))
            if getattr(args, 'loss_semantics', 'legacy') == 'paper':
                alpha = evidence + 1.0
                S = alpha.sum(dim=-1, keepdim=True)
                p = alpha / S
            else:
                S = evidence.sum(dim=-1, keepdim=True) + 1.0
                p = evidence / S
            if getattr(args, 'ipr_risk', 'vacuity') == 'proxy':
                # Proxy ablations use the empirical within-batch rank so that
                # entropy, margin and EDL are compared on the same ordinal
                # risk scale.  The predictive probability ``p`` is unchanged.
                proxy_risk = risk_proxy_from_logits(
                    logits / T_val,
                    getattr(args, 'risk_proxy', 'edl'),
                    evidence_type=args.evidence,
                    prior_strength=1.0,
                )
                from experiments.structured_risk.risk import rank_normalize
                u = rank_normalize(proxy_risk)
                return p, u.unsqueeze(-1)
            if getattr(args, 'ipr_risk', 'vacuity') == 'beta_total_variance':
                from experiments.structured_risk.risk import beta_total_variance_gate
                beta_risk = beta_total_variance_gate(evidence + 0.0)
                u = beta_risk.unsqueeze(-1)
            else:
                u = 2.0 / S
                if getattr(args, 'ipr_risk', 'vacuity') == 'rank_vacuity':
                    from experiments.structured_risk.risk import rank_normalize
                    u = rank_normalize(u)
            return p, u

        a_prob, a_u = calibrated_pu(a_logits, T)
        v_prob, v_u = calibrated_pu(v_logits, T)
        a_f_prob, a_f_u = calibrated_pu(a_frame_logits, T)
        v_f_prob, v_f_u = calibrated_pu(v_frame_logits, T)

        a_frame_p = a_f_prob[:, :, 0, :, 0]  # (B, T, C)
        v_frame_p = v_f_prob[:, :, 1, :, 0]  # (B, T, C)
        a_frame_u = a_f_u[:, :, 0, :, 0]      # (B, T, C)
        v_frame_u = v_f_u[:, :, 1, :, 0]      # (B, T, C)

        a_conf = 1.0 - a_frame_u
        v_conf = 1.0 - v_frame_u

        # ---- Neighbor-Consensus 增强 ----
        if nc_gamma > 0:
            # Pool features over time for neighbor search
            a_feat = audio.mean(dim=1)          # (B, Da)
            v_feat = video.mean(dim=1)          # (B, Dv)

            a_frame_p = _neighbor_consensus(
                a_frame_p, a_conf, a_feat,
                k=nc_k, gamma=nc_gamma, temperature=nc_temperature,
            )
            v_frame_p = _neighbor_consensus(
                v_frame_p, v_conf, v_feat,
                k=nc_k, gamma=nc_gamma, temperature=nc_temperature,
            )

        a_pred = np.clip(a_frame_p.cpu().numpy(), 0, 1)
        v_pred = np.clip(v_frame_p.cpu().numpy(), 0, 1)
        a_conf = np.clip(a_conf.cpu().numpy(), 0, 1)
        v_conf = np.clip(v_conf.cpu().numpy(), 0, 1)

        if sparsity_th > 0:
            a_pred[a_pred < sparsity_th] = 0.0
            v_pred[v_pred < sparsity_th] = 0.0

        refine_mode = getattr(args, 'ipr_mode', 'risk_aware')
        if refine_mode == 'risk_aware':
            gate_side = getattr(args, 'ipr_gate', 'low')
            if gate_side == 'high':
                alpha_a = blend_w * (1.0 - a_conf)
                alpha_v = blend_w * (1.0 - v_conf)
            elif gate_side == 'low':
                alpha_a = blend_w * a_conf
                alpha_v = blend_w * v_conf
            else:
                raise ValueError(f"Unsupported IPR risk gate: {gate_side!r}")
        elif refine_mode == 'uniform':
            alpha_a = np.full_like(a_conf, blend_w)
            alpha_v = np.full_like(v_conf, blend_w)
        elif refine_mode == 'hard':
            alpha_a = None
            alpha_v = None
            a_pred = (a_pred >= args.ipr_hard_threshold).astype(np.float32)
            v_pred = (v_pred >= args.ipr_hard_threshold).astype(np.float32)
        else:
            raise ValueError(f"Unsupported IPR mode: {refine_mode!r}")
        for item_idx, video_name in enumerate(video_names):
            vname = os.path.basename(str(video_name))
            try:
                orig_a = np.load(os.path.join(a_source, vname + '.npy'))
                orig_v = np.load(os.path.join(v_source, vname + '.npy'))
            except (FileNotFoundError, OSError, ValueError):
                orig_a = a_pred[item_idx].copy()
                orig_v = v_pred[item_idx].copy()

            if refine_mode == 'hard':
                a_refined = a_pred[item_idx]
                v_refined = v_pred[item_idx]
            else:
                a_refined = ((1 - alpha_a[item_idx]) * orig_a
                             + alpha_a[item_idx] * a_pred[item_idx])
                v_refined = ((1 - alpha_v[item_idx]) * orig_v
                             + alpha_v[item_idx] * v_pred[item_idx])
            np.save(os.path.join(a_saved, vname + '.npy'), a_refined)
            np.save(os.path.join(v_saved, vname + '.npy'), v_refined)

    exp_logger(f"Refined pseudo-labels saved to {a_saved} and {v_saved}")
    return a_saved, v_saved


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument("--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument("--video_dir", type=str, default='data/feats/res152/', help="video dir")
    parser.add_argument("--st_dir", type=str, default='data/feats/r2plus1d_18/', help="video dir")
    parser.add_argument("--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument("--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument("--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--warm_up_epoch', type=float, default=0.9, help='warm-up epochs')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_step_size', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--start_class', type=int, default=0)
    parser.add_argument("--mode", type=str, default='train_with_label_smoothing',
                        choices=['train_with_label_smoothing', 'train_with_label_refinement',
                                 'train_with_pseudo_labels', 'test_model', 'select_thresholds'],
                        help="with mode to use")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--noise_ratio_file', type=str, default='noise_ratios.npz')
    parser.add_argument('--a_smooth', type=float, default=1.0)
    parser.add_argument('--v_smooth', type=float, default=0.9)
    parser.add_argument('--clamp', type=float, default=1e-7)
    parser.add_argument('--nce_smooth', type=float, default=0.1)
    parser.add_argument('--max_thresh', type=float, default=0.8)
    parser.add_argument('--min_thresh', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.2, help='feature temperature number')
    parser.add_argument('--warm_num', type=int, default=1000, help='warm epoch number')
    parser.add_argument('--log_interval', type=int, default=700, help='how many batches for logging training status')
    parser.add_argument('--log_file', type=str, help="log file path")
    parser.add_argument('--save_model', type=str, default="true", choices=["true", "false"],
                        help='whether to save model')
    parser.add_argument("--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument("--checkpoint", type=str, default='NREP_PL', help="save model name")
    parser.add_argument('--resume_path', type=str, default=None,
                        help='explicit .best path for test_model')
    parser.add_argument(
        '--round1_resume_path', type=str, default=None,
        help='locked Round-1 .best checkpoint; when set, skip Round-1 training and run refinement/Round 2 from identical weights',
    )

    parser.add_argument("--logger", type=str, default='logger/', help="save model name")
    parser.add_argument("--best_audio_metric", type=str, default='audio_seg',
                        choices=["audio_seg", "visual_seg", "av_seg", "avg_type_seg", "avg_event_seg", "audio_eve",
                                 "visual_eve", "av_eve", "avg_type_eve", "avg_event_eve"],
                        help="best audio metric type")
    parser.add_argument("--best_video_metric", type=str, default='avg_event_eve',
                        choices=["audio_seg", "visual_seg", "av_seg", "avg_type_seg", "avg_event_seg", "audio_eve",
                                 "visual_eve", "av_eve", "avg_type_eve", "avg_event_eve"], help="best metric type")
    parser.add_argument("--no-log", action='store_true', default=False, help="logger switcher")
    parser.add_argument("--selected_thre", action='store_true', default=False, help="logger switcher")
    parser.add_argument("--vggish", action='store_true', default=False, help="feature switcher")

    parser.add_argument("--v_pseudo_data_dir", type=str, default=str(project_path('data', 'CLIP', 'segment_pseudo_labels')),
                        help="visual segment-level pseudo labels dir")
    parser.add_argument("--a_pseudo_data_dir", type=str, default=str(project_path('data', 'CLAP', 'segment_pseudo_labels')),
                        help="audio segment-level pseudo labels dir")

    parser.add_argument('--lamda_1', type=float, default=1.0)
    parser.add_argument('--lamda_2', type=float, default=1.0)
    parser.add_argument('--lamda_3', type=float, default=1.0)
    parser.add_argument('--lamda_4', type=float, default=1.0)
    parser.add_argument('--audio_factor', type=float, default=2.0)
    parser.add_argument('--video_factor', type=float, default=3.0)
    parser.add_argument("--evidence", type=str, default='exp', choices=["relu", "exp", "softplus"])
    parser.add_argument("--loss_type", type=str, default='bce', choices=["bce", "log"])
    parser.add_argument("--uncertain", action='store_true', default=False, help="feature switcher")
    parser.add_argument(
        '--loss_semantics',
        choices=['legacy', 'paper'],
        default='legacy',
        help='legacy reproduces reported runs; paper enables alpha=e+1 evidence loss',
    )
    parser.add_argument('--brier_weight', type=float, default=0.0, help='Brier score regularization weight')
    parser.add_argument('--edl_temperature', type=float, default=1.0, help='EDL evidence temperature scaling')
    parser.add_argument(
        '--risk_loss', choices=['none', 'global', 'class_mass', 'smooth'], default='none',
        help='Ordinal-risk weighting for pseudo-supervision; default preserves the locked loss.',
    )
    parser.add_argument(
        '--risk_loss_scope', choices=['ps', 'ps_mel'], default='ps',
        help='Apply ordinal-risk weighting to frame pseudo labels only or to frame and modality pseudo labels.',
    )
    parser.add_argument('--risk_weight_floor', type=float, default=0.5)
    parser.add_argument('--risk_weight_gamma', type=float, default=1.0)
    parser.add_argument('--risk_weight_warmup', type=int, default=3)
    parser.add_argument('--risk_smooth_max', type=float, default=0.2)
    parser.add_argument('--use_ema', action='store_true', default=False, help='Use EMA weight averaging')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--ipr_rounds', type=int, default=2, help='IPR 迭代轮数')
    parser.add_argument('--ipr_lr_decay', type=float, default=0.5, help='每轮学习率衰减因子')
    parser.add_argument(
        '--mixing', choices=['random', 'similarity', 'uhnm'], default='uhnm',
        help='Cross-modal mixing scheduler: random is quality-agnostic; similarity uses feature proximity only; uhnm multiplies similarity by evidence uncertainty.',
    )
    parser.add_argument(
        '--ipr_mode', choices=['risk_aware', 'uniform', 'hard'], default='risk_aware',
        help='Pseudo-label refinement policy used when --ipr_rounds >= 2.',
    )
    parser.add_argument(
        '--ipr_gate', choices=['low', 'high'], default='low',
        help='Risk side permitted to update the soft target in risk_aware IPR.',
    )
    parser.add_argument('--ipr_hard_threshold', type=float, default=0.5,
                        help='Foreground threshold for --ipr_mode hard.')
    parser.add_argument('--ipr_temperature', type=float, default=1.0, help='IPR精炼温度校准 (T>1→更平滑的不确定性)')
    parser.add_argument(
        '--ipr_risk', choices=['vacuity', 'rank_vacuity', 'beta_total_variance', 'proxy'], default='vacuity',
        help='Risk signal for IPR gating; proxy selects --risk_proxy and rank-normalizes it within each batch.',
    )
    parser.add_argument(
        '--risk_proxy', choices=['edl', 'entropy', 'margin', 'random'], default='edl',
        help='Risk proxy used by the proxy_uhnm scheduler and --ipr_risk proxy.',
    )
    parser.add_argument('--ipr_sparsity', type=float, default=0.0, help='IPR精炼硬阈值 (0=禁用, 建议0.2-0.3)')
    parser.add_argument('--ipr_blend', type=float, default=0.3, help='IPR精炼融合权重 (max model influence)')
    parser.add_argument('--ipr_nc_k', type=int, default=5, help='Neighbor-Consensus: top-k 邻居数')
    parser.add_argument('--ipr_nc_gamma', type=float, default=0.0,
                        help='Neighbor-Consensus: 邻居影响权重 (0=仅自身预测, 0.3=30%%邻居共识)')
    parser.add_argument('--ipr_nc_temperature', type=float, default=0.1,
                        help='Neighbor-Consensus: 相似度 softmax 温度')
    parser.add_argument('--refine_batch_size', type=int, default=128, help='IPR批量推理大小')
    parser.add_argument('--refined_output_root', type=str, default=None,
                        help='run-scoped root for refined pseudo-labels')

    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--eval_num_workers', type=int, default=1)
    args = parser.parse_args()

    if args.no_log:
        exp_logger.disable_file()
    else:
        os.makedirs(os.path.dirname(args.logger), exist_ok=True)
        logger_timestamps = datetime.datetime.now()
        exp_logger.set_file(os.path.join(args.logger, args.mode + '_' + datetime.datetime.strftime(logger_timestamps,
                                                                                                   '%Y-%m-%d %H:%M:%S') + ".log"))

    save_dir = osp.join(args.model_save_dir, args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)

    # print parameters
    exp_logger('----------------args-----------------')
    for k in list(vars(args).keys()):
        exp_logger('%s: %s' % (k, vars(args)[k]))
    exp_logger('----------------args-----------------')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    exp_logger(f'current time: {cur}')

    set_random_seed(args.seed)
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    save_model = args.save_model == 'true'
    os.makedirs(args.model_save_dir, exist_ok=True)

    start = time.time()

    if args.mode == 'train_with_pseudo_labels':
        logger = SummaryWriter(args.log_file) if args.log_file else None

        args.with_ca = False
        train_loader, val_loader, test_loader = get_LLP_dataloader(args)

        if args.vggish:
            audio_model = NREP(args.num_layers, args.temperature, dim_a=128, dim_v=2048).to(args.device)
            video_model = copy.deepcopy(audio_model)
        else:
            audio_model = NREP(args.num_layers, args.temperature).to(args.device)
            video_model = copy.deepcopy(audio_model)
        models = [audio_model, video_model]

        # EMA initialization
        if args.use_ema:
            args.ema_audio = ModelEMA(audio_model, decay=args.ema_decay)
            args.ema_video = ModelEMA(video_model, decay=args.ema_decay)
            exp_logger(f"EMA enabled with decay={args.ema_decay}")
        else:
            args.ema_audio = None
            args.ema_video = None

        optimizers = [optim.Adam(audio_model.parameters(), lr=args.lr),
                      optim.Adam(video_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]
        schedulers = [optim.lr_scheduler.StepLR(optimizers[0], step_size=args.lr_step_size, gamma=args.lr_gamma),
                      optim.lr_scheduler.StepLR(optimizers[1], step_size=args.lr_step_size, gamma=args.lr_gamma)]

        criterion = nn.BCELoss()

        best_audio_F = 0
        best_audio_model = None
        best_video_F = 0
        best_video_model = None
        best_state_dict = get_random_state()

        round1_epochs = 0 if args.round1_resume_path else args.epochs
        if args.round1_resume_path:
            exp_logger(
                'Round 1 training skipped; loading locked checkpoint from {}'.format(
                    args.round1_resume_path
                )
            )

        for i, scheduler in enumerate(schedulers):
            for epoch in range(1, round1_epochs + 1):
                train_with_pseudo_labels(args, models, train_loader, optimizers, criterion,
                                         epoch=round1_epochs * i + epoch, logger=logger)
                scheduler.step(epoch)

                state_dict = build_training_state(
                    models, optimizers, schedulers,
                    epoch=round1_epochs * i + epoch, round_id=1,
                )

                os.makedirs(os.path.dirname(osp.join(args.model_save_dir, args.checkpoint)), exist_ok=True)
                save_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint + '.ckpt')
                exp_logger("Saving checkpoint at epoch {}: {} \n".format(round1_epochs * i + epoch, save_path))
                torch.save(state_dict, save_path)

                exp_logger("Validation Performance of Epoch {}:".format(round1_epochs * i + epoch))

                # Use EMA weights for validation
                if args.use_ema:
                    args.ema_audio.apply_shadow()
                    args.ema_video.apply_shadow()

                audio_seg, visual_seg, av_seg, avg_type_seg, avg_event_seg, audio_eve, visual_eve, av_eve, avg_type_eve, avg_event_eve = eval(
                    args, models, val_loader, args.label_val, round1_epochs * i + epoch, dual_eval=True)

                # Restore original weights for continued training
                if args.use_ema:
                    args.ema_audio.restore()
                    args.ema_video.restore()

                if locals()[args.best_audio_metric] >= best_audio_F:
                    best_audio_F = locals()[args.best_audio_metric]
                    best_audio_model = copy.deepcopy(models[0])
                    if save_model:
                        best_state_dict['model_audio'] = best_audio_model.state_dict()
                        save_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint + '.best')
                        torch.save(best_state_dict, save_path)
                        exp_logger("Best audio results at {} have been updated: {} \n".format(args.best_audio_metric,
                                                                                              save_path))

                if locals()[args.best_video_metric] >= best_video_F:
                    best_video_F = locals()[args.best_video_metric]
                    best_video_model = copy.deepcopy(models[1])
                    if save_model:
                        best_state_dict['model_video'] = best_video_model.state_dict()
                        save_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint + '.best')
                        torch.save(best_state_dict, save_path)
                        exp_logger("Best video results at {} have been updated: {} \n".format(args.best_video_metric,
                                                                                              save_path))

        if args.round1_resume_path:
            locked = torch.load(args.round1_resume_path, map_location=args.device)
            audio_model.load_state_dict(locked['model_audio'])
            video_model.load_state_dict(locked['model_video'])
            best_audio_model = copy.deepcopy(audio_model)
            best_video_model = copy.deepcopy(video_model)
            best_state_dict['model_audio'] = best_audio_model.state_dict()
            best_state_dict['model_video'] = best_video_model.state_dict()
            locked_copy = osp.join(
                args.model_save_dir, args.checkpoint, args.checkpoint + '.best'
            )
            torch.save(best_state_dict, locked_copy)
            exp_logger('Locked Round 1 checkpoint copied to {} \n'.format(locked_copy))

        if logger:
            logger.close()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        models = [best_audio_model, best_video_model]
        exp_logger("=== Round 1 Test ===")
        eval(args, models, test_loader, args.label_test)

        # ============================================================
        # IPR 多轮迭代精炼 (默认 2 轮，可配 --ipr_rounds 3)
        # ============================================================
        num_ipr_rounds = getattr(args, 'ipr_rounds', 2)
        lr_decay = getattr(args, 'ipr_lr_decay', 0.5)
        best_audio_model_prev = best_audio_model
        best_video_model_prev = best_video_model

        for round_id in range(2, num_ipr_rounds + 1):
            exp_logger(f"\n=== IPR Round {round_id}: Refining pseudo-labels ===")
            refine_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                          video_dir=args.video_dir, st_dir=args.st_dir,
                                          transform=transforms.Compose([ToTensor()]),
                                          v_pseudo_data_dir=args.v_pseudo_data_dir,
                                          a_pseudo_data_dir=args.a_pseudo_data_dir)
            refine_loader = DataLoader(
                refine_dataset,
                batch_size=args.refine_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            models_refine = [best_audio_model_prev, best_video_model_prev]
            refined_a_dir, refined_v_dir = refine_pseudo_labels(args, models_refine, refine_loader, round_id=round_id)

            args.a_pseudo_data_dir = refined_a_dir
            args.v_pseudo_data_dir = refined_v_dir

            train_dataset_r = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                           video_dir=args.video_dir, st_dir=args.st_dir,
                                           transform=transforms.Compose([ToTensor()]),
                                           v_pseudo_data_dir=args.v_pseudo_data_dir,
                                           a_pseudo_data_dir=args.a_pseudo_data_dir)
            train_loader_r = DataLoader(train_dataset_r, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers, pin_memory=True)

            cur_lr = args.lr * (lr_decay ** (round_id - 1))
            audio_model_r = copy.deepcopy(best_audio_model_prev)
            video_model_r = copy.deepcopy(best_video_model_prev)
            models_r = [audio_model_r, video_model_r]

            if args.use_ema:
                args.ema_audio = ModelEMA(audio_model_r, decay=args.ema_decay)
                args.ema_video = ModelEMA(video_model_r, decay=args.ema_decay)

            optimizers_r = [optim.Adam(audio_model_r.parameters(), lr=cur_lr),
                             optim.Adam(video_model_r.parameters(), lr=cur_lr, weight_decay=args.weight_decay)]
            schedulers_r = [optim.lr_scheduler.StepLR(optimizers_r[0], step_size=args.lr_step_size, gamma=args.lr_gamma),
                              optim.lr_scheduler.StepLR(optimizers_r[1], step_size=args.lr_step_size, gamma=args.lr_gamma)]

            best_audio_Fr, best_audio_model_r = 0, None
            best_video_Fr, best_video_model_r = 0, None
            best_state_dict_r = get_random_state()

            for i, scheduler in enumerate(schedulers_r):
                for epoch in range(1, args.epochs + 1):
                    train_with_pseudo_labels(args, models_r, train_loader_r, optimizers_r, criterion,
                                             epoch=args.epochs * i + epoch, logger=logger)
                    scheduler.step(epoch)

                    state_dict = build_training_state(
                        models_r, optimizers_r, schedulers_r,
                        epoch=args.epochs * i + epoch, round_id=round_id,
                    )

                    save_path = osp.join(args.model_save_dir, args.checkpoint,
                                          args.checkpoint + f'_r{round_id}.ckpt')
                    torch.save(state_dict, save_path)

                    exp_logger(f"Round {round_id} Val Epoch {args.epochs * i + epoch}:")
                    if args.use_ema:
                        args.ema_audio.apply_shadow()
                        args.ema_video.apply_shadow()
                    metrics = eval(args, models_r, val_loader, args.label_val, args.epochs * i + epoch, dual_eval=True)
                    if args.use_ema:
                        args.ema_audio.restore()
                        args.ema_video.restore()
                    audio_seg, visual_seg, av_seg, avg_type_seg, avg_event_seg, \
                        audio_eve, visual_eve, av_eve, avg_type_eve, avg_event_eve = metrics

                    if locals()[args.best_audio_metric] >= best_audio_Fr:
                        best_audio_Fr = locals()[args.best_audio_metric]
                        best_audio_model_r = copy.deepcopy(models_r[0])
                        if save_model:
                            best_state_dict_r['model_audio'] = best_audio_model_r.state_dict()
                            torch.save(best_state_dict_r, osp.join(args.model_save_dir, args.checkpoint,
                                                                    args.checkpoint + f'_r{round_id}.best'))

                    if locals()[args.best_video_metric] >= best_video_Fr:
                        best_video_Fr = locals()[args.best_video_metric]
                        best_video_model_r = copy.deepcopy(models_r[1])
                        if save_model:
                            best_state_dict_r['model_video'] = best_video_model_r.state_dict()
                            torch.save(best_state_dict_r, osp.join(args.model_save_dir, args.checkpoint,
                                                                    args.checkpoint + f'_r{round_id}.best'))

            best_audio_model_prev = best_audio_model_r
            best_video_model_prev = best_video_model_r
            models_r = [best_audio_model_prev, best_video_model_prev]
            exp_logger(f"=== Round {round_id} Test (IPR Refined) ===")
            eval(args, models_r, test_loader, args.label_test)

    elif args.mode == 'test_model':
        dataset = args.label_test
        args.with_ca = True

        audio_model = NREP(args.num_layers, args.temperature).to(args.device)
        video_model = copy.deepcopy(audio_model)

        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir,
                                   video_dir=args.video_dir, st_dir=args.st_dir,
                                   transform=transforms.Compose([ToTensor()]), v_pseudo_data_dir=args.v_pseudo_data_dir,
                                   a_pseudo_data_dir=args.a_pseudo_data_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.eval_num_workers, pin_memory=True)

        resume_path = args.resume_path or osp.join(
            args.model_save_dir, args.checkpoint, args.checkpoint + '.best'
        )
        exp_logger("Resuming model from {}".format(resume_path))
        resume = torch.load(resume_path)
        audio_model.load_state_dict(resume['model_audio'])
        video_model.load_state_dict(resume['model_video'])

        models = [audio_model, video_model]
        eval(args, models, test_loader, dataset)

    end = time.time()
    exp_logger(f'duration time {(end - start) / 60} mins.')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    exp_logger(f'current time: {cur}')


if __name__ == '__main__':
    main()
