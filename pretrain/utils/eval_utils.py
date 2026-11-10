import torch
import numpy as np
from einops import repeat


class AVVPEvaluator:
    def __init__(self):
        self.f_scores = {'segment': {'f_a': [],
                                     'f_v': [],
                                     'f':   [],
                                     'f_av':[]},
                         'event': {'f_a': [],
                                   'f_v': [],
                                   'f':   [],
                                   'f_av':[]}
                        }

    def update(self, level, f_a, f_v, f, f_av):
        self.f_scores[level]['f_a'].append(f_a)
        self.f_scores[level]['f_v'].append(f_v)
        self.f_scores[level]['f'].append(f)
        self.f_scores[level]['f_av'].append(f_av)

    def output_result(self):
        F_seg_a = self.f_scores['segment']['f_a']
        F_seg_v = self.f_scores['segment']['f_v']
        F_seg = self.f_scores['segment']['f']
        F_seg_av = self.f_scores['segment']['f_av']
        
        F_event_a = self.f_scores['event']['f_a']
        F_event_v = self.f_scores['event']['f_v']
        F_event = self.f_scores['event']['f']
        F_event_av = self.f_scores['event']['f_av']

        seg_type = (100 * np.mean(np.array(F_seg_av)) + 100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(np.array(F_seg_v))) / 3.
        seg_event = 100 * np.mean(np.array(F_seg))
        event_type = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(np.array(F_event_v))) / 3.
        event_event = 100 * np.mean(np.array(F_event))

        F_scores = {'Seg-a': 100 * np.mean(np.array(F_seg_a)),
                    'Seg-v': 100 * np.mean(np.array(F_seg_v)),
                    'Seg-av': 100 * np.mean(np.array(F_seg_av)),
                    'Seg-type': seg_type,
                    'Seg-event': seg_event,
                    'Event-a': 100 * np.mean(np.array(F_event_a)),
                    'Event-v': 100 * np.mean(np.array(F_event_v)),
                    'Event-av': 100 * np.mean(np.array(F_event_av)),
                    'Event-type': event_type,
                    'Event-event': event_event}
        
        return F_scores
    

class MicroFscoreEvaluator:
    def __init__(self, num_classes):
        self.micro_stats = {
            'a': {'TP': np.zeros(num_classes, dtype=np.float32),
                  'FN': np.zeros(num_classes, dtype=np.float32),
                  'FP': np.zeros(num_classes, dtype=np.float32)},
            'v': {'TP': np.zeros(num_classes, dtype=np.float32),
                  'FN': np.zeros(num_classes, dtype=np.float32),
                  'FP': np.zeros(num_classes, dtype=np.float32)},
            'av': {'TP': np.zeros(num_classes, dtype=np.float32),
                   'FN': np.zeros(num_classes, dtype=np.float32),
                   'FP': np.zeros(num_classes, dtype=np.float32)}
        }

    def update(self, modality, pred, gt):
        assert modality in self.micro_stats, f'invalid modality: {modality} for Micro_F_score_Evaluator.micro_stats'
        TP = np.sum(pred * gt, axis=1)
        FN = np.sum((1 - pred) * gt, axis=1)
        FP = np.sum(pred * (1 - gt), axis=1)
        self.micro_stats[modality]['TP'] += TP
        self.micro_stats[modality]['FN'] += FN
        self.micro_stats[modality]['FP'] += FP

    def output_result(self):
        F_seg_a = (self.micro_stats['a']['TP'].sum()*2*100) / (self.micro_stats['a']['TP'].sum()*2 + self.micro_stats['a']['FN'].sum() + self.micro_stats['a']['FP'].sum() + 1e-8)
        F_seg_v = (self.micro_stats['v']['TP'].sum()*2*100) / (self.micro_stats['v']['TP'].sum()*2 + self.micro_stats['v']['FN'].sum() + self.micro_stats['v']['FP'].sum() + 1e-8)
        F_seg_av = (self.micro_stats['av']['TP'].sum()*2*100) / (self.micro_stats['av']['TP'].sum()*2 + self.micro_stats['av']['FN'].sum() + self.micro_stats['av']['FP'].sum())

        F_scores = {'Seg-a': F_seg_a,
                    'Seg-v': F_seg_v,
                    'Seg-av': F_seg_av}
        
        return F_scores
    
    def get_classwise_result(self, modality):
        classwise_micro_F = self.micro_stats[modality]['TP']*2*100 / (self.micro_stats[modality]['TP']*2 +\
                                                                      self.micro_stats[modality]['FN'] +\
                                                                      self.micro_stats[modality]['FP'] + 1e-8)
        return classwise_micro_F, self.micro_stats[modality]['TP'], self.micro_stats[modality]['FN'], self.micro_stats[modality]['FP']


class AccEvaluator:
    def __init__(self):
        self.correct = 0.0
        self.total = 0.0

    def update(self, binary_pred, binary_gt):
        assert binary_pred.shape == binary_gt.shape, f'binary_pred and binary_gt must have the same shape, but got {binary_pred.shape} and {binary_gt.shape}'
        assert isinstance(binary_pred, np.ndarray), 'binary_pred must be numpy arrays'
        assert isinstance(binary_gt, np.ndarray), 'binary_gt must be numpy arrays'

        C, T = binary_pred.shape

        ''' Transform binary gt (C, T) to class-indexed array (T,) '''
        gt = np.full(T, C, dtype=int)       # initialize an array of length T with background index
        rows_with_ones = np.argmax(binary_gt, axis=0)   # find the row indices where binary_gt is 1 along axis 0
        gt[binary_gt.any(axis=0)] = rows_with_ones[binary_gt.any(axis=0)]    # set gt[i] to the row index if there's a 1 in that position

        if binary_pred.sum() == 0:
            pred = np.full(T, C, dtype=int)       # initialize an array of length T with background index
        else:
            assert np.all(binary_pred.sum(axis=0) <= 1), 'more than one event exist in one of the segments in binary_pred'

            pred = np.full(T, C, dtype=int)
            rows_with_ones = np.argmax(binary_pred, axis=0)
            pred[binary_pred.any(axis=0)] = rows_with_ones[binary_pred.any(axis=0)]

        self.correct += (pred == gt).sum()
        self.total += pred.size

    def output_result(self):
        return (self.correct / (self.total + 1e-8)) * 100


def show_result(logger, dataset, mode, avvp_F_scores_dict):

    logger.info(f'{dataset} {mode}')
    if dataset == 'UnAV':
        logger.info(f'AVVP F-scores:  Seg-av: {avvp_F_scores_dict["Seg-av"]:.3f}  Event-av: {avvp_F_scores_dict["Event-av"]:.3f}')
    elif dataset == 'LLP':
        result_list = [f'{key}: {val:.3f}' for key, val in avvp_F_scores_dict.items()]
        result_str = "  ".join(result_list)
        result_str = "AVVP F-scores:  " + result_str
        logger.info(result_str)
    print()


@torch.no_grad()
def calculate_classwise_thresholds(args, model, data_loader, device):

    assert args.dataset != 'UnAV', 'DON\'t calculate classwise thresholds for the UnAV dataset!'

    model.eval()

    categories = data_loader.dataset.categories

    total_a_logits, total_v_logits = [], []
    total_a_GT, total_v_GT, total_av_GT = [], [], []

    for batch_idx, batch_data in enumerate(data_loader):
        clip_feats, clap_feats = batch_data['clip_feat'].to(device), batch_data['clap_feat'].to(device)
        valid_masks = batch_data['valid_mask'].to(device)
        attn_masks = batch_data['attn_mask'].to(device)
        weak_labels = batch_data['weak_label'].float().to(device)
        duration = batch_data['duration'][0].item()
        batch_size, T = clip_feats.size()[:2]


        outputs = model(clap_feats, clip_feats, valid_mask=valid_masks, attn_mask=attn_masks)
        a_logits, v_logits = outputs; a_logits = a_logits.float(); v_logits = v_logits.float()
        a_logits = a_logits.squeeze(0).permute(1, 0)  # (C, T)
        v_logits = v_logits.squeeze(0).permute(1, 0)  # (C, T)
        C, T = a_logits.shape

        weak_labels = repeat(weak_labels.squeeze(0), 'c -> c t', t=T).to(torch.float32)[:C, ]
        a_logits = torch.where(weak_labels == 1, a_logits, float('-inf'))
        v_logits = torch.where(weak_labels == 1, v_logits, float('-inf'))

        # remove padded tokens
        a_logits = a_logits[:, :duration]
        v_logits = v_logits[:, :duration]
        total_a_logits.append(a_logits)
        total_v_logits.append(v_logits)

        GT_a = batch_data['gt_a_label'].to(device).float().squeeze(0).permute(1, 0)  # (C, T)
        GT_v = batch_data['gt_v_label'].to(device).float().squeeze(0).permute(1, 0)  # (C, T)
        total_a_GT.append(GT_a)
        total_v_GT.append(GT_v)
        total_av_GT.append(torch.zeros_like(GT_a))

    total_a_logits = torch.cat(total_a_logits, dim=1)
    total_v_logits = torch.cat(total_v_logits, dim=1)
    total_a_GT = torch.cat(total_a_GT, dim=1)
    total_v_GT = torch.cat(total_v_GT, dim=1)
    total_av_GT = torch.cat(total_av_GT, dim=1)


    '''
    Calculate the best audio and visual thresholds
    Note: Calculation for the LLP dataset is different from that for the AVE dataset.
        - For LLP, the thresholds are calculated deterministically.
        - For AVE, the thresholds are calculated by grid search.
    '''
    if args.threshold_type == 'classwise':
        best_a_thresholds = torch.zeros(len(categories), device=device)
        best_v_thresholds = torch.zeros(len(categories), device=device)
        for class_idx in range(len(categories)):
            a_thres, a_f_score = get_best_f_threshold(total_a_GT[class_idx], total_a_logits[class_idx])
            v_thres, v_f_score = get_best_f_threshold(total_v_GT[class_idx], total_v_logits[class_idx])
            best_a_thresholds[class_idx] = a_thres
            best_v_thresholds[class_idx] = v_thres

    elif args.threshold_type == 'class_agnostic':
        best_a_thresholds, a_f_score = get_best_f_threshold(total_a_GT.ravel(), total_a_logits.ravel())
        best_v_thresholds, v_f_score = get_best_f_threshold(total_v_GT.ravel(), total_v_logits.ravel())

    return best_a_thresholds, best_v_thresholds


def get_best_f_threshold(labels, preds):
    """Get the best threshold w.r.t. F1 score.

    Args:
        labels: 1d tensor of binary ground truth labels.
        preds: 1d tensor of predicted probabilities.

    Returns:
        best_threshold: the threshold giving the best F1 score.
        f1_score: F1 score achieved with the best threshold
    """
    # sort by predicted probabilities
    sort_idx = torch.argsort(preds)
    labels = labels[sort_idx]
    preds = preds[sort_idx]

    # compute intermediate statistics when threshold falls below each probability
    num_tps = torch.cumsum(labels.flip(0), dim=0).flip(0)
    total_true = num_tps[0].item()
    total_pred = len(preds) - torch.arange(len(preds), dtype=torch.float32, device=preds.device)

    # reduce to unique predicted probabilities (have bugs here, comment these 3 lines doesn't affect result)
    # preds, unique_idx = torch.unique(preds, return_inverse=False, return_counts=False, dim=0, sorted=True, return_index=True)
    # num_tps = num_tps[unique_idx]
    # total_pred = total_pred[unique_idx]

    # compute f1 scores for each threshold
    p = num_tps / total_pred
    r = num_tps / total_true
    f = 2 * p * r / torch.clamp(p + r, min=1e-12)

    # find best threshold
    thresholds = (preds + torch.cat((torch.tensor([-float('inf')], device=preds.device), preds[:-1]))) / 2
    best_idx = torch.argmax(f).item()

    return thresholds[best_idx], f[best_idx].item()
