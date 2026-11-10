import numpy as np


def to_vec(start, end, num_segments):
    x = np.zeros(num_segments)
    x[start:end] = 1
    return x


def extract_event(seq):
    T = seq.shape[0]

    x = []
    i = 0
    while i < T:
        if seq[i] == 1:
            start = i
            if i + 1 == T:
                i = i + 1
                end = i
                x.append(to_vec(start, end, T))
                break

            for j in range(i + 1, T):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end, T))
                    break
                else:
                    i = j + 1
                    if i == T:
                        end = i
                        x.append(to_vec(start, end, T))
                        break
        else:
            i += 1
    return x


def event_wise_metric(event_p, event_gt):
    TP = 0
    FN = 0
    FP = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  # IoU, threshold=0.5
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  # 0.5
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FN, FP


def calculate_event_level_confusion_matrix(pred, gt):
    # pred shape = (C, T), gt shape = (C, T)
    assert pred.shape == gt.shape, "inconsistent shape between prediction matrix and ground-truth matrix!"
    C, T = pred.shape
    
    event_pred = [None for c in range(C)]
    event_gt = [None for c in range(C)]
    TP = np.zeros(C)
    FN = np.zeros(C)
    FP = np.zeros(C)

    for c in range(C):
        seq_pred = pred[c, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred)
            event_pred[c] = x
        seq_gt = gt[c, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt)
            event_gt[c] = x

        tp, fn, fp = event_wise_metric(event_pred[c], event_gt[c])
        TP[c] += tp
        FN[c] += fn
        FP[c] += fp

    return TP, FN, FP


def event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):

    TP_a, FN_a, FP_a = calculate_event_level_confusion_matrix(SO_a, GT_a)
    TP_v, FN_v, FP_v = calculate_event_level_confusion_matrix(SO_v, GT_v)
    TP_av, FN_av, FP_av = calculate_event_level_confusion_matrix(SO_av, GT_av)
    TP, FN, FP = TP_a + TP_v, FN_a + FN_v, FP_a + FP_v

    f_a = calculate_F_score(TP_a, FN_a, FP_a)
    f_v = calculate_F_score(TP_v, FN_v, FP_v)
    f = calculate_F_score(TP, FN, FP)
    f_av = calculate_F_score(TP_av, FN_av, FP_av)

    return f_a, f_v, f, f_av


def calculate_segment_level_confusion_matrix(pred, gt):
    # pred shape = (C, T), gt shape = (C, T)
    assert pred.shape == gt.shape, "inconsistent shape between prediction matrix and ground-truth matrix!"

    TP = np.sum(pred * gt, axis=1)          # (C, )
    FN = np.sum((1 - pred) * gt, axis=1)    # (C, )
    FP = np.sum(pred * (1 - gt), axis=1)    # (C, )
    return TP, FN, FP


def calculate_F_score(TP, FN, FP):
    assert TP.shape == FN.shape, "TP array should have the same length as FN array!"
    assert TP.shape == FP.shape, "TP array should have the same length as FP array!"
    C = TP.shape[0]
    
    classwise_F_scores = []
    for ii in range(C):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            classwise_F_scores.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    if len(classwise_F_scores) == 0:
        F_score = 1.0  # all true negatives
    else:
        F_score = (sum(classwise_F_scores) / len(classwise_F_scores))     # average across classes
    return F_score
    

def segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores = 2 * TP / (2 * TP + FP + FN)
    # all inputs shapes are (C, T)
    # False negative: prediction shows negative, but it actually is positive
    # False positive: prediction shows positive, but it actually is negative

    TP_a, FN_a, FP_a = calculate_segment_level_confusion_matrix(SO_a, GT_a)
    TP_v, FN_v, FP_v = calculate_segment_level_confusion_matrix(SO_v, GT_v)
    TP_av, FN_av, FP_av = calculate_segment_level_confusion_matrix(SO_av, GT_av)
    TP, FN, FP = TP_a + TP_v, FN_a + FN_v, FP_a + FP_v

    f_a = calculate_F_score(TP_a, FN_a, FP_a)
    f_v = calculate_F_score(TP_v, FN_v, FP_v)
    f = calculate_F_score(TP, FN, FP)
    f_av = calculate_F_score(TP_av, FN_av, FP_av)

    return f_a, f_v, f, f_av
