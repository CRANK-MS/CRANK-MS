import numpy as np
from collections import Sequence
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import interp


_depth = lambda L: isinstance(L, (Sequence, np.ndarray)) and max(map(_depth, L)) + 1


def get_metrics(y_true_all, y_preds_all, scores_all):

    y_true_all = [y_true_all] if _depth(y_true_all) == 1 else y_true_all
    scores_all = [scores_all] if _depth(scores_all) == 1 else scores_all

    met = defaultdict(list)

    # metrics that are based on predictions
    for y_true, y_pred, scores in zip(y_true_all, y_preds_all, scores_all):

        cnf = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cnf.ravel()
        N = TN + TP + FN + FP
        S = (TP + FN) / N
        P = (TP + FP) / N
        acc = (TN + TP) / N
        sen = TP / (TP + FN)
        spc = TN / (TN + FP)
        prc = TP / (TP + FP)
        f1s = 2 * (prc * sen) / (prc + sen)
        mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))

        # metrics that are based on scores,
        auc_roc = roc_auc_score(y_true, scores)
        auc_pr = average_precision_score(y_true, scores)

        met['Confusion Matrix'].append(cnf)
        met['Accuracy'].append(acc)
        met['Precision'].append(prc)
        met['Sensitivity/Recall'].append(sen)
        met['Specificity'].append(spc)
        met['F1 score'].append(f1s)
        met['MCC'].append(mcc)
        met['AUC (ROC)'].append(auc_roc)
        met['AUC (PR)'].append(auc_pr)

    # calculate mean & std
    for k in met.keys():
        if k not in ['Confusion Matrix']:
            arr = np.array(met[k])
            met[k] = (arr.mean(), arr.std())

    return met


def pr_interp(rc_, rc, pr):

    pr_ = np.zeros_like(rc_)
    locs = np.searchsorted(rc, rc_)

    for idx, loc in enumerate(locs):
        l = loc - 1
        r = loc
        r1 = rc[l] if l > -1 else 0
        r2 = rc[r] if r < len(rc) else 1
        p1 = pr[l] if l > -1 else 1
        p2 = pr[r] if r < len(rc) else 0

        t1 = (1 - p2) * r2 / p2 / (r2 - r1) if p2 * (r2 - r1) > 1e-16 else (1 - p2) * r2 / 1e-16
        t2 = (1 - p1) * r1 / p1 / (r2 - r1) if p1 * (r2 - r1) > 1e-16 else (1 - p1) * r1 / 1e-16
        t3 = (1 - p1) * r1 / p1 if p1 > 1e-16 else (1 - p1) * r1 / 1e-16

        a = 1 + t1 - t2
        b = t3 - t1 * r1 + t2 * r1
        pr_[idx] = rc_[idx] / (a * rc_[idx] + b)

    return pr_


def get_roc_info(y_true_all, scores_all):

    fpr_pt = np.linspace(0, 1, 1001)
    tprs, aucs = [], []

    for i in range(len(y_true_all)):
        y_true = y_true_all[i]
        scores = scores_all[i]
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=scores, drop_intermediate=True)
        tprs.append(interp(fpr_pt, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    tprs_mean = np.mean(tprs, axis=0)
    tprs_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tprs_mean + tprs_std, 1)
    tprs_lower = np.maximum(tprs_mean - tprs_std, 0)
    auc_mean = auc(fpr_pt, tprs_mean)
    auc_std = np.std(aucs)

    rslt = {
        'xs': fpr_pt,
        'ys_mean': tprs_mean,
        'ys_upper': tprs_upper,
        'ys_lower': tprs_lower,
        'auc_mean': auc_mean,
        'auc_std': auc_std
    }

    return rslt


def get_pr_info(y_true_all, scores_all):

    rc_pt = np.linspace(0, 1, 1001)
    rc_pt[0] = 1e-16
    prs = []
    aps = []

    for i in range(len(y_true_all)):
        y_true = y_true_all[i]
        scores = scores_all[i]
        pr, rc, _ = precision_recall_curve(y_true=y_true, probas_pred=scores)
        aps.append(average_precision_score(y_true=y_true, y_score=scores))
        pr, rc = pr[::-1], rc[::-1]
        prs.append(pr_interp(rc_pt, rc, pr))

    prs_mean = np.mean(prs, axis=0)
    prs_std = np.std(prs, axis=0)
    prs_upper = np.minimum(prs_mean + prs_std, 1)
    prs_lower = np.maximum(prs_mean - prs_std, 0)
    aps_mean = np.mean(aps)
    aps_std = np.std(aps)

    rslt = {
        'xs': rc_pt,
        'ys_mean': prs_mean,
        'ys_upper': prs_upper,
        'ys_lower': prs_lower,
        'auc_mean': aps_mean,
        'auc_std': aps_std
    }

    return rslt


def l1_regularizer(model, lambda_l1=0.01):
    ''' LASSO '''

    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1