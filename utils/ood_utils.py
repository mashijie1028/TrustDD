import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as sk

recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, net, ood_num_examples, args, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.batch_test and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            output_ = to_np(output)

            # if args.use_xent:
            #     _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            # else:
            #     _score.append(-np.max(smax, axis=1))

            if args.score == 'energy':
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
            elif args.score == 'mls':
                _score.append(-np.max(output_, axis=1))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


# def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
#     '''
#     :param pos: 1's class, class to detect, outliers, or wrongly predicted
#     example scores
#     :param neg: 0's class scores
#     '''

#     auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

#     print('\t\t\t' + method_name)
#     print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
#     print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
#     print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
#     # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures(auroc, aupr, fpr, recall_level=recall_level_default):
    print('FPR{:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: {:.2f}'.format(100 * auroc))
    print('AUPR: {:.2f}'.format(100 * aupr))



def print_measures_with_std(aurocs, auprs, fprs, recall_level=recall_level_default):
    print('FPR{:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs), 100*np.std(fprs)))
    print('AUROC: {:.2f} +/- {:.2f}'.format(100*np.mean(aurocs), 100*np.std(aurocs)))
    print('AUPR: {:.2f} +/- {:.2f}'.format(100*np.mean(auprs), 100*np.std(auprs)))


# print pro
def print_measures_pro(auroc, aupr_in, aupr_out, fpr_in, fpr_out, recall_level=recall_level_default):
    print('FPR(IN){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_in))
    print('FPR(OUT){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_out))
    print('AUROC: {:.2f}'.format(100 * auroc))
    print('AUPR(IN): {:.2f}'.format(100 * aupr_in))
    print('AUPR(OUT): {:.2f}'.format(100 * aupr_out))



def print_measures_with_std_pro(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, recall_level=recall_level_default):
    print('FPR(IN){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_in), 100*np.std(fprs_in)))
    print('FPR(OUT){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_out), 100*np.std(fprs_out)))
    print('AUROC: {:.2f} +/- {:.2f}'.format(100*np.mean(aurocs), 100*np.std(aurocs)))
    print('AUPR(IN): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_in), 100*np.std(auprs_in)))
    print('AUPR(OUT): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_out), 100*np.std(auprs_out)))




def write_measures(auroc, aupr, fpr, file_path, recall_level=recall_level_default):
    with open(file_path, 'a+') as f_log:
        f_log.write('FPR{:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr))
        f_log.write('\n')
        f_log.write('AUROC: {:.2f}'.format(100 * auroc))
        f_log.write('\n')
        f_log.write('AUPR: {:.2f}'.format(100 * aupr))
        f_log.write('\n')



def write_measures_with_std(aurocs, auprs, fprs, file_path, recall_level=recall_level_default):
    with open(file_path, 'a+') as f_log:
        f_log.write('FPR{:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs), 100*np.std(fprs)))
        f_log.write('\n')
        f_log.write('AUROC: {:.2f} +/- {:.2f}'.format(100*np.mean(aurocs), 100*np.std(aurocs)))
        f_log.write('\n')
        f_log.write('AUPR: {:.2f} +/- {:.2f}'.format(100*np.mean(auprs), 100*np.std(auprs)))
        f_log.write('\n')



# write pro
def write_measures_pro(auroc, aupr_in, aupr_out, fpr_in, fpr_out, file_path, recall_level=recall_level_default):
    with open(file_path, 'a+') as f_log:
        f_log.write('FPR(IN){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_in))
        f_log.write('\n')
        f_log.write('FPR(OUT){:d}: {:.2f}'.format(int(100 * recall_level), 100 * fpr_out))
        f_log.write('\n')
        f_log.write('AUROC: {:.2f}'.format(100 * auroc))
        f_log.write('\n')
        f_log.write('AUPR(IN): {:.2f}'.format(100 * aupr_in))
        f_log.write('\n')
        f_log.write('AUPR(OUT): {:.2f}'.format(100 * aupr_out))
        f_log.write('\n')



def write_measures_with_std_pro(aurocs, auprs_in, auprs_out, fprs_in, fprs_out, file_path, recall_level=recall_level_default):
    with open(file_path, 'a+') as f_log:
        f_log.write('FPR(IN){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_in), 100*np.std(fprs_in)))
        f_log.write('\n')
        f_log.write('FPR(OUT){:d}: {:.2f} +/- {:.2f}'.format(int(100*recall_level), 100*np.mean(fprs_out), 100*np.std(fprs_out)))
        f_log.write('\n')
        f_log.write('AUROC: {:.2f} +/- {:.2f}'.format(100*np.mean(aurocs), 100*np.std(aurocs)))
        f_log.write('\n')
        f_log.write('AUPR(IN): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_in), 100*np.std(auprs_in)))
        f_log.write('\n')
        f_log.write('AUPR(OUT): {:.2f} +/- {:.2f}'.format(100*np.mean(auprs_out), 100*np.std(auprs_out)))
        f_log.write('\n')
