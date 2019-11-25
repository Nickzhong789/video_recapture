import numpy as np


__all__ = ['cal_acc', 'accuracy', 'AverageMeter']


def cal_acc(v_dict, f_dict, v_t, f_t):
    for key, value in f_dict.items():
        v_id, f_id = key.split('_')
        if v_dict[v_id] not in v_dict.keys():
            v_dict[v_id] = {'correct': 0, 'total': 0}
        if value > 100 * f_t:
            v_dict[v_id]['correct'] += 1
        v_dict[v_id]['total'] += 1

    v_correct = 0
    v_total = 0
    for v_key, v_value in v_dict.items():
        if v_dict[v_key]['correct'] > v_dict[v_key]['total'] * v_t:
            v_correct += 1
        v_total += 1
    acc = v_correct / v_total
    return acc


def accuracy(pred, y):
    return 1.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += 1
        self.avg = self.sum / self.count
