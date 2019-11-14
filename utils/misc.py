from os.path import join
import shutil
import torch


def adjust_learing_rate(optimizer, epoch, lr, step_epoch):
    lr = lr * (0.1 ** min((epoch // step_epoch), 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pkl', snapshot=None):
    filepath = join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, join(checkpoint, 'checkpoint_{}.pkl'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, join(checkpoint, 'model_best.pkl'))
        print("Save checkpoint at epoch {}".format(state['epoch']))
