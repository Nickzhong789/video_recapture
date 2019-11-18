import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from model.resnet import ResNet
from dataset import CFDataset
from utils.osutil import *
from utils.misc import adjust_learing_rate, save_checkpoint
from utils.progress.bar import Bar
from utils.eval import AverageMeter, cal_auc
from utils.logger import Logger

import time
import argparse


best_acc = 0


def main(args):
    global best_acc

    if not isdir(args.model):
        mkdir_p(args.model)

    print("==> create model ResComicFaceNet('{}', {})".format(args.level1, args.level2))

    model = ResComicFaceNet(level1=args.level1, level2=args.level2)

    criterion = nn.L1Loss(size_average=True).cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )
    title = 'comic-face'

    if args.resume:
        if isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.model, 'log.txt'), title=title, resume=True)
        else:
            print("==> checkpoint not found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.model, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    cudnn.benchmark = True
    print('  Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    trainPath = args.training_dataset
    train_transform = transforms.Compose([
        transforms.Resize((512, 448)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        CFDataset(trainPath, 'annotations.csv', train=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    valid_transform = transforms.Compose([
        transforms.Resize((512, 448)),
        # transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    valid_loader = DataLoader(
        CFDataset(trainPath, 'annotations.csv', train=False, transform=valid_transform),
        batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learing_rate(optimizer, epoch, lr, args.step_epoch)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = validate(valid_loader, model, criterion)

        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best, checkpoint=args.model)

    logger.close()


def train(model, train_loader, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.cuda()
    model.train()

    end = time.time()

    y_pred = torch.Tensor()
    y = torch.Tensor()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if not type(data) in (tuple, list):
            data = (data,)

        input_var = tuple(d.cuda() for d in data)

        target = target.float()
        target_var = target.cuda()

        optimizer.zero_grad()
        output = model(*input_var)

        loss = criterion(output, target_var)
        out = output.data.cpu()
        y_pred = torch.cat((y_pred, out))
        y = torch.cat((y, target))

        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |' \
                     ' Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss,
        )
        bar.next()

    bar.finish()

    acc = cal_auc(y_pred.numpy(), y.numpy())

    return losses.avg, acc


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()

    y_pred = torch.Tensor()
    y = torch.Tensor()

    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (data, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        if not type(data) in (tuple, list):
            data = (data,)

        input_var = tuple(d.cuda() for d in data)

        target = target.float()
        target_var = target.cuda()

        output = model(*input_var)

        loss = criterion(output, target_var)
        out = output.data.cpu()
        y_pred = torch.cat((y_pred, out))
        y = torch.cat((y, target))

        losses.update(loss.item())

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |' \
                     ' Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss,
        )
        bar.next()

    bar.finish()

    acc = cal_auc(y_pred.numpy(), y.numpy())
    print(y)

    return losses.avg, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Comic-face')
    parser.add_argument('--level1', type=int, default=18, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--level2', type=int, default=18, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight-decay', type=float, default=0.00005, metavar='N',
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--resume', type=str, default='', metavar='N',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--step_epoch', default=30, type=int, metavar='N',
                        help='decend the lr in epoch number')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='manual epoch number (default: 100)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--training-dataset', type=str, default='./data/af2019-ksyun-training-20190416',
                        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--model', type=str, default='./checkpoint/res18-18',
                        help='Path to the data directory containing aligned face patches.')

    main(parser.parse_args())
