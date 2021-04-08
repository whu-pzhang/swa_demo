import os
import os.path as osp
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

import models
import utils


def get_tranforms(train=True):
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


def get_data(args):
    dataset = torchvision.datasets.CIFAR100(root='./data',
                                            train=True,
                                            download=True,
                                            transform=get_tranforms(True))
    dataset_val = torchvision.datasets.CIFAR100(root='./data',
                                                train=False,
                                                download=True,
                                                transform=get_tranforms(False))
    train_sampler = torch.utils.data.RandomSampler(dataset)
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    return dataset, dataset_val, train_sampler, val_sampler


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                    print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=', ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))

    header = f'Epoch: [{epoch}]'
    for image, target in metric_logger.log_every(data_loader, print_freq,
                                                 header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1 = utils.accuracy(output, target, topk=(1, ))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1[0], n=batch_size)


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=', ')
    header = 'Validation:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq,
                                                     header):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)

            acc1 = utils.accuracy(output, target, topk=(1, ))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1[0], n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18d', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # SWA
    parser.add_argument('--swa',
                        action='store_true',
                        help='swa usage flag (default: off)')
    parser.add_argument('--swa-start',
                        type=float,
                        default=160,
                        metavar='N',
                        help='SWA start epoch number (default: 161)')
    parser.add_argument('--swa-lr',
                        type=float,
                        default=0.05,
                        metavar='LR',
                        help='SWA LR (default: 0.05)')
    parser.add_argument(
        '--swa-c-epochs',
        type=int,
        default=1,
        metavar='N',
        help=
        'SWA model collection frequency/cycle length in epochs (default: 1)')
    parser.add_argument('--swa-on-cpu',
                        action='store_true',
                        help='store swa model on cpu flag (default: off)')
    #
    parser.add_argument('--eval-interval',
                        default=5,
                        type=int,
                        help='evaluate interval')
    parser.add_argument('--save-interval',
                        default=25,
                        type=int,
                        help='checkpoint interval')
    parser.add_argument('--print-freq',
                        default=100,
                        type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument("--test-only",
                        dest="test_only",
                        help="Only test the model",
                        action="store_true")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='S',
                        help='random seed (default: 42)')

    args = parser.parse_args()

    return args


def main(args):
    utils.mkdir(args.output_dir)
    set_random_seed(args.seed)

    device = torch.device(args.device)

    # dataset
    dataset, dataset_val, train_sampler, val_sampler = get_data(args)
    data_loader_train = torch.utils.data.DataLoader(dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    # model
    # model = timm.create_model(args.model, num_classes=100)
    # model_cfg = getattr(models, args.model)
    # model = model_cfg.base(*model_cfg.args,
    #                        num_classes=100,
    #                        **model_cfg.kwargs)
    model = models.__dict__[args.model](num_classes=100)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=args.epochs)

    if args.swa:
        swa_model = optim.swa_utils.AveragedModel(model)
        swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=args.swa_lr)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.swa:
            swa_model.load_state_dict(checkpoint['swa_model'])

    if args.test_only:
        evaluate(model, criterion, data_loader_val, device=device)
        if args.swa:
            print('SWA eval')
            evaluate(swa_model, criterion, data_loader_val, device=device)
        return

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader_train, device,
                        epoch + 1, args.print_freq)

        if not (epoch + 1) % args.eval_interval:
            evaluate(model, criterion, data_loader_val, device, print_freq=50)

        if args.swa and (epoch + 1) > args.swa_start:
            swa_scheduler.step()
        else:
            lr_scheduler.step()

        if args.swa and (epoch + 1) > args.swa_start and (
                epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
            swa_model.update_parameters(model)

            if not (epoch +
                    1) % args.eval_interval or epoch == args.epochs - 1:
                optim.swa_utils.update_bn(data_loader_train, swa_model, device)
                print('SWA eval')
                evaluate(swa_model,
                         criterion,
                         data_loader_val,
                         device,
                         print_freq=50)

        if args.output_dir and (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'swa_model': swa_model.state_dict() if args.swa else None,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            torch.save(checkpoint,
                       osp.join(args.output_dir, f'model_{epoch+1}.pth'))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
