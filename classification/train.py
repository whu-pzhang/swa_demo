import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

import utils


def get_tranforms(train=True):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
        ])


def get_loader(args):
    dataset = torchvision.datasets.CIFAR10(root='./data',
                                           train=True,
                                           download=True,
                                           transform=get_tranforms(True))
    dataset_val = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               download=True,
                                               transform=get_tranforms(False))
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        val_sampler = torch.utils.data.DistributedSampler(dataset_val)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=args.batch_size,
                                                  sampler=val_sampler,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    return data_loader_train, data_loader_val


def train_one_epoch(model,
                    criterion,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq,
                    swa_cfg=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=', ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    metric_logger.add_meter(
        'img/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

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
        metric_logger.meters['img/s'].update(batch_size /
                                             (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=', ')
    header = 'Validation:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq,
                                                     header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--max-epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j',
                        '--workers',
                        default=16,
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

    parser.add_argument('--print-freq',
                        default=10,
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

    # distributed training parameters
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url',
                        default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def main(args):
    MAX_EPOCHS = 100
    LR = 0.1
    SWA = False
    SWA_START = 80

    device = torch.device(args.device)

    # dataset
    data_loader_train, data_loader_valid = get_loader(args)

    # model
    model = torchvision.models.resnet18()
    model.to(device)

    # sync batchnorm
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=LR,
                          momentum=0.9,
                          weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=MAX_EPOCHS)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpus])
        model_without_ddp = model.module

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, val_loader, device=device)
        return

    if SWA:
        swa_cfg = {
            'swa_model': optim.swa_utils.AveragedModel(model),
            'swa_start': SWA_START,
            'swa_scheduler': optim.swa_utils.SWALR(optimizer, swa_lr=0.01)
        }
    else:
        swa_cfg = None

    # Start training
    for epoch in range(args.start_epoch, args.max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, lr_scheduler,
                        train_loader, 'cuda', epoch + 1, 50, swa_cfg)
        evaluate(model, criterion, val_loader, 'cuda', print_freq=20)

    if SWA:
        optim.swa_utils.update_bn(train_loader, swa_cfg['swa_model'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
