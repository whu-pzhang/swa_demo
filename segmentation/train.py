import datetime
import os
import os.path as osp
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import distributed as dist
import torchvision

import utils
import transforms as T
import models


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


def get_dataset(data_root, name, image_set, transforms=None):
    data_map = {
        'voc': (data_root, torchvision.datasets.VOCSegmentation, 21),
    }

    path, ds_fn, num_classes = data_map[name]

    dataset = ds_fn(path, image_set=image_set, transforms=transforms)
    return dataset, num_classes


def get_transforms(train=False,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    base_size = 520
    crop_size = 480
    min_size = int(0.5 * base_size)
    max_size = int(2.0 * base_size)

    if train:
        tsfs = [
            T.RandomResize(min_size, max_size),
            T.RandomHorizontalFlip(0.5),
            T.RandomCrop(crop_size),
        ]
    else:
        tsfs = [T.RandomResize(base_size, base_size)]

    tsfs.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(tsfs)


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
                    device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=', ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'Epoch: [{epoch}]'
    for image, target in metric_logger.log_every(data_loader, print_freq,
                                                 header, device):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]['lr'])


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter=', ')
    header = 'Val: '
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header,
                                                     device):
            image, target = image.to(device), target.to(device)
            output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    set_random_seed(args.seed)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path,
                                       args.dataset,
                                       image_set='train',
                                       transforms=get_transforms(True))
    dataset_val, _ = get_dataset(args.data_path,
                                 args.dataset,
                                 image_set='val',
                                 transforms=get_transforms(False))

    if args.distributed:
        train_sampler = dist.DistributedSampler(dataset)
        val_sampler = dist.DistributedSampler(dataset_val)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        #   collate_fn=utils.collate_fn,
        drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=1,
                                                  sampler=val_sampler,
                                                  num_workers=args.workers)

    model = models.__dict__[args.model](pretrained=args.pretrained,
                                        in_channels=3,
                                        num_classes=num_classes)
    model.to(device)

    # setup swa
    if args.model_swa:
        model_swa = torch.optim.swa_utils.AveragedModel(model)
        swa_lr_scheduler = torch.optim.swa_utils.SWALR(optimizer)
        if args.resume:
            pass

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {
            "params": [
                p for p in model_without_ddp.backbone.parameters()
                if p.requires_grad
            ]
        },
        {
            "params": [
                p for p in model_without_ddp.head.parameters()
                if p.requires_grad
            ]
        },
    ]
    if args.aux_loss:
        params = [
            p for p in model_without_ddp.aux_head.parameters()
            if p.requires_grad
        ]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(data_loader) * args.epochs))**0.9)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'],
                                          strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        confmat = evaluate(model,
                           data_loader_test,
                           device=device,
                           num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
                        device, epoch, args.print_freq)
        if epoch % args.eval_interval == 0:
            confmat = evaluate(model,
                               data_loader_val,
                               device=device,
                               num_classes=num_classes)
            print(confmat)

        if epoch % args.save_interval == 0:
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                }, os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Segmentation Training')

    # Dataset / Model parameters
    parser.add_argument('--data-path', default='/data1/', help='dataset path')
    parser.add_argument('--dataset', default='voc', help='dataset name')
    parser.add_argument('--model',
                        default='lraspp_mobilenetv3_large',
                        help='model')
    parser.add_argument('--aux-loss',
                        action='store_true',
                        help='auxiliar loss')
    parser.add_argument("--pretrained",
                        action="store_true",
                        default=True,
                        help="Use pre-trained models from the model-zoo")
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
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 16)')

    # Learning rate scheduler parameters
    parser.add_argument('--lr',
                        default=0.01,
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

    # Model Exponential Moving Average
    parser.add_argument('--model-swa',
                        action='store_true',
                        default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument(
        '--model-swa-decay',
        type=float,
        default=0.9998,
        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--eval-interval',
                        default=5,
                        type=int,
                        help='evaluate interval')
    parser.add_argument('--save-interval',
                        default=10,
                        type=int,
                        help='checkpoint interval')
    parser.add_argument('--print-freq',
                        default=20,
                        type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='S',
                        help='random seed')

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


if __name__ == '__main__':
    args = parse_args()
    main(args)
