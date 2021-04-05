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


def get_loader(train=True):
    dataset = torchvision.datasets.CIFAR10(root='./data',
                                           train=train,
                                           download=True,
                                           transform=get_tranforms(train))
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=128,
                                         shuffle=True if train else False,
                                         num_workers=2)

    return loader


def train_one_epoch(model,
                    criterion,
                    optimizer,
                    lr_scheduler,
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

        lr_scheduler.step()

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

    args = parser.parse_args()

    return args


def main():
    MAX_EPOCHS = 100
    LR = 0.1
    SWA = False
    SWA_START = 80

    # dataset
    train_loader = get_loader(train=True)
    val_loader = get_loader(train=False)

    # model
    model = torchvision.models.resnet18()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=LR,
                          momentum=0.9,
                          weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=MAX_EPOCHS)
    if SWA:
        swa_cfg = {
            'swa_model': optim.swa_utils.AveragedModel(model),
            'swa_start': SWA_START,
            'swa_scheduler': optim.swa_utils.SWALR(optimizer, swa_lr=0.01)
        }
    else:
        swa_cfg = None

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(model, criterion, optimizer, lr_scheduler,
                        train_loader, 'cuda', epoch + 1, 50, swa_cfg)
        evaluate(model, criterion, val_loader, 'cuda', print_freq=20)

    if SWA:
        optim.swa_utils.update_bn(train_loader, swa_cfg['swa_model'])


if __name__ == '__main__':
    main()
