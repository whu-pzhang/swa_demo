import datetime
import os
import os.path as osp
import time

import torch
import torch.nn as non_blocking
import torchvision


def get_dataset(data_root, name, image_set, transform):
    data_map = {
        'voc': (data_root, torchvision.datasets.VOCSegmentation, 21),
    }

    path, ds_fn, num_classes = data_map[name]

    dataset = ds_fn(path, image_set=image_set, transforms=transforms)
    return dataset, num_classes


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                    print_freq):
    model.train()
