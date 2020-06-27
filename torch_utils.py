import json
import os

import requests
import torch
from torch.utils.data import DistributedSampler
from torchvision import datasets
from torchvision.transforms import transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def woof_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.35, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform_imagenet(augment=True, input_size=224):
    normalize = __imagenet_stats
    scale_size = int(input_size / 0.875)
    if augment:
        return woof_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


def get_loaders_imagenet(dataroot, val_batch_size, train_batch_size, input_size, workers, num_nodes, local_rank):
    # TODO: pin-memory currently broken for distributed
    pin_memory = False
    # TODO: datasets.ImageNet
    val_data = datasets.ImageFolder(root=os.path.join(dataroot, 'val'),
                                    transform=get_transform_imagenet(False, input_size))
    val_sampler = DistributedSampler(val_data, num_nodes, local_rank)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler,
                                             num_workers=workers, pin_memory=pin_memory)

    train_data = datasets.ImageFolder(root=os.path.join(dataroot, 'train'),
                                      transform=get_transform_imagenet(input_size=input_size))
    train_sampler = DistributedSampler(train_data, num_nodes, local_rank)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler,
                                               num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader


def get_loaders_objectnet(dataroot, imagenet_dataroot, val_batch_size, input_size, workers, num_nodes, local_rank):
    # TODO: pin-memory currently broken for distributed
    pin_memory = False
    # TODO: datasets.ImageNet
    val_data_im = datasets.ImageFolder(root=os.path.join(dataroot, 'val'),
                                       transform=get_transform_imagenet(False, input_size))
    # TODO: datasets.ImageNet
    val_data = datasets.ImageFolder(root=os.path.join(dataroot, 'images'),
                                    transform=get_transform_imagenet(False, input_size))
    val_sampler = DistributedSampler(val_data, num_nodes, local_rank)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler,
                                             num_workers=workers, pin_memory=pin_memory)
    imagenet_to_objectnet, objectnet_to_imagenet, objectnet_both, imagenet_both = objectnet_imagenet_mappings(dataroot,
                                                                                                              val_data,
                                                                                                              val_data_im)
    return val_loader, imagenet_to_objectnet, objectnet_to_imagenet, objectnet_both, imagenet_both


def objectnet_imagenet_mappings(dataroot, object_data, imagenet_data):
    import numpy as np
    mappings = os.path.join(dataroot, 'mappings')
    object_to_imagenet = json.load(open(os.path.join(mappings, 'objectnet_to_imagenet_1k.json')))
    folder_to_object = json.load(open(os.path.join(mappings, 'folder_to_objectnet_label.json')))
    map_url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    response = json.loads(requests.get(map_url).text)
    name_map = {}
    name_to_syn = {}
    name_to_num = {}
    for r in response:
        name_map[response[r][0]] = response[r][1]
        name_to_syn[response[r][1]] = response[r][0]
        # print(response[r][1].replace('_',' '))
        name_to_num[response[r][1].replace('_', ' ')] = imagenet_data.class_to_idx[response[r][0]]


    imagenet_to_name = []
    imagenet_to_objectnet = - np.ones(1000, dtype=int)
    objectnet_to_imagenet = {}

    name_to_imagenet = {}
    for i, cl in enumerate(open(os.path.join(mappings, 'imagenet_to_label_2012_v2'))):
        cl = cl.strip()
        imagenet_to_name.append(cl)
        name_to_imagenet[cl] = i

    cnt_both, cnt = 0, 0
    objectnet_both = []
    imagenet_both = []
    for cl in object_data.class_to_idx:
        obj = folder_to_object[cl]
        if obj in object_to_imagenet:
            imagenet_classes = [s.strip() for s in object_to_imagenet[obj].split(';')]
            pt_classes = [name_to_num[ic.split(',')[0].strip()] for ic in imagenet_classes]
            objectnet_to_imagenet[object_data.class_to_idx[cl]] = pt_classes
            cnt_both += 1
            objectnet_both.append(object_data.class_to_idx[cl])
            for icl in pt_classes:
                imagenet_both.append(icl)
                imagenet_to_objectnet[icl] = object_data.class_to_idx[cl]
        else:
            objectnet_to_imagenet[object_data.class_to_idx[cl]] = []
            cnt += 1

    return imagenet_to_objectnet, objectnet_to_imagenet, objectnet_both, imagenet_both
