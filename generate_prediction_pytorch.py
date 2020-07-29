import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from PyContrast.pycontrast.networks.build_backbone import build_model
from torch_utils import get_loaders_imagenet, get_loaders_objectnet

device, dtype = 'cuda:0', torch.float32


def get_model(model='resnet50_infomin'):
    if model == 'resnet50_infomin':
        args = SimpleNamespace()

        args.jigsaw = True
        args.arch, args.head, args.feat_dim = 'resnet50', 'mlp', 128
        args.mem = 'moco'
        args.modal = 'RGB'
        model, _ = build_model(args)
        cp = torch.load('checkpoints/InfoMin_800.pth')

        sd = cp['model']
        new_sd = {}
        for entry in sd:
            new_sd[entry.replace('module.', '')] = sd[entry]
        model.load_state_dict(new_sd, strict=False)  # no head, don't need linear model

        model = model.to(device=device)
        return model
    elif model == 'resnext152_infomin':
        args = SimpleNamespace()

        args.jigsaw = True
        args.arch, args.head, args.feat_dim = 'resnext152v1', 'mlp', 128
        args.mem = 'moco'
        args.modal = 'RGB'
        model, _ = build_model(args)
        cp = torch.load('checkpoints/InfoMin_resnext152v1_e200.pth')

        sd = cp['model']
        new_sd = {}
        for entry in sd:
            new_sd[entry.replace('module.', '')] = sd[entry]
        model.load_state_dict(new_sd, strict=False)  # no head, don't need linear model

        model = model.to(device=device)
        return model
    elif model == 'resnet50_mocov2':
        args = SimpleNamespace()

        args.jigsaw = False
        args.arch, args.head, args.feat_dim = 'resnet50', 'linear', 2048
        args.mem = 'moco'
        args.modal = 'RGB'
        model, _ = build_model(args)
        cp = torch.load('checkpoints/MoCov2.pth')

        sd = cp['model']
        new_sd = {}
        for entry in sd:
            new_sd[entry.replace('module.', '')] = sd[entry]
        model.load_state_dict(new_sd, strict=False)  # no head, don't need linear model

        model = model.to(device=device)
        return model
    elif model == 'resnet50_swav':
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
        model = model.to(device=device)
        return model
    else:
        raise ValueError('Wrong model')


def eval_swav(model, loader):
    reses = []
    labs = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        output = model.forward_backbone(data)
        reses.append(output.detach().cpu().numpy())
        labs.append(target.detach().cpu().numpy())

    rss = np.concatenate(reses, axis=0)
    lbs = np.concatenate(labs, axis=0)
    return rss, lbs


def eval(model, loader, kwargs):
    reses = []
    labs = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        output = model.forward(data, mode=2)
        reses.append(output.detach().cpu().numpy())
        labs.append(target.detach().cpu().numpy())

    rss = np.concatenate(reses, axis=0)
    lbs = np.concatenate(labs, axis=0)
    return rss, lbs


imagenet_path = '/home/vista/Datasets/ILSVRC/Data/CLS-LOC'
imagenet_path = '/home/chaimb/ILSVRC/Data/CLS-LOC'
objectnet_path = '/home/chaimb/objectnet-1.0'


def eval_and_save(model='resnet50_infomin'):
    mdl = get_model(model)
    bs = 32 if model in ['resnet50_infomin'] else 16
    train_loader, val_loader = get_loaders_imagenet(imagenet_path, bs, bs, 224, 8, 1, 0)
    obj_loader, _, _, _, _ = get_loaders_objectnet(objectnet_path, imagenet_path, bs, 224, 8, 1, 0)
    eval_f = eval_swav if 'swav' in model else eval
    train_embs, train_labs = eval_f(mdl, train_loader)
    val_embs, val_labs = eval_f(mdl, val_loader)
    obj_embs, obj_labs = eval_f(mdl, obj_loader)
    os.makedirs('./results', exist_ok=True)
    np.savez(os.path.join('./results', model + '.npz'), train_embs=train_embs, train_labs=train_labs, val_embs=val_embs,
             val_labs=val_labs, obj_embs=obj_embs, obj_labs=obj_labs)


models = ['resnet50_infomin', 'resnext152_infomin', 'resnet50_mocov2', 'resnet50_swav']
parser = argparse.ArgumentParser(description='IM')
parser.add_argument('--model', dest='model', type=str, default='resnext152_infomin',
                    help='Model: one of ' + ', '.join(models))
args = parser.parse_args()

eval_and_save(args.model)
