import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from PyContrast.pycontrast.networks.build_backbone import build_model
from torch_utils import get_loaders_imagenet, get_loaders_objectnet

device, dtype = 'cuda:0', torch.float32
gh = 'rwightman/pytorch-image-models'

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
    else:
        raise ValueError('Wrong model')


def eval(model, loader):
    reses = []
    labs = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        output = model.forward_features(data)
        reses.append(output.detach().cpu().numpy())
        labs.append(target.detach().cpu().numpy())

    rss = np.concatenate(reses, axis=0)
    lbs = np.concatenate(labs, axis=0)
    return rss, lbs


imagenet_path = '/home/vista/Datasets/ILSVRC/Data/CLS-LOC'
imagenet_path = '/home/chaimb/ILSVRC/Data/CLS-LOC'
objectnet_path = '/home/chaimb/objectnet-1.0'


def eval_and_save(model='resnet50_infomin', dim=224):
    mdl = torch.hub.load(gh, model, pretrained=True)
    bs = 16
    train_loader, val_loader = get_loaders_imagenet(imagenet_path, bs, bs, dim, 8, 1, 0)
    obj_loader, _, _, _, _ = get_loaders_objectnet(objectnet_path, imagenet_path, bs, dim, 8, 1, 0)
    train_embs, train_labs = eval(mdl, train_loader)
    val_embs, val_labs = eval(mdl, val_loader)
    obj_embs, obj_labs = eval(mdl, obj_loader)
    os.makedirs('./results', exist_ok=True)
    np.savez(os.path.join('./results', model + '.npz'), train_embs=train_embs, train_labs=train_labs, val_embs=val_embs,
             val_labs=val_labs, obj_embs=obj_embs, obj_labs=obj_labs)


models = torch.hub.list(gh)
# models to run:
# tf_efficientnet_l2_ns_475
# gluon_resnet152_v1s
# ig_resnext101_32x48d
parser = argparse.ArgumentParser(description='IM')
parser.add_argument('--model', dest='model', type=str, default='resnext152_infomin',
                    help='Model: one of ' + ', '.join(models))
args = parser.parse_args()

dims = {'tf_efficientnet_l2_ns_475': 475, 'gluon_resnet152_v1s': 224, 'ig_resnext101_32x48d': 224}
eval_and_save(args.model, dim=dims[args.model])
