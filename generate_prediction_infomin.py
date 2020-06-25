import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from PyContrast.pycontrast.networks.build_backbone import build_model
from torch_utils import get_loaders_imagenet

device, dtype = 'cuda:0', torch.float32


def get_model(model='resnet50_infomin'):
    if model == 'resnet50_infomin':
        parser = argparse.ArgumentParser(description='CPC')
        args = parser.parse_args()

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


def eval(model, loader):
    reses = []
    labs = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        output = model.forward(data, mode=2)
        reses.append(output.detach().cpu().numpy())
        labs.append(target.detach().cpu().numpy())
        break

    rss = np.concatenate(reses, axis=0)
    lbs = np.concatenate(labs, axis=0)
    return rss, lbs


imagenet_path = '/home/vista/Datasets/ILSVRC/Data/CLS-LOC'
imagenet_path = '/home/chaimb/ILSVRC/Data/CLS-LOC'

def eval_and_save(model='resnet50_infomin'):
    mdl = get_model(model)
    train_loader, val_loader = get_loaders_imagenet(imagenet_path, 32, 32, 224, 8, 1, 0)
    train_embs, train_labs = eval(mdl, train_loader)
    val_embs, val_labs = eval(mdl, val_loader)
    os.makedirs('./results', exist_ok=True)
    np.savez(os.path.join('./results', model + '.npz'), train_embs=train_embs, train_labs=train_labs, val_embs=val_embs,
             val_labs=val_labs)


eval_and_save('resnet50_infomin')
