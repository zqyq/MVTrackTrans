"""
Multi-view Crowd Tracking Transformer with View-Ground Interactions
Under Large Real-World Scenes

Authors:
Qi Zhang, Jixuan Chen, Kaiyi Zhang, Xinquan Yu,
Antoni B. Chan, Hui Huang

Affiliations:
1. College of Computer Science and Software Engineering, Shenzhen University, China
2. Department of Computer Science, City University of Hong Kong, China

Conference:
CVPR 2026

Description:
This file is part of the official implementation of the proposed
Multi-view Crowd Tracking Transformer (MVTrackTrans).

If you use this code, please cite our paper.
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import tqdm
import numpy as np
import torch
from argparse import Namespace
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer
import sacred
from multiview_detector.util.misc import nested_dict_to_namespace
from multiview_detector.build import build_model
from multiview_detector.util import misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from multiview_detector.feat_backbone_v2 import Backbone

ex = sacred.Experiment('train')
ex.add_named_config('reid','cfgs/track_reid.yaml')
ex.add_config('cfgs/train.yaml')
ex.add_config('cfgs/track.yaml')
ex.add_named_config('deformable', 'cfgs/train_deformable.yaml')
ex.add_named_config('tracking', 'cfgs/train_tracking.yaml')




def train(args: Namespace) -> None:
    args.dataset ='mot17'
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True
    print()
    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    #train_trans = T.Compose([T.Resize([720, 1280]),T.ToTensor(),])
    if 1:
        data_path = os.path.expanduser('/mnt/d/jixuan/dataset/Citystreet')
        base = Citystreet(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4,img_reduce = 4)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4,img_reduce = 4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                                               num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True)

    accelerator = Accelerator()

    args = nested_dict_to_namespace(config)
    model, criterion, postprocessors = build_model(args,train_set)
    backbone = Backbone(train_set,1)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names + args.lr_linear_proj_names + [
                'layers_track_attention'])
                    and p.requires_grad],
         "lr": args.lr},
        {"params": [p for n, p in model.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
         "lr": args.lr * args.lr_linear_proj_mult},
        {"params": [p for n, p in backbone.named_parameters() if p.requires_grad], "lr": args.lr_backbone},

    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs
    warmup_steps = int(0.05 * total_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    train_loader, test_loader, model, backbone,optimizer = accelerator.prepare(
        train_loader, test_loader, model, backbone,optimizer
    )


    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop])

    logdir = "log"
    save_dir = 'loss_curves'
    output_dir = Path("log")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    trainer = PerspectiveTrainer(model,backbone, criterion, logdir, denormalize, args, 0.4)

    loss = []

    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        print(f'=== Epoch {epoch} Training ===')


        losses = trainer.train(epoch, train_loader, optimizer, args, accelerator)
        loss.append(losses)
        lr_scheduler.step()

        if accelerator.is_main_process:
            plt.figure()
            plt.plot(range(1, len(loss) + 1), loss, label='Train Loss', color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss (Epoch {epoch})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_dir}/loss_epoch_{epoch:03d}.png')
            plt.close()

            # === 保存当前 epoch 模型 ===
            model_to_save = accelerator.unwrap_model(model)
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({
                'model': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'backbone': backbone.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)




@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

if __name__ == '__main__':
    # TODO: hierachical Namespacing for nested dict
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    # args.train = Namespace(**config['train'])
    train(args)
