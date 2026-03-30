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
from multiview_detector.feat_backbone_v2 import Backbone
from accelerate import Accelerator

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

    backbone = Backbone(train_set, 1)
    backbone.to("cuda")

    args = nested_dict_to_namespace(config)
    model, criterion, postprocessors = build_model(args,train_set)

    output_dir = Path("log")

    best_mota = -float('inf')
    best_epoch = -1

    logdir = output_dir / 'log'
    # 遍历 log 文件夹
    for ckpt_file in sorted(output_dir.glob("checkpoint_epoch_*.pth")):
        epoch_num = int(ckpt_file.stem.split('_')[-1])
        if epoch_num < 40:
            continue

        print(f"=== Testing epoch {epoch_num} ===")

        # 初始化模型
        obj_detect_args = nested_dict_to_namespace(config)
        obj_detect_args.tracking = False
        obj_detector, _, obj_detector_post = build_model(obj_detect_args, test_set)

        # 加载权重
        obj_detect_checkpoint = torch.load(ckpt_file, map_location='cuda')
        obj_detect_state_dict = obj_detect_checkpoint['model']

        new_state_dict = {}
        for k, v in obj_detect_state_dict.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v

        obj_detector.load_state_dict(new_state_dict)
        obj_detector.cuda()

        obj_detect_checkpoint = torch.load(ckpt_file, map_location='cuda')
        obj_detect_state_dict = obj_detect_checkpoint['backbone']

        new_state_dict = {}
        for k, v in obj_detect_state_dict.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v

        backbone.load_state_dict(new_state_dict)
        backbone.cuda()

        # 测试
        trainer_test = PerspectiveTrainer(obj_detector, backbone,criterion, logdir, denormalize, args, 0.4)
        mota = trainer_test.test(test_loader, args, obj_detector_post)

        print(f"[Epoch {epoch_num}] Test MOTA: {mota:.4f}")

        # 记录最优 MOTA
        if mota > best_mota:
            print(f"New best MOTA {mota:.4f} at epoch {epoch_num}")
            best_mota = mota
            best_epoch = epoch_num

    # 最终输出
    print(f"=== Best MOTA: {best_mota:.4f} at epoch {best_epoch} ===")


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
