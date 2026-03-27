# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import pdb
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork,
                                                     LastLevelMaxPool)
import numpy as np
from ..util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from timm.utils.model import freeze_batch_norm_2d
from kornia.geometry.transform import warp_perspective
from multiview_detector import utils
from multiview_detector.utils import vox
from multiview_detector.utils import basic


class Joiner(nn.Sequential):
    def __init__(self,position_embedding):
        super().__init__(position_embedding)
        self.num_channels = [64,128,256]

    def forward(self, tensor_dict: Dict[str, NestedTensor]):
        out: List[NestedTensor] = []
        pos = []

        for x in tensor_dict.values():
            out.append(x)
            # position encoding
            pos.append(self[0](x).to(x.tensors.dtype).squeeze())

        return out, pos


def build_backbone(args):

    position_embedding = build_position_encoding(args)
    model = Joiner(position_embedding)
    return model
