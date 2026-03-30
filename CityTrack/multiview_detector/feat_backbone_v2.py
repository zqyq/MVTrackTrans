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
from multiview_detector.util.misc import NestedTensor, is_main_process

from timm.utils.model import freeze_batch_norm_2d
from kornia.geometry.transform import warp_perspective
from multiview_detector import utils
from multiview_detector.utils import vox
from multiview_detector.utils import basic


def freeze_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            freeze_bn(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, n, freeze_batch_norm_2d(module))

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class Backbone(nn.Module):

    def __init__(self,train_set,z_sign):
        super().__init__()

        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        """
        for param in resnet.parameters():
            param.requires_grad = False
        """
        freeze_bn(resnet)
        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.rand_flip = True
        self.resolution = (int(768*0.5), 4, int(640 * 0.5))
        self.Y, self.Z, self.X = self.resolution
        self.bounds = [0, 640, 0, 768, -1700, 0]
        self.scene_centroid = torch.tensor([0.0, 0.0, 0.0]).reshape([1, 3])
        self.vox_utils = [
            vox.VoxelUtil(
                self.Y // (2 * i if 2 * i != 0 else 1),
                self.Z,
                self.X // (2 * i if 2 * i != 0 else 1),
                scene_centroid=self.scene_centroid,
                bounds=self.bounds
            )
            for i in range(3)
        ]
        self.z_sign = z_sign
        self.out_channel = [64, 128, 256]
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        self.dataset = train_set
        dataset = train_set

        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))

        self.cam_compressor = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(64* (2 * i if 2 * i != 0 else 1) * self.num_cam, 64* (2 * i if 2 * i != 0 else 1), kernel_size=1),
                nn.InstanceNorm3d(64* (2 * i if 2 * i != 0 else 1)), nn.ReLU(),
                nn.Conv3d(64* (2 * i if 2 * i != 0 else 1), 64* (2 * i if 2 * i != 0 else 1), kernel_size=1),
            )
            for i in range(3)
        ])

        self.bev_compressor = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(64* (2 * i if 2 * i != 0 else 1)  * self.Z, 64* (2 * i if 2 * i != 0 else 1) , kernel_size=1),
            nn.InstanceNorm2d(64* (2 * i if 2 * i != 0 else 1) ), nn.ReLU(),
            nn.Conv2d(64* (2 * i if 2 * i != 0 else 1) , 64* (2 * i if 2 * i != 0 else 1) , kernel_size=1),
            )
            for i in range(3)
        ])
        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, 64, kernel_size=1, bias=False)



    def forward(self, tensor_list: NestedTensor,cal):
        B, N, C, H, W = tensor_list.tensors.shape
        # reshape tensors
        pix_T_cams, cams_T_global, ref_T_global = cal
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(N, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams_)  # B*S,4,4



        out: Dict[str, NestedTensor] = {}
        world_features = []


        device = tensor_list.tensors.device
        tensor_list.tensors = (tensor_list.tensors - self.mean.to(device)) / self.std.to(device)
        tensor_list.tensors = __p(tensor_list.tensors)
        img_feature_0 = self.layer0(tensor_list.tensors.to("cuda"))
        img_feature_1 = self.layer1(img_feature_0)
        img_feature_2 = self.layer2(img_feature_1)
        img_feature_3 = self.layer3(img_feature_2)
        feat_list = [img_feature_1, img_feature_2, img_feature_3]

        img_feature = self.upsampling_layer1(img_feature_3, img_feature_2)
        img_feature = self.upsampling_layer2(img_feature, img_feature_1)
        img_feature = self.depth_layer(img_feature)


        for i in range(3):
            _, C, Hf, Wf = feat_list[i].shape
            sy = Hf / float(H)
            sx = Wf / float(W)
            featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4

            featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3

            scale = 2 * i if 2 * i != 0 else 1
            ref_T_mem = self.vox_utils[i].get_ref_T_mem(B, self.Y // scale, self.Z, self.X // scale)  # B,4,4
            ref_T_mem = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3

            ref_T_mem = ref_T_mem.to(featpix_T_ref_.device)

             # B*S,3,3

            feat_mems_ = self.vox_utils[i].unproject_image_to_mem(
                feat_list[i],  # B*S,128,H/8,W/8
                utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),  # featpix_T_ref  B*S,4,4
                cams_T_ref_, self.Y//scale, self.Z, self.X//scale,
                xyz_refA=None, z_sign=self.z_sign)

            feat_mems = __u(feat_mems_)
            feat_mem = self.cam_compressor[i](feat_mems.flatten(1, 2))
            C_bev = feat_mem.shape[1]
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, C_bev * self.Z, self.Y//scale, self.X//scale)
            feat_bev = self.bev_compressor[i](feat_bev_)
            world_features.append(feat_bev)

        m = tensor_list.mask
        assert m is not None

        for i, feat in enumerate(world_features):
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[str(i)] = NestedTensor(feat, mask)

        tensor_list.tensors = __u(tensor_list.tensors)
        return out,img_feature