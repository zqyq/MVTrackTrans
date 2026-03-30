# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math
import pdb

import torch
import torch.nn.functional as F
from torch import nn
from multiview_detector.transformer_models.dla import IDAUpV3_bis
from multiview_detector.utils import decode
from multiview_detector.utils import basic

from ..util import box_ops
from ..util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from .detr import DETR, PostProcess, SetCriterion


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 multi_frame_attention=False, multi_frame_encoding=False, merge_frame_features=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.transformer = transformer
        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding
        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        self.backbone = backbone
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = 3

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.ida_up = IDAUpV3_bis(
            64, [self.transformer.d_model,self.transformer.d_model,self.transformer.d_model])

        '''
        (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        '''


        self.hm = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.tracking = nn.Sequential(
            nn.Linear(self.transformer.d_model,self.transformer.d_model),
            SiLU(),
            nn.Linear(self.transformer.d_model, 2)
        )
        #self.tracking = MLP(self.transformer.d_model, self.transformer.d_model, 2, 3)

        # init weights #
        # prior bias
        self.hm[-1].bias.data.fill_(-2.19)

        fill_fc_weights(self.tracking)

        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.img_center_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model
    # def fpn_channels(self):
    #     """ Returns FPN channels. """
    #     num_backbone_outs = len(self.backbone.strides)
    #     return [self.hidden_dim, ] * num_backbone_outs

    def forward(self, samples: NestedTensor,prev_samples: NestedTensor = None,pre_cts = None,pre_memories=None,img_features=None,valid_mask=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if img_features is not None:
            img_hm = self.img_center_head(img_features)
        features, pos = self.backbone(samples)

        if valid_mask is None:
            valid_mask = []
            b, c, H, W = img_hm.shape
            for i in range(b):
                valid = torch.zeros((1, H, W), dtype=torch.bool)
                xy_e, scores = decode.decoder(
                    img_hm[i].sigmoid().unsqueeze(0), K=200
                )
                v = scores > 0.4
                xy_e = xy_e[v.expand(-1, -1, 2)]

                for x, y in enumerate(xy_e):
                    if int(x) < W and int(y) < H:
                        valid[:, int(y), int(x)] = True
                valid_mask.append(valid)
            valid_mask = torch.stack(valid_mask)
            #pdb.set_trace()


        valid_mask=valid_mask.squeeze(0).flatten(1)
        #pdb.set_trace()
        feats = torch.cat([
            img_features.flatten(2).transpose(1, 2)[valid_mask]
        ])
        features = features[-3:]
        src_list = []
        mask_list = []
        pos_list = []
        frame_features = [features]
        for frame, frame_feat in enumerate(frame_features):
            pos_list.extend(pos[-3:])
            for l, feat in enumerate(frame_feat):
                src, mask = feat.decompose()
                src_list.append(self.input_proj[l](src))
                mask_list.append(mask)
                assert mask is not None

        if pre_memories is not None:
            pre_cts = pre_cts.to('cuda')
            merged_hs, memory, *_ = self.transformer(src_list, mask_list, pos_list, pre_cts=pre_cts,pre_memories=pre_memories,feats= feats)
        elif prev_samples is not None:
            with torch.no_grad():
                prev_features, prev_pos = self.backbone(prev_samples)
                prev_features = prev_features[-3:]
                prev_src_list = []
                prev_mask_list = []
                prev_pos_list = []
                prev_frame_features = [prev_features]
                for frame, frame_feat in enumerate(prev_frame_features):
                    prev_pos_list.extend(prev_pos[-3:])
                    for l, feat in enumerate(frame_feat):
                        src, mask = feat.decompose()
                        prev_src_list.append(self.input_proj[l](src))
                        prev_mask_list.append(mask)
                        assert mask is not None

            merged_hs, memory, *_ = self.transformer(src_list, mask_list, pos_list, prev_src_list,prev_mask_list,prev_pos_list, pre_cts,feats= feats)
        else:
            merged_hs, memory, *_ = self.transformer(src_list, mask_list, pos_list)


        hs = []
        pre_hs = []
        for hs_m, pre_hs_m in merged_hs:
            hs.append(hs_m)
            pre_hs.append(pre_hs_m)


        outputs_hms = []
        outputs_tracking = []

        for layer_lvl in range(len(hs)):
            # print([hss.shape for hss in hs[layer_lvl]])
            # print(pre_hs[layer_lvl].shape)
            for i in range(len(hs[layer_lvl])):
                hs[layer_lvl][i] = hs[layer_lvl][i].to(torch.float32)

            with torch.cuda.amp.autocast(enabled=False):
               hs[layer_lvl] = self.ida_up(hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]
            # print("wh head: ", wh_head.shape)
            hm_head = self.hm(hs[layer_lvl])
            # gather features #
            # (x,y) to index
            # pre_reference_points = pre_hm.clone()
            # normalize
            # pre_reference_points[:, :, 0] /= self.output_shape[1]
            # pre_reference_points[:, :, 1] /= self.output_shape[0]
            # clamp #
            # pre_reference_points = torch.clamp(pre_reference_points, min=0.0, max=1.0)
            # assert pre_gathered_features.shape[:-1] == pre_hs[layer_lvl].shape[:-1] == pre_reference_points.shape[:-1]
            if pre_memories is not None or prev_samples is not None:

               tracking_head = self.tracking(pre_hs[layer_lvl])

               outputs_tracking.append(tracking_head)
            outputs_hms.append(hm_head)

        if outputs_tracking is not None and len(outputs_tracking) > 0:
            tracking = torch.stack(outputs_tracking)
        else:
            tracking = None

        out = {'hm': torch.stack(outputs_hms)
               ,'tracking': tracking,
               'img_hm':img_hm }

        return out,memory



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformablePostProcess(PostProcess):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        #assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        ###
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        ###

        scores, labels = prob.max(-1)
        # scores, labels = prob[..., 0:1].max(-1)
        boxes  = out_bbox#.unbind(-1)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h], dim=1)
        #pdb.set_trace()
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
