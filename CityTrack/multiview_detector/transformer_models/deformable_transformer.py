# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math
from typing import Optional

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from ..util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
from .transformer import _get_clones, _get_activation_fn
import pdb
from torch import Tensor, nn
import torch.nn.functional as F

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 multi_frame_attention_separate_encoder=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels
        self.multi_frame_attention_separate_encoder = multi_frame_attention_separate_encoder

        enc_num_feature_levels = num_feature_levels
        if multi_frame_attention_separate_encoder:
            enc_num_feature_levels = enc_num_feature_levels // 2

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          enc_num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=False)

        fusion_layer = TransformerFusionLayer1(d_model, nhead, dim_feedforward, dropout, activation)
        self.fusion = DeformableTransformerFusion1(fusion_layer, 4, return_intermediate=True)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.linear1 = nn.Linear(d_model, 512)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def my_forward_ffn(self, memory):
        memory2 = self.linear2(self.dropout2(self.activation(self.linear1(memory))))
        return self.norm2(memory + self.dropout3(memory2))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            #proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposal = grid.view(N_, -1, 2)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self,srcs, masks, pos_embeds, prev_srcs=None,prev_masks=None,prev_pos_embeds=None, pre_cts=None,pre_memories=None,feats = None):
        prev_memories_per_level = None
        #pdb.set_trace()
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)

            mask = mask.flatten(1)
            pos_embed = pos_embed.unsqueeze(0)

            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed = pos_embed + self.level_embed[lvl % self.num_feature_levels].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)


        memory = self.encoder(src_flatten, spatial_shapes,valid_ratios, lvl_pos_embed_flatten, mask_flatten)


        if pre_cts is not None:
           pre_sample = pre_cts.clone()

        if pre_memories is not None:
            prev_memories_per_level = []
            start = 0
            for (h, w) in spatial_shapes:
                end = start + h * w
                mem_l = pre_memories[:, start:end, :].transpose(1, 2).reshape(1, self.d_model, h, w)
                prev_memories_per_level.append(mem_l)
                start = end
        elif prev_srcs is not None:
            with torch.no_grad():
                prev_src_flatten = []
                prev_mask_flatten = []
                prev_lvl_pos_embed_flatten = []
                prev_spatial_shapes = []
                for lvl, (src, mask, pos_embed) in enumerate(zip(prev_srcs, prev_masks, prev_pos_embeds)):
                    bs, c, h, w = src.shape
                    spatial_shape = (h, w)
                    prev_spatial_shapes.append(spatial_shape)
                    src = src.flatten(2).transpose(1, 2)
                    mask = mask.flatten(1)
                    pos_embed = pos_embed.unsqueeze(0)

                    pos_embed = pos_embed.flatten(2).transpose(1, 2)
                    lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                    # lvl_pos_embed = pos_embed + self.level_embed[lvl % self.num_feature_levels].view(1, 1, -1)
                    prev_lvl_pos_embed_flatten.append(lvl_pos_embed)
                    prev_src_flatten.append(src)
                    prev_mask_flatten.append(mask)
                prev_src_flatten = torch.cat(prev_src_flatten, 1)
                prev_mask_flatten = torch.cat(prev_mask_flatten, 1)
                prev_lvl_pos_embed_flatten = torch.cat(prev_lvl_pos_embed_flatten, 1)
                prev_spatial_shapes = torch.as_tensor(prev_spatial_shapes, dtype=torch.long, device=src_flatten.device)
                prev_level_start_index = torch.cat(
                    (prev_spatial_shapes.new_zeros((1,)), prev_spatial_shapes.prod(1).cumsum(0)[:-1]))
                prev_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in prev_masks], 1)

                pre_memories = self.encoder(prev_src_flatten, prev_spatial_shapes, prev_valid_ratios,
                                            prev_lvl_pos_embed_flatten, prev_mask_flatten)

                # (x,y) to index
                prev_memories_per_level = []
                start = 0
                for (h, w) in prev_spatial_shapes:
                    end = start + h * w
                    mem_l = pre_memories[:, start:end, :].transpose(1, 2).reshape(1, self.d_model, h, w)
                    prev_memories_per_level.append(mem_l)
                    start = end




        memories_per_level = []
        start = 0
        for (h, w) in spatial_shapes:
            end = start + h * w

            mem_l = memory[:, start:end, :].transpose(1, 2).reshape(1, self.d_model, h, w)
            #pdb.set_trace()
            memories_per_level.append(mem_l)
            start = end

        sampled_feats = []

        prev_hs = None

        if pre_cts is not None:
            for lvl, mem in enumerate(prev_memories_per_level):
                # pre_sample: [bs, num_points, 2], normalized grid

                normalized_grid = pre_sample * 2 - 1
                grid = normalized_grid.unsqueeze(2)
                feat = F.grid_sample(mem, grid, align_corners=False)  # [bs, c, num_points, 1, 1]

                feat = feat.squeeze(-1).squeeze(-1).transpose(1, 2)  # [bs, num_points, c]
                sampled_feats.append(feat)

            feat_all = torch.stack(sampled_feats, dim=0).mean(dim=0)  # [bs, num_points, C]

            # transform to track queries
            prev_query_embed = self.my_forward_ffn(feat_all)  # [B, L1, C]
            prev_query_pos = torch.zeros_like(prev_query_embed)  # or your learned pos

            # current frame features
            feats = feats.unsqueeze(0)  # [B, L2, C]
            prev_view_query_pos = torch.zeros_like(feats)  # optional pos for memory

            # Cross-attention: tgt=prev_query_embed, memory=feats
            out = self.fusion(
                tgt=prev_query_embed.transpose(0, 1),  # [L1, B, C]
                memory=feats.transpose(0, 1),  # [L2, B, C]
                query_pos=prev_query_pos.transpose(0, 1),
                pos=prev_view_query_pos.transpose(0, 1),
            )

            out = out.transpose(0, 1)  # [B, L1, C]

            # updated track queries
            prev_query_embed = out

            prev_pos_embed = torch.zeros_like(prev_query_embed)
            # print(pre_reference_points.shape)

            # decoder
            # pre_query_embed is from sampled features of pre_memories,
            # pre_ref_pts are tracks pos at t-1, these are queries to question memory t
            # print([m.shape for m in masks_flatten])
            # pdb.set_trace()

            prev_hs, inter_references = self.decoder(prev_query_embed, pre_sample, memory,
                                                     spatial_shapes, valid_ratios, prev_pos_embed, mask_flatten)
            #pdb.set_trace()


        # for inference speed up #
        if prev_hs is not None:
           return [[memories_per_level, prev_hs]], memory
        else:
            return [[memories_per_level, memories_per_level]], memory


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output




class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_padding_mask=None, query_attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=query_attn_mask)[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, src_padding_mask, query_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_valid_ratios,
                query_pos=None, src_padding_mask=None, query_attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_padding_mask, query_attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

class TransformerFusionLayer1(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Cross-Attention: query from tgt, key/value from memory
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # Cross-Attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        tgt2 = self.cross_attn(q, k, value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


class DeformableTransformerFusion1(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)
            """
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        """
        return output



def build_deforamble_transformer(args):

    num_feature_levels = args.num_feature_levels
    if args.multi_frame_attention:
        num_feature_levels *= 2

    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        multi_frame_attention_separate_encoder=args.multi_frame_attention and args.multi_frame_attention_separate_encoder)
