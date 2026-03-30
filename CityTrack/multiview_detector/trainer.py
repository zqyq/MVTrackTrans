import pdb

import torch

import numpy as np

import cv2

from multiview_detector.util import misc as utils

from tracking.multitracker import JDETracker

from evaluation.mot_bev import mot_metrics_pedestrian
from multiview_detector.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from multiview_detector.loss.loss import FocalLoss
from multiview_detector.utils import basic
import torch.nn as nn
from multiview_detector.utils import decode
import matplotlib.pyplot as plt


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model,backbone, criterion, logdir, denormalize, args,cls_thres=0.4):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.backbone = backbone
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.args = args
        self.center_loss_fn = FocalLoss()
        self.offset_criterion = nn.L1Loss()


    def train(self, epoch, data_loader, optimizer,args,accelerator):
        self.model.train()
        self.criterion.train()
        self.backbone.train()

        total_loss_value = 0.0

        num_batches = 0

        for i, (samples, targets,cal) in enumerate(data_loader):
            samples = samples.to(self.args.device)
            prev_samples = targets[0]['prev_image'].unsqueeze(0)
            prev_samples = prev_samples.to(self.args.device)


            targets = [utils.nested_dict_to_device(t, self.args.device) for t in targets]
            # in order to be able to modify targets inside the forward call we need
            # to pass it through as torch.nn.parallel.DistributedDataParallel only
            # passes copies
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            if not isinstance(prev_samples, NestedTensor):
                prev_samples = nested_tensor_from_tensor_list(prev_samples)

            vis_bev = targets[0]['bev_center']

            heatmap_img = vis_bev.detach().cpu().numpy()[0, 0]  # (H, W)
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap_img, cmap='hot')
            plt.axis('off')
            plt.savefig("gtheatmap.png", bbox_inches='tight', pad_inches=0)
            plt.close()


            with torch.no_grad():
                prev_features,_= self.backbone(prev_samples, cal)
            features,img_features= self.backbone(samples, cal)


            outputs,_ = self.model(features, prev_features,targets[0]['track_boxes'],img_features=img_features,valid_mask=targets[0]['valids'])
            B, S = targets[0]['img_centers'].shape[:2]
            #print(B,S)
            center_img_loss = self.center_loss_fn(basic._sigmoid(outputs['img_hm']), targets[0]['img_centers']) / S

            center_loss = self.center_loss_fn(basic._sigmoid(outputs['hm'].squeeze(2)), targets[0]['bev_center'])

            heatmap = outputs['hm'].squeeze(2)

            heatmap_img = heatmap.detach().sigmoid().cpu().numpy()  # (1, 1, H, W)
            heatmap_img = heatmap_img[0, 0]  # (H, W)

            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap_img, cmap='hot')
            plt.colorbar()
            plt.axis('off')
            plt.savefig("heatmap.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            pred = outputs['tracking'].squeeze(1)  # [1, 16, 2]
            target = targets[0]['track_offsets']  # [1, 16, 2]
            offset_loss = self.offset_criterion(pred, target)

            center_factor = 1 / torch.exp(self.model.module.center_weight)
            center_loss_weight = center_factor * center_loss
            center_uncertainty_loss = self.model.module.center_weight

            tracking_factor = 1 / torch.exp(self.model.module.tracking_weight)
            tracking_loss_weight = tracking_factor * offset_loss
            tracking_uncertainty_loss = self.model.module.tracking_weight

            loss_weight_dict = {
                'center_loss': 10 * center_loss_weight,
                'tracking_loss': tracking_loss_weight,
                'img_loss': center_img_loss,

            }

            stats_dict = {
                'center_uncertainty_loss': center_uncertainty_loss,
                'tracking_uncertainty_loss': tracking_uncertainty_loss,

            }

            total_loss = sum(loss_weight_dict.values()) + sum(stats_dict.values())

            optimizer.zero_grad()

            accelerator.backward(total_loss)


            optimizer.step()

            num_batches += 1
            batch_loss = total_loss.item()
            total_loss_value += batch_loss
            avg_total_loss = total_loss_value / num_batches

            print(f"[Epoch {epoch}] Avg Total Loss: {avg_total_loss:.4f}")

        return np.array(total_loss_value / num_batches)


    def test(self,data_loader,args,postprocessors):
        self.model.eval()
        self.criterion.eval()
        self.backbone.eval()
        mota_pred_list= []
        frame_id = 0
        test_tracker = JDETracker(conf_thres=0.6)
        pre_memories = None
        for i, (samples, targets,cal) in enumerate(data_loader):



            samples = samples.to(self.args.device)
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            with torch.no_grad():
               features,img_features, = self.backbone(samples, cal)

            offset = None
            tracked_stracks = test_tracker.tracked_stracks
            confirmed = []

            for track in tracked_stracks:
                if track.is_activated:
                    confirmed.append(track)
            for track in test_tracker.lost_stracks:
                confirmed.append(track)
            #pdb.set_trace()


            if len(confirmed) > 0 and pre_memories is not None:
                strack_pool_xy = [track.xy  for track in confirmed]
                strack_pool_xy = torch.stack(strack_pool_xy)
                strack_pool_xy[..., 0] /= 640
                strack_pool_xy[..., 1] /= 768
                strack_pool_xy = strack_pool_xy.unsqueeze(0)
                with torch.no_grad():
                    out, pre_memories = self.model(features, pre_cts=strack_pool_xy, pre_memories=pre_memories,
                                                   img_features=img_features)

            else:
                with torch.no_grad():
                    out, pre_memories = self.model(features, pre_memories=None, img_features=img_features)
            heatmap = out['hm'].squeeze(2)


            heatmap_img = heatmap.detach().sigmoid().cpu().numpy()  # (1, 1, H, W)
            heatmap_img = heatmap_img[0, 0]  # (H, W)

            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap_img, cmap='viridis')
            #plt.colorbar()
            plt.axis('off')
            plt.savefig(f"test/{frame_id*5+1800}_heatmap.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            xy_e, scores = decode.decoder(
                heatmap.sigmoid() ,K=200
            )
            xy_e*=2
            if out['tracking'] is not None:
                offset = out['tracking'].detach().cpu().numpy().squeeze(0).squeeze(0)
                # print(offset.shape)
                offset[..., 0] *= 640
                offset[..., 1] *= 768
                track_offsets = targets[0]['track_offsets']
                track_offsets[..., 0] *= 640
                track_offsets[..., 1] *= 768
                #print(offset)
                #print(track_offsets)
                #pdb.set_trace()



            output_stracks = test_tracker.update(xy_e.cpu(), scores.cpu(),offset)
            mota_pred_list.extend([[frame_id+1948, s.track_id, -1, -1, -1, -1, s.score.item()]
                                        + s.xy.tolist() + [-1]
                                        for s in output_stracks])
            frame_id += 1

        gt_path = 'mota_gt.txt'
        pred_path = 'output.txt'

        np.savetxt(pred_path, np.array(mota_pred_list), '%f', delimiter=',')
        summary = mot_metrics_pedestrian(pred_path, gt_path)
        for key, value in summary.iloc[0].to_dict().items():
            print(f'track/{key}', value)

        return summary.iloc[0]['mota']





