import os
import json
import pdb
import math
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *
import numpy as np
from multiview_detector.utils import geom, basic
import torchvision.transforms.functional as F
from operator import itemgetter
import random

class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=1.5, train_ratio=0.9, force_download=True
                 , final_dim: tuple = (720, 1280),
                 resize_lim: list = (0.8, 1.2),
                 ):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        self.is_train = train
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        if train:
            frame_range = range(1, int(self.num_frame * 0.8))
        else:
            frame_range = range(int(self.num_frame * 0.8), self.num_frame+1)
        self.augmentation = False

        self.img_fpaths = self.base.get_image_fpaths(frame_range)

        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        self.img_height, self.img_width = self.img_shape
        self.calibration = {}
        self.setup()
        self.img_downsample = 4


        pass

    def setup(self):
        intrinsic = torch.tensor(np.stack(self.base.intrinsic_matrices, axis=0), dtype=torch.float32)  # S,3,3
        intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()  # S,4,4
        self.calibration['intrinsic'] = intrinsic
        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.base.extrinsic_matrices, axis=0), dtype=torch.float32)

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def get_bev_gt(self, mem_pts, pids):
        centers = []
        person_ids = []

        for pt, pid in zip(mem_pts[0], pids):
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or x >= 1200 or y < 0 or y >= 800:
                continue
            centers.append(torch.tensor([x, y]))
            person_ids.append(torch.tensor(pid, dtype=torch.long))

        if len(centers) == 0:
            return torch.empty((0, 2)), torch.empty((0,), dtype=torch.long)
        centers = torch.stack(centers, dim=0).float()
        person_ids = torch.stack(person_ids, dim=0)

        center = torch.zeros((1, 200, 300), dtype=torch.float32)
        offset = torch.zeros((2, 200, 300), dtype=torch.float32)

        for pts, pid in zip(mem_pts[0], pids):
            x_idx, y_idx = int(pts[0]/4), int(pts[1]/4)
            #print(x_idx,y_idx)
            ct = pts[:2]
            ct_int = ct.int()
            if x_idx < 0 or x_idx >= 300 or y_idx < 0 or y_idx >= 200:
                continue


            basic.draw_umich_gaussian(center[0], [x_idx, y_idx], 1.5)
            offset[:, y_idx, x_idx] = ct - ct_int

        return centers.unsqueeze(0), person_ids,center

    def download(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(int(pedestrian['personID']))
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))



    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.augmentation:
            min_scale, max_scale = self.data_aug_conf['resize_lim']


            resize = float(torch.rand(1) * (max_scale - min_scale) + min_scale)
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        center = torch.zeros((1, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        # 处理img_pts为空的情况
        if len(img_pts) == 0:
            return center, person_ids, valid_mask

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)
        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)
        size_pts = torch.tensor(size_pts, dtype=torch.float32)
        foot_pts = np.stack(((xmin + xmax) / 2, (ymax+ymin)/2), axis=1)
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)
        head_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)
        head_pts = torch.tensor(head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate((foot_pts[pt_idx], )):  # , center_pts[pt_idx], head_pts[pt_idx])):
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                basic.draw_umich_gaussian(center[idx], pt.int(), 1.5)

            ct_int = foot_pts[pt_idx].int()
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, person_ids,valid_mask

    def get_img_boxes(self, img_pts, img_pids, sx, sy, crop):

        fH, fW = self.data_aug_conf['final_dim']

        if len(img_pts) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        area0 = (img_pts[:, 2] - img_pts[:, 0]) * (img_pts[:, 3] - img_pts[:, 1])

        xmin = img_pts[:, 0] * sx - crop[0]
        ymin = img_pts[:, 1] * sy - crop[1]
        xmax = img_pts[:, 2] * sx - crop[0]
        ymax = img_pts[:, 3] * sy - crop[1]

        xmin = np.clip(xmin, 0, sx*self.img_width)
        ymin = np.clip(ymin, 0, sy*self.img_height)
        xmax = np.clip(xmax, 0, sx*self.img_width)
        ymax = np.clip(ymax, 0, sy*self.img_height)


        w = xmax - xmin
        h = ymax - ymin
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        keep = (w > 4) & (h > 4) & (area >= 0.1 * area0) & (ar < 10)

        if np.sum(keep) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        xmin, ymin, xmax, ymax = xmin[keep], ymin[keep], xmax[keep], ymax[keep]
        img_pids = np.array(img_pids)[keep]


        cx = (xmin + xmax) / 2 / fW
        cy = (ymin + ymax) / 2 / fH
        w = (xmax - xmin) / fW
        h = (ymax - ymin) / fH

        points = torch.stack([torch.tensor(cx, dtype=torch.float32),
                              torch.tensor(cy, dtype=torch.float32),
                              torch.tensor(w, dtype=torch.float32),
                              torch.tensor(h, dtype=torch.float32)], dim=1)
        # pdb.set_trace()
        ids = torch.tensor(img_pids, dtype=torch.long)

        return points, ids

    def get_image_data(self, index, cameras, resize_dims=None, crop=None):
        imgs, intrins, extrins = [], [], []
        if not resize_dims and not crop:
            resize_dims,crop = [],[]

        img_centers, img_pids = [], []
        valids = []
        frame = list(self.world_gt.keys())[index]

        for cam in cameras:
            with Image.open(self.img_fpaths[cam][frame]) as img:
                img = img.convert('RGB')
                W, H = img.size
                #pdb.set_trace()
                if not resize_dims and not crop :
                    s_resize_dims, s_crop = self.sample_augmentation()
                    resize_dims.append(s_resize_dims)
                    crop.append(s_crop)
                if len(resize_dims) <=cam:
                    s_resize_dims, s_crop = self.sample_augmentation()
                    resize_dims.append(s_resize_dims)
                    crop.append(s_crop)

                sx = resize_dims[cam][0] / float(W)
                sy = resize_dims[cam][1] / float(H)

                extrin = self.calibration['extrinsic'][cam]
                intrin = self.calibration['intrinsic'][cam]
                intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

                fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

                new_x0 = x0 - crop[cam][0]
                new_y0 = y0 - crop[cam][1]

                pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
                intrin = pix_T_cam.squeeze(0)  # 4,4

                img = basic.img_transform(img, resize_dims[cam], crop[cam])

            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids1 = self.imgs_gt[frame][cam]
            img_center, img_pid,valid = self.get_img_gt(img_pts, img_pids1,resize_dims[cam][0] / float(W),resize_dims[cam][1] / float(H), crop[cam])
            img_centers.append(img_center)
            img_pids.append(img_pid)
            valids.append(valid)


        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), resize_dims, crop, torch.stack(img_centers), torch.stack(valids)

    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]

        cameras = list(range(self.num_cam))

        target = {
            'image_id': torch.tensor(frame),
            'orig_size': torch.tensor([800,1200]),
            'size': torch.tensor([800,1200]),
            'boxes': None,
            'track_ids': None,
            'labels': torch.zeros(0),
            'area': torch.zeros(0),
            'iscrowd': torch.zeros(0),
            'labels_ignore': torch.zeros(0),
            'area_ignore': torch.zeros(0),
            'iscrowd_ignore': torch.zeros(0),
            'boxes_ignore': torch.zeros(0),
            'track_ids_ignore': torch.zeros(0),
            'prev_image': None,
            'prev_target': None
        }


        prev_target = {
            'image_id': None,
            'orig_size': torch.tensor([800,1200]),
            'size': torch.tensor([800,1200]),
            'boxes': None,
            'track_ids': None,
            'labels': torch.zeros(0),
            'area': torch.zeros(0),
            'iscrowd': torch.zeros(0),
            'labels_ignore': torch.zeros(0),
            'area_ignore': torch.zeros(0),
            'iscrowd_ignore': torch.zeros(0),
            'boxes_ignore': torch.zeros(0),
            'track_ids_ignore': torch.zeros(0),
        }

        imgs, intrins, extrins, resize_dims,crop,img_centers,valids = \
            self.get_image_data(index, cameras)

        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldgrid2worldcoord_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        worldgrid_pts, world_pids = self.world_gt[frame]
        worldgrid_pts = torch.tensor(worldgrid_pts, dtype=torch.float32)
        worldgrid_pts = torch.cat((worldgrid_pts, torch.zeros_like(worldgrid_pts[:, 0:1])), dim=1).unsqueeze(0)
        augment = None
        if self.augmentation:

            Rz = torch.eye(3)
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = 0.25
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        prev_frame = None
        #if self.is_train and frame < 1800:
        if frame != int(self.num_frame * 0.8):
            frames = list(self.world_gt.keys())
            candidate_prev_frames = [f for f in [frame - 1] if f >= 1]

            if candidate_prev_frames:
                prev_frame = random.choice(candidate_prev_frames)
            else:
                prev_frame = frame

            prev_index = frames.index(prev_frame)
            prev_target['image_id'] = torch.tensor(prev_frame)

            prev_worldgrid_pts, prev_world_pids = self.world_gt[prev_frame]
            prev_worldgrid_pts = torch.tensor(prev_worldgrid_pts, dtype=torch.float32)
            prev_worldgrid_pts = torch.cat((prev_worldgrid_pts, torch.zeros_like(prev_worldgrid_pts[:, 0:1])),
                                           dim=1).unsqueeze(0)

            if self.augmentation:
                prev_worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), prev_worldgrid_pts)


            prev_image, prev_intrins, prev_extrins, _, _, prev_img_centers, prev_valids = \
                self.get_image_data(prev_index, cameras, resize_dims, crop)

        else:
            prev_target['image_id'] = torch.tensor(frame)
            prev_worldgrid_pts, prev_world_pids = self.world_gt[frame]
            prev_worldgrid_pts = torch.tensor(prev_worldgrid_pts, dtype=torch.float32)
            prev_worldgrid_pts = torch.cat((prev_worldgrid_pts, torch.zeros_like(prev_worldgrid_pts[:, 0:1])),
                                           dim=1).unsqueeze(0)

            if self.augmentation:
                prev_worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), prev_worldgrid_pts)

            prev_frame = frame

            prev_image, prev_intrins, prev_extrins, _, _, prev_img_centers, prev_valids = \
                self.get_image_data(index, cameras, resize_dims, crop)


        """worldgrid_pts = worldgrid_pts[..., :2]
        worldgrid_pts = worldgrid_pts[..., [1, 0]]


        prev_worldgrid_pts = prev_worldgrid_pts[..., :2]
        prev_worldgrid_pts = prev_worldgrid_pts[..., [1, 0]]"""

        worldgrid_pts,world_pids,bev_center = self.get_bev_gt(worldgrid_pts, world_pids)
        prev_worldgrid_pts, prev_worldgrid_pids,prev_bev_center = self.get_bev_gt(prev_worldgrid_pts, prev_world_pids)

        worldgrid_pts[..., 0] /= 1200
        worldgrid_pts[..., 1] /= 800

        prev_worldgrid_pts[..., 0] /= 1200
        prev_worldgrid_pts[..., 1] /= 800

        prev_pts = dict(zip(prev_worldgrid_pids.int().tolist(), prev_worldgrid_pts[0]))

        track_pts = []
        track_offsets = []

        for pts, pid in zip(worldgrid_pts[0], world_pids):
            ct = pts[:2]

            if pid in prev_worldgrid_pids:
                prev_ct = prev_pts[pid.int().item()][:2]  
                t_off = prev_ct - ct
                #print(t_off)
                if t_off.abs().max() > 15:
                    continue
                track_pts.append(prev_ct)  
                track_offsets.append(t_off)

        if len(track_pts) > 0:
            track_pts = torch.stack(track_pts, dim=0)  # [N, 2]
            track_offsets = torch.stack(track_offsets, dim=0)  # [N, 2]
        else:
            track_pts = torch.empty(1, 2, device=worldgrid_pts.device)
            track_offsets = torch.empty(1, 2, device=worldgrid_pts.device)


        target['boxes'],target['track_ids'] = torch.tensor(worldgrid_pts).squeeze(),torch.tensor(world_pids)
        prev_target['boxes'], prev_target['track_ids'] = torch.tensor(prev_worldgrid_pts).squeeze(), torch.tensor(
            prev_world_pids)
        target['valids'] = valids

        target['track_boxes'] = track_pts
        target['track_offsets'] = track_offsets

        shape = target['boxes'].shape
        prev_shape = prev_target['boxes'].shape
        target['labels'] = torch.zeros(shape[0], dtype=torch.long)
        prev_target['labels'] = torch.zeros(prev_shape[0], dtype=torch.long)
        target[f'prev_image'] = prev_image
        target['area'] = torch.zeros(shape[0])
        prev_target['area'] = torch.zeros(prev_shape[0])
        target['iscrowd'] = torch.zeros(shape[0])
        prev_target['iscrowd'] = torch.zeros(prev_shape[0])

        target['area_ignore'] = torch.zeros(0)
        target['labels_ignore'] = torch.zeros(0)
        target['track_ids_ignore'] = torch.zeros(0)
        target['boxes_ignore'] = torch.zeros(0)
        target['iscrowd_ignore'] = torch.zeros(0)
        target['bev_center'] = bev_center
        prev_target['area_ignore'] = torch.zeros(0)
        prev_target['labels_ignore'] = torch.zeros(0)
        prev_target['track_ids_ignore'] = torch.zeros(0)
        prev_target['boxes_ignore'] = torch.zeros(0)
        prev_target['iscrowd_ignore'] = torch.zeros(0)
        prev_target['bev_center'] = prev_bev_center

        target[f'prev_target']=prev_target
        imgs = imgs.unsqueeze(0)
        target['img_centers'] = img_centers

        for cam in range(self.num_cam):

            target[f'view{cam}'] = {
                'image_id': torch.tensor(frame),
                'size': torch.tensor([2988,5312]),
                'orig_size': torch.tensor([2988,5312]),
                'boxes': img_centers[cam],

                'boxes_ignore': torch.zeros(0),
                'track_ids_ignore': torch.zeros(0),
                'labels_ignore': torch.zeros(0),
                'area_ignore': torch.zeros(0),
                'iscrowd_ignore': torch.zeros(0),


                'prev_target': {
                    'image_id': torch.tensor(prev_frame),
                    'boxes': prev_img_centers[cam],

                    'boxes_ignore': torch.zeros(0),
                    'track_ids_ignore': torch.zeros(0),
                    'labels_ignore': torch.zeros(0),
                    'area_ignore': torch.zeros(0),
                    'iscrowd_ignore': torch.zeros(0),
                    'size': torch.tensor([2988,5312]),
                    'orig_size': torch.tensor([2988,5312]),

                }
            }

        return imgs , (target,),(intrins, extrins,worldgrid_T_worldcoord)

    def __len__(self):
        return len(self.world_gt.keys())


def test():
    from multiview_detector.datasets.Wildtrack import Wildtrack
    # from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
    # test projection
    world_grid_maps = []
    xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    import matplotlib.pyplot as plt
    for cam in range(dataset.num_cam):
        world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
                                                      dataset.base.extrinsic_matrices[cam])
        world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
        world_grid_map = np.zeros(dataset.worldgrid_shape)
        for i in range(H):
            for j in range(W):
                x, y = world_grids[i, j]
                if dataset.base.indexing == 'xy':
                    if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                        world_grid_map[int(y), int(x)] += 1
                else:
                    if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                        world_grid_map[int(x), int(y)] += 1
        world_grid_map = world_grid_map != 0
        plt.imshow(world_grid_map)
        plt.show()
        world_grid_maps.append(world_grid_map)
        pass
    plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
    plt.show()
    pass
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)
    pass


if __name__ == '__main__':
    test()
