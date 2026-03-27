# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Tracker which achieves MOT with the provided object detector.
"""
import pdb
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import clip_boxes_to_image, nms, box_iou

from ..util.box_ops import box_xyxy_to_cxcywh
from filterpy.kalman import KalmanFilter


class Tracker:
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detector, obj_detector_post, tracker_cfg,
                 generate_attention_maps, logger=None, verbose=False):
        self.obj_detector = obj_detector
        self.obj_detector_post = obj_detector_post
        self.detection_obj_score_thresh = tracker_cfg['detection_obj_score_thresh']
        self.track_obj_score_thresh = tracker_cfg['track_obj_score_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.track_nms_thresh = tracker_cfg['track_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = float(tracker_cfg['inactive_patience'])
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_sim_only = tracker_cfg['reid_sim_only']
        self.generate_attention_maps = generate_attention_maps
        self.reid_score_thresh = tracker_cfg['reid_score_thresh']
        self.reid_greedy_matching = tracker_cfg['reid_greedy_matching']
        self.prev_frame_dist = tracker_cfg['prev_frame_dist']
        self.steps_termination = tracker_cfg['steps_termination']
        self.nms = 8
        if self.generate_attention_maps:
            assert hasattr(self.obj_detector.transformer.decoder.layers[-1], 'multihead_attn'), 'Generation of attention maps not possible for deformable DETR.'

            attention_data = {
                'maps': None,
                'conv_features': {},
                'hooks': []}

            hook = self.obj_detector.backbone[-2].register_forward_hook(
                lambda self, input, output: attention_data.update({'conv_features': output}))
            attention_data['hooks'].append(hook)

            def add_attention_map_to_data(self, input, output):
                height, width = attention_data['conv_features']['3'].tensors.shape[-2:]
                attention_maps = output[1].view(-1, height, width)

                attention_data.update({'maps': attention_maps})

            multihead_attn = self.obj_detector.transformer.decoder.layers[-1].multihead_attn
            hook = multihead_attn.register_forward_hook(
                add_attention_map_to_data)
            attention_data['hooks'].append(hook)

            self.attention_data = attention_data

        self._logger = logger
        if self._logger is None:
            self._logger = lambda *log_strs: None
        self._verbose = verbose

    @property
    def num_object_queries(self):
        return self.obj_detector.num_queries

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        self._prev_features = deque([None], maxlen=self.prev_frame_dist)

        if hard:
            self.track_num = 0
            self.results = {}
            self.frame_index = 0
            self.num_reids = 0

    @property
    def device(self):
        return next(self.obj_detector.parameters()).device

    def point_nms(slef,points, scores, dist_thresh):

        scores, order = scores.sort(0, descending=True)
        order = order.to(points.device)  # 这里把order转到points所在设备

        points = points[order]

        keep = []
        suppressed = torch.zeros(len(points), dtype=torch.bool, device=points.device)

        for i in range(len(points)):
            if suppressed[i]:
                continue
            keep.append(order[i].item())
            dists = torch.norm(points[i] - points[i + 1:], dim=1)
            suppressed[i + 1:] |= dists < dist_thresh

        return keep

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]

        for track in tracks:
            track.pos = track.last_pos[-1]
        self.inactive_tracks += tracks

    def add_tracks(self, pos, scores, hs_embeds, indices, masks=None, attention_maps=None, aux_results=None):
        """Initializes new Track objects and saves them."""
        new_track_ids = []
        for i in range(len(pos)):
            self.tracks.append(Track(
                pos[i],
                scores[i],
                self.track_num + i,
                hs_embeds[i],
                indices[i],
            ))
            new_track_ids.append(self.track_num + i)
        self.track_num += len(new_track_ids)

        if new_track_ids:
            self._logger(
                f'INIT TRACK IDS (detection_obj_score_thresh={self.detection_obj_score_thresh}): '
                f'{new_track_ids}')

            if aux_results is not None:
                aux_scores = torch.cat([
                    a['scores'][-self.num_object_queries:][indices]
                    for a in aux_results] + [scores[..., None], ], dim=-1)

                for new_track_id, aux_score in zip(new_track_ids, aux_scores):
                    self._logger(f"AUX SCORES ID {new_track_id}: {[f'{s:.2f}' for s in aux_score]}")

        return new_track_ids

    def reid(self, new_det_boxes, new_det_scores, new_det_hs_embeds,
             new_det_masks=None, new_det_attention_maps=None):
        """Tries to ReID inactive tracks with provided detections."""

        self.inactive_tracks = [
            t for t in self.inactive_tracks
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        if not self.inactive_tracks or not len(new_det_boxes):
            return torch.ones(new_det_boxes.size(0)).bool().to(self.device)

        # calculate distances
        dist_mat = []
        if self.reid_greedy_matching:
            new_det_boxes_cxcyhw = box_xyxy_to_cxcywh(new_det_boxes).cpu().numpy()
            inactive_boxes_cxcyhw = box_xyxy_to_cxcywh(torch.stack([
                track.pos for track in self.inactive_tracks])).cpu().numpy()

            dist_mat = inactive_boxes_cxcyhw[:, :2].reshape(-1, 1, 2) - \
                new_det_boxes_cxcyhw[:, :2].reshape(1, -1, 2)
            dist_mat = (dist_mat ** 2).sum(axis=2)

            track_size = inactive_boxes_cxcyhw[:, 2] * inactive_boxes_cxcyhw[:, 3]
            item_size = new_det_boxes_cxcyhw[:, 2] * new_det_boxes_cxcyhw[:, 3]

            invalid = ((dist_mat > track_size.reshape(len(track_size), 1)) + \
                       (dist_mat > item_size.reshape(1, len(item_size))))
            dist_mat = dist_mat + invalid * 1e18

            def greedy_assignment(dist):
                matched_indices = []
                if dist.shape[1] == 0:
                    return np.array(matched_indices, np.int32).reshape(-1, 2)
                for i in range(dist.shape[0]):
                    j = dist[i].argmin()
                    if dist[i][j] < 1e16:
                        dist[:, j] = 1e18
                        dist[i, j] = 0.0
                        matched_indices.append([i, j])
                return np.array(matched_indices, np.int32).reshape(-1, 2)

            matched_indices = greedy_assignment(dist_mat)
            row_indices, col_indices = matched_indices[:, 0], matched_indices[:, 1]

        else:
            for track in self.inactive_tracks:
                track_sim = track.hs_embed[-1]

                track_sim_dists = torch.cat([
                    F.pairwise_distance(track_sim, sim.unsqueeze(0))
                    for sim in new_det_hs_embeds])

                dist_mat.append(track_sim_dists)

            dist_mat = torch.stack(dist_mat)

            dist_mat = dist_mat.cpu().numpy()
            row_indices, col_indices = linear_sum_assignment(dist_mat)

        assigned_indices = []
        remove_inactive = []
        for row_ind, col_ind in zip(row_indices, col_indices):
            if dist_mat[row_ind, col_ind] <= self.reid_sim_threshold:
                track = self.inactive_tracks[row_ind]

                self._logger(
                    f'REID: track.id={track.id} - '
                    f'count_inactive={track.count_inactive} - '
                    f'to_inactive_frame={self.frame_index - track.count_inactive}')

                track.count_inactive = 0
                track.pos = new_det_boxes[col_ind]
                track.score = new_det_scores[col_ind]
                track.hs_embed.append(new_det_hs_embeds[col_ind])
                track.reset_last_pos()

                if new_det_masks is not None:
                    track.mask = new_det_masks[col_ind]
                if new_det_attention_maps is not None:
                    track.attention_map = new_det_attention_maps[col_ind]

                assigned_indices.append(col_ind)
                remove_inactive.append(track)

                self.tracks.append(track)

                self.num_reids += 1

        for track in remove_inactive:
            self.inactive_tracks.remove(track)

        reid_mask = torch.ones(new_det_boxes.size(0)).bool().to(self.device)

        for ind in assigned_indices:
            reid_mask[ind] = False

        return reid_mask



    def step(self, blob,cal):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        self.inactive_tracks = [
            t for t in self.inactive_tracks
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self._logger(f'FRAME: {self.frame_index + 1}')
        if self.inactive_tracks:
            self._logger(f'INACTIVE TRACK IDS: {[t.id for t in self.inactive_tracks]}')

        # add current position to last_pos list
        for track in self.tracks:
            track.last_pos.append(track.pos.clone())

        img = blob['img'].to(self.device)
        orig_size = blob['orig_size'].to(self.device)

        target = None
        num_prev_track = len(self.tracks + self.inactive_tracks)
        if num_prev_track:
            track_query_boxes = torch.stack([
                torch.from_numpy(t.predict()).to(self.device).float() if isinstance(t.predict(), np.ndarray)
                else t.predict().to(self.device).float()
                for t in self.tracks + self.inactive_tracks
            ], dim=0)

            track_query_boxes = track_query_boxes / torch.tensor([
                orig_size[0, 1], orig_size[0, 0]], dtype=torch.float32, device=self.device)

            target = {'track_query_boxes': track_query_boxes}

            target['image_id'] = torch.tensor([1]).to(self.device)
            target['track_query_hs_embeds'] = torch.stack([
                t.hs_embed[-1] for t in self.tracks + self.inactive_tracks], dim=0)

            target = {k: v.to(self.device) for k, v in target.items()}
            target = [target]

        outputs, _, features, _, _ = self.obj_detector(img, cal,target, self._prev_features[0])

        hs_embeds = outputs['hs_embed'][0]

        results = self.obj_detector_post['bbox'](outputs, orig_size)
        if "segm" in self.obj_detector_post:
            results = self.obj_detector_post['segm'](
                results,
                outputs,
                orig_size,
                blob["size"].to(self.device),
                return_probs=True)

        result = results[0]
        boxes = result['boxes']


        # TRACKS
        if num_prev_track:
            track_scores = result['scores'][:-self.num_object_queries]
            track_boxes = boxes[:-self.num_object_queries]

            track_keep = torch.logical_and(
                track_scores > self.track_obj_score_thresh,
                result['labels'][:-self.num_object_queries] == 0)

            tracks_to_inactive = []
            tracks_from_inactive = []

            for i, track in enumerate(self.tracks):
                if track_keep[i]:
                    track.update(track_boxes[i].cpu().numpy())
                    track.score = track_scores[i]
                    track.hs_embed.append(hs_embeds[i])
                    #track.pos = track_boxes[i]
                    track.count_termination = 0

                else:
                    track.count_termination += 1
                    if track.count_termination >= self.steps_termination:
                        tracks_to_inactive.append(track)

            track_keep = torch.logical_and(
                track_scores > self.reid_score_thresh,
                result['labels'][:-self.num_object_queries] == 0)

            # reid queries
            for i, track in enumerate(self.inactive_tracks, start=len(self.tracks)):
                if track_keep[i]:
                    track.update(track_boxes[i].cpu().numpy())
                    track.score = track_scores[i]
                    track.hs_embed.append(hs_embeds[i])
                    #track.pos = track_boxes[i]
                    tracks_from_inactive.append(track)

            if tracks_to_inactive:
                self._logger(
                    f'NEW INACTIVE TRACK IDS '
                    f'(track_obj_score_thresh={self.track_obj_score_thresh}): '
                    f'{[t.id for t in tracks_to_inactive]}')

            self.num_reids += len(tracks_from_inactive)
            for track in tracks_from_inactive:
                self.inactive_tracks.remove(track)
                self.tracks.append(track)

            self.tracks_to_inactive(tracks_to_inactive)
            # self.tracks = [
            #         track for track in self.tracks
            #         if track not in tracks_to_inactive]

            if self.track_nms_thresh and self.tracks:
                track_boxes = torch.stack([t.pos for t in self.tracks])
                track_scores = torch.stack([t.score for t in self.tracks])

                keep = self.point_nms(track_boxes, track_scores, self.nms)
                remove_tracks = [
                    track for i, track in enumerate(self.tracks)
                    if i not in keep]

                if remove_tracks:
                    self._logger(
                        f'REMOVE TRACK IDS (track_nms_thresh={self.track_nms_thresh}): '
                        f'{[track.id for track in remove_tracks]}')

                # self.tracks_to_inactive(remove_tracks)
                self.tracks = [
                    track for track in self.tracks
                    if track not in remove_tracks]

        # NEW DETS

        new_det_scores = result['scores'][-self.num_object_queries:]
        new_det_boxes = boxes[-self.num_object_queries:]
        new_det_hs_embeds = hs_embeds[-self.num_object_queries:]

        new_det_keep = torch.logical_and(
            new_det_scores > self.detection_obj_score_thresh,
            result['labels'][-self.num_object_queries:] == 0)

        new_det_boxes = new_det_boxes[new_det_keep]
        new_det_scores = new_det_scores[new_det_keep]
        new_det_hs_embeds = new_det_hs_embeds[new_det_keep]
        new_det_indices = new_det_keep.float().nonzero()
        """
        reid_mask = self.reid(
            new_det_boxes,
            new_det_scores,
            new_det_hs_embeds,
)

        new_det_boxes = new_det_boxes[reid_mask]
        new_det_scores = new_det_scores[reid_mask]
        new_det_hs_embeds = new_det_hs_embeds[reid_mask]
        new_det_indices = new_det_indices[reid_mask]
        """
        new_track_ids = self.add_tracks(
            new_det_boxes,
            new_det_scores,
            new_det_hs_embeds,
            new_det_indices,
            )

        # NMS

        if self.detection_nms_thresh and self.tracks:
            track_boxes = torch.stack([t.pos for t in self.tracks])
            track_scores = torch.stack([t.score for t in self.tracks])

            new_track_mask = torch.tensor([
                True if t.id in new_track_ids
                else False
                for t in self.tracks])
            track_scores[~new_track_mask] = np.inf

            keep = self.point_nms(track_boxes, track_scores, self.nms)
            remove_tracks = [track for i, track in enumerate(self.tracks) if i not in keep]

            if remove_tracks:
                self._logger(
                    f'REMOVE TRACK IDS (detection_nms_thresh={self.detection_nms_thresh}): '
                    f'{[track.id for track in remove_tracks]}')

            self.tracks = [track for track in self.tracks if track not in remove_tracks]

        ####################
        # Generate Results #
        ####################

        for track in self.tracks:
            if track.id not in self.results:
                self.results[track.id] = {}

            self.results[track.id][self.frame_index] = {}

            if self.obj_detector.overflow_boxes:
                self.results[track.id][self.frame_index]['bbox'] = track.pos.cpu().numpy()
            else:
                self.results[track.id][self.frame_index]['bbox'] = clip_boxes_to_image(track.pos, orig_size[0]).cpu().numpy()

            self.results[track.id][self.frame_index]['score'] = track.score.cpu().numpy()
            self.results[track.id][self.frame_index]['obj_ind'] = track.obj_ind.cpu().item()


        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.frame_index += 1
        self._prev_features.append(features)

        if self.reid_sim_only:
            self.tracks_to_inactive(self.tracks)

    def get_results(self):
        """Return current tracking results."""
        return self.results


class Track:
    def __init__(self, pos, score, track_id, hs_embed=None, obj_ind=None):
        self.id = track_id
        self.score = score
        self.hs_embed = [hs_embed] if hs_embed is not None else []
        self.obj_ind = obj_ind

        if isinstance(pos, torch.Tensor):
            pos_np = pos.detach().cpu().numpy().reshape((2, 1))
            self.pos = pos.detach().cpu()
        else:
            pos_np = np.array(pos).reshape((2, 1))
            self.pos = torch.from_numpy(pos_np).reshape(-1).float()

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)

        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=np.float32)

        self.kf.P *= 1000  # 初始位置/速度很不确定
        self.kf.Q *= 0.001
        self.kf.R *= 0.0005

        self.kf.x[:2] = pos_np
        self.kf.x[2:] = 0.

        self.last_pos = deque([pos_np.reshape(-1)])
        self.ims = deque([])
        self.count_inactive = 0
        self.count_termination = 0
        self.gt_id = None

    def predict(self):
        self.kf.predict()
        pred_pos = self.kf.x[:2].reshape(-1)
        self.last_pos.append(pred_pos)
        self.pos = torch.from_numpy(pred_pos.copy()).float()
        return self.pos

    def update(self, detection_pos):
        detection = np.array(detection_pos).reshape((2,))
        # 正常卡尔曼滤波 update，不手动覆盖
        self.kf.update(detection)

        updated_pos = self.kf.x[:2].reshape(-1)  # 更新后的状态位置
        self.last_pos.append(updated_pos)
        self.pos = torch.from_numpy(updated_pos.copy()).float()
        return self.pos

    def get_pos(self):
        # 返回 tensor 方便外部使用
        return self.pos

    def has_positive_area(self):
        pos = self.get_pos()
        return pos[0].item() > 0 and pos[1].item() > 0

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.kf.x[:2].reshape(-1))

