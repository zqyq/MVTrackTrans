import pdb
from collections import OrderedDict
from collections import deque
from typing import List

import numpy as np

from tracking import matching


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):

    def __init__(self, xy, score, buffer_size=30):

        # wait activate
        self._xy = xy

        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        # self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)



    def activate(self, frame_id):
        """Start a new tracklet"""

        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):


        # self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self._xy = new_track.xy
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self._xy = new_track.xy
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def xy(self):
        # if self.state == TrackState.Lost:
        #     return self.mean[:2]

        return self._xy




    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker:
    def __init__(self, conf_thres=0.1, track_buffer=5):
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = conf_thres
        self.max_time_lost = track_buffer



    def update(self, dets, score,offset=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        score = score.squeeze(-1)
        remain_inds = score > self.det_thresh - 0.1

        dets = dets[remain_inds]
        score = score[remain_inds]



        if len(dets) > 0:
            """Detections"""
            detections = [STrack(xy, s, self.max_time_lost) for
                          (xy, s) in zip(dets, score)]
        else:
            detections = []

        #pdb.set_trace()

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2:association'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        if len(strack_pool) > 0 and offset is not None:
            #pdb.set_trace()
            strack_pool_xy = [track.xy - offset[i] for i, track in enumerate(strack_pool)]
        else:
            strack_pool_xy = [track.xy for track in strack_pool]
        detections_xy = [det.xy for det in detections]

        dists = matching.center_distance(strack_pool_xy, detections_xy)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=75)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        detections_xy = [det.xy for det in detections]
        unconfirmed_xy = [track.xy for track in unconfirmed]
        # dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.center_distance(unconfirmed_xy, detections_xy)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=100)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 3: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]

            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # print('===========Frame {}=========='.format(self.frame_id))
        # print('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # print('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # print('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    track_a = [t.xy for t in stracksa]
    track_b = [t.xy for t in stracksb]
    pdist = matching.center_distance(track_a, track_b)
    pairs = np.where(pdist < 6)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
