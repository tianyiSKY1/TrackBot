import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker_z import kalman_filter
import torch



def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    mult_features = []
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix

def mult_embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    mult_features = []
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    for track in tracks:
        if (len(track.features) > 5) and (len(track.features) % 5 == 0):
        # if (track.score < 0.9) and (len(track.features) > 5):
            his_feats = np.array(track.features)
            # his_feats = torch.from_numpy(his_feats).to(torch.device("cuda"))
            # cluster_ids_x, cluster_centers = kmeans(X=his_feats, num_clusters=3, distance='cosine', device=torch.device("cuda"))
            # # print(cluster_ids_x)
            # track.mult_feats = cluster_centers.cpu().detach().numpy()
            # torch.cuda.empty_cache()
            cluster = KMeans(n_clusters=3, n_init='auto').fit(his_feats)
            # print(cluster.labels_)
            track.mult_feats = cluster.cluster_centers_



    for track in tracks:
        if track.mult_feats is not None:
                # np.append(track.mult_feats, track.smooth_feat)
            mult_features.append(np.vstack((track.mult_feats, track.smooth_feat)))


        else:
                # cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
                # return cost_matrix
                # mult_features.append(np.zeros((3, 2048), dtype=np.float))
            mult_features.append(np.stack((track.smooth_feat, track.smooth_feat, track.smooth_feat, track.smooth_feat), axis=0))
    mult_features = np.asarray(mult_features)
    mult_features = mult_features.transpose(1, 0, 2)
    cost_matrix_0 = np.maximum(0.0, cdist(mult_features[0], det_features, metric))
    cost_matrix_1 = np.maximum(0.0, cdist(mult_features[1], det_features, metric))
    cost_matrix_2 = np.maximum(0.0, cdist(mult_features[2], det_features, metric))
    cost_matrix_s = np.maximum(0.0, cdist(mult_features[3], det_features, metric))
    cost_matrix_01 = np.minimum(cost_matrix_0, cost_matrix_1)
    cost_matrix_02 = np.minimum(cost_matrix_2, cost_matrix_s)
    cost_matrix = np.minimum(cost_matrix_01, cost_matrix_02)

    # cost_matrix_01 = np.minimum(cost_matrix_0, cost_matrix_1)
    # cost_matrix_his = np.minimum(cost_matrix_2, cost_matrix_01)
    #
    # cost_matrix = (cost_matrix_his + cost_matrix_s) / 2.0

    # cost_matrix = (cost_matrix_0 + cost_matrix_1 + cost_matrix_2 + cost_matrix_3) / 4
    return cost_matrix

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix

def mults_embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    mult_features = []
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    for track in tracks:
        if (len(track.features) > 5):
            his_feats = np.array(track.features)
            cluster = KMeans(n_clusters=3, n_init='auto').fit(his_feats)
            track.mult_feats = cluster.cluster_centers_
    for track in tracks:
        if track.mult_feats is not None:
            mult_features.append(np.vstack((track.mult_feats, track.smooth_feat)))


        else:

            mult_features.append(np.stack((track.smooth_feat, track.smooth_feat, track.smooth_feat, track.smooth_feat), axis=0))
    mult_features = np.asarray(mult_features)
    mult_features = mult_features.transpose(1, 0, 2)
    cost_matrix_0 = np.maximum(0.0, cdist(mult_features[0], det_features, metric))
    cost_matrix_1 = np.maximum(0.0, cdist(mult_features[1], det_features, metric))
    cost_matrix_2 = np.maximum(0.0, cdist(mult_features[2], det_features, metric))
    cost_matrix_s = np.maximum(0.0, cdist(mult_features[3], det_features, metric))
    cost_matrix_01 = np.minimum(cost_matrix_0, cost_matrix_1)
    cost_matrix_02 = np.minimum(cost_matrix_2, cost_matrix_s)
    cost_matrix = np.minimum(cost_matrix_01, cost_matrix_02)

    # cost_matrix_01 = np.minimum(cost_matrix_0, cost_matrix_1)
    # cost_matrix_his = np.minimum(cost_matrix_2, cost_matrix_01)
    # for i, track in enumerate(tracks):
    #     cost_matrix[i] = track.score*cost_matrix_s[i] + (1-track.score)*cost_matrix_his[i]

    # cost_matrix = (cost_matrix_0 + cost_matrix_1 + cost_matrix_2 + cost_matrix_3) / 4
    return cost_matrix



def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    # measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost