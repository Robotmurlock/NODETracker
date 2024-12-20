from nodetracker.tracker.tracklet import Tracklet
from typing import List, Tuple


def remove_duplicates(threshold: float, tracklets_lhs: List[Tracklet], tracklets_rhs: List[Tracklet]) -> Tuple[List[Tracklet], List[Tracklet]]:
    lhs_exclude: List[int] = []
    rhs_exclude: List[int] = []

    for i_lhs, t_lhs in enumerate(tracklets_lhs):
        for i_rhs, t_rhs in enumerate(tracklets_rhs):
            iou_score = t_lhs.bbox.iou(t_rhs.bbox)
            if iou_score >= threshold:
                if t_lhs.age >= t_rhs.age:
                    rhs_exclude.append(i_rhs)
                else:
                    lhs_exclude.append(i_lhs)

    tracklets_lhs = [t_lhs for i_lhs, t_lhs in enumerate(tracklets_lhs) if i_lhs not in lhs_exclude]
    tracklets_rhs = [t_rhs for i_rhs, t_rhs in enumerate(tracklets_rhs) if i_rhs not in rhs_exclude]

    return tracklets_lhs, tracklets_rhs