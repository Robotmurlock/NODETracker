"""
Implementation of SOT (single-object-tracking) metrics.
"""
from typing import List, Union, Optional, Dict

import numpy as np

ThresholdType = Union[float, List[float], np.ndarray]


def traj_bbox_to_traj_center(traj: np.ndarray) -> np.ndarray:
    """
    Converts bbox trajectory to center trajectory (4D -> 2D)

    Args:
        traj: BBox Trajectory

    Returns:
        Center Trajectory
    """
    xc = (traj[..., 0] + traj[..., 2]) / 2
    yc = (traj[..., 1] + traj[..., 3]) / 2
    return np.dstack([xc, yc])


def traj_bbox_xyhw_to_xyxy(bbox: np.ndarray, eps=1e-9) -> np.ndarray:
    """
    Converts bbox format xyhw to xyxy.

    Args:
        bbox: BBox in xywh format.
        eps: minimum width and height

    Returns:
        Bbox in xyxy format
    """
    bbox = bbox.copy()
    bbox[..., 2] = np.maximum(bbox[..., 0] + bbox[..., 2], bbox[..., 0] + eps)
    bbox[..., 3] = np.maximum(bbox[..., 1] + bbox[..., 3], bbox[..., 1] + eps)
    return bbox


def point_to_point_distance(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Calculates point to point distance.

    Args:
        lhs: Left hand side point
        rhs: Right hand side point

    Returns:
        Distance (Array)
    """
    return np.sqrt((lhs[..., 0] - rhs[..., 0]) ** 2 + (lhs[..., 1] - rhs[..., 1]) ** 2)


def mse(gt_traj: np.ndarray, pred_traj: np.ndarray) -> float:
    """
    Mean Squared Error.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory

    Returns:
        MSE metric.
    """
    return float(((gt_traj - pred_traj) ** 2).mean())


def iou(bboxes1: np.ndarray, bboxes2: np.ndarray, is_xyhw_format: bool = True) -> np.ndarray:
    """
    Calculates iou between each bbox (for each time point and each batch).
    Used as helper function to calculate other IOU based metrics.

    Args:
        bboxes1: lhs bboxes
        bboxes2: rhs bboxes
        is_xyhw_format: If true then it is required to be converted to xyxy first

    Returns:
        iou score for each bbox
    """
    if bboxes1.shape != bboxes2.shape:
        raise AttributeError(f'BBoxes do not have the same shape: {bboxes1.shape} != {bboxes2.shape}')

    if bboxes1.size == 0:
        return np.array([], dtype=np.float32)

    if is_xyhw_format:
        bboxes1 = traj_bbox_xyhw_to_xyxy(bboxes1)
        bboxes2 = traj_bbox_xyhw_to_xyxy(bboxes2)

    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    # Calculate the coordinates of the intersection rectangles
    left = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    up = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    right = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    down = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # Calculate the area of intersection rectangles
    width = np.maximum(right - left, 0)
    height = np.maximum(down - up, 0)
    intersection_area = width * height

    # Calculate the IoU
    union_area = bboxes1_area + bboxes2_area - intersection_area
    iou_scores = np.divide(intersection_area, union_area, out=np.zeros_like(union_area), where=union_area != 0)

    return iou_scores


def accuracy(gt_traj: np.ndarray, pred_traj: np.ndarray) -> float:
    """
    Calculates average IOU between GT and PRED bbox.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory

    Returns:
        Accuracy metric.
    """
    return iou(gt_traj, pred_traj).mean()


def success(gt_traj: np.ndarray, pred_traj: np.ndarray, threshold: Optional[ThresholdType] = None) -> float:
    """
    Calculates average success between GT and PRED bbox.
    Success is 1 if iou is greater than the `threshold` else it is 0.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory
        threshold: Success threshold

    Returns:
        Accuracy metric.
    """
    if threshold is None:
        threshold = np.arange(0, 1.05, 0.05)
    if isinstance(threshold, float):
        threshold = [threshold]

    iou_scores = iou(gt_traj, pred_traj)

    scores = []
    for t in threshold:
        score = (iou_scores >= t).astype(np.float32).mean()
        scores.append(score)
    return float(np.array(scores).mean())


def precision(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    imheight: int,
    imwidth: int,
    threshold: Optional[ThresholdType] = None
) -> float:
    """
    Calculates average precision between GT and PRED bbox.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory
        imheight: Image width
        imwidth: Image height
        threshold: Success threshold

    Returns:
        Precision metric.
    """
    if threshold is None:
        threshold = np.arange(0.0, 51.0, 1.0)
    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = [threshold]

    # Calculate centers
    gt_center = traj_bbox_to_traj_center(gt_traj)
    pred_center = traj_bbox_to_traj_center(pred_traj)

    # Scale relative coordinates to image coordinates
    gt_center[..., 0] *= imheight
    gt_center[..., 1] *= imwidth
    pred_center[..., 0] *= imheight
    pred_center[..., 1] *= imwidth

    # Calculate precision
    distance = point_to_point_distance(gt_center, pred_center)

    # Calculate AUC
    scores = []
    for t in threshold:
        score = (distance <= t).astype(np.float32).mean()
        scores.append(score)
    return float(np.array(scores).mean())


def norm_precision(gt_traj: np.ndarray, pred_traj: np.ndarray, threshold: Optional[ThresholdType] = None) -> float:
    """
    Calculates average normalized precision between GT and PRED bbox.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory
        threshold: Success threshold

    Returns:
        Normalized precision metric.
    """
    if threshold is None:
        threshold = np.arange(0.0, 51.0, 1.0) / 100
    if isinstance(threshold, float):
        threshold = [threshold]

    # Calculate centers
    gt_center = traj_bbox_to_traj_center(gt_traj)
    pred_center = traj_bbox_to_traj_center(pred_traj)

    # Calculate normalized precision
    distance = point_to_point_distance(gt_center, pred_center)

    # Calculate AUC
    scores = []
    for t in threshold:
        score = (distance <= t).astype(np.float32).mean()
        scores.append(score)
    return float(np.array(scores).mean())


def metrics_func(gt_traj: np.ndarray, pred_traj: np.ndarray) -> Dict[str, float]:
    """
    Calculates Accuracy, Success and NormPrecision. Supports only default metric parameters.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory

    Returns:
        Mappings for each metric
    """
    return {
        'Accuracy': accuracy(gt_traj, pred_traj),
        'Success': success(gt_traj, pred_traj),
        'NormPrecision': norm_precision(gt_traj, pred_traj)
    }
