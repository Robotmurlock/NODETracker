"""
Implementation of SOT (single-object-tracking) metrics.
"""
import numpy as np
from nodetracker.library.cv.bbox import BBox
from typing import List, Tuple


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


def _iou(gt_traj: np.ndarray, pred_traj: np.ndarray) -> Tuple[List[float], int, int]:
    """
    Calculates iou between each bbox (for each time point and each batch).
    Used as helper function to calculate other IOU based metrics.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory

    Returns:
        List of ious, trajectory length, trajectory batch size
    """
    assert gt_traj.shape == pred_traj.shape, f'Shape does not match. {gt_traj.shape} != {pred_traj.shape}.'

    if len(gt_traj.shape) == 2:
        gt_traj = gt_traj[:, np.newaxis, :]
        pred_traj = pred_traj[:, np.newaxis, :]

    assert len(gt_traj.shape) == 3, f'Expected 3 dimensions for trajectory input. Got {gt_traj.shape}.'
    time_len, batch_size, dim = gt_traj.shape
    assert dim == 4, f'Expected trajectory to be 4D (x, y, w, h). But got {gt_traj.shape}!'

    iou_scores: List[float] = []
    for time_index in range(time_len):
        for batch_index in range(batch_size):
            gt_bbox = BBox.from_yxwh(*gt_traj[time_index, batch_index, :])
            pred_bbox = BBox.from_yxwh(*pred_traj[time_index, batch_index, :])
            iou_score = gt_bbox.iou(pred_bbox)
            iou_scores.append(iou_score)

    return iou_scores, time_len, batch_size


def accuracy(gt_traj: np.ndarray, pred_traj: np.ndarray) -> float:
    """
    Calculates average IOU between GT and PRED bbox.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory

    Returns:
        Accuracy metric.
    """
    iou_scores, time_len, batch_size = _iou(gt_traj, pred_traj)
    return sum(iou_scores) / (time_len * batch_size)


def success(gt_traj: np.ndarray, pred_traj: np.ndarray, threshold: float = 0.5) -> float:
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
    iou_scores, time_len, batch_size = _iou(gt_traj, pred_traj)
    return sum([(1.0 if score >= threshold else 0.0) for score in iou_scores]) / (time_len * batch_size)


def precision(gt_traj: np.ndarray, pred_traj: np.ndarray, imheight: int, imwidth: int, threshold: int = 20) -> float:
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
    return (distance <= threshold).astype(np.float32).mean()


def norm_precision(gt_traj: np.ndarray, pred_traj: np.ndarray, threshold: int = 0.05) -> float:
    """
    Calculates average normalized precision between GT and PRED bbox.

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory
        threshold: Success threshold

    Returns:
        Normalized precision metric.
    """
    # Calculate centers
    gt_center = traj_bbox_to_traj_center(gt_traj)
    pred_center = traj_bbox_to_traj_center(pred_traj)

    # Calculate normalized precision
    distance = point_to_point_distance(gt_center, pred_center)
    return (distance <= threshold).astype(np.float32).mean()
