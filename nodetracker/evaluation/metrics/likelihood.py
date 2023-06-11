import numpy as np
import torch


def gaussian_nll_loss(gt_traj: np.ndarray, pred_traj: np.ndarray, pred_var: np.ndarray) -> float:
    """
    GaussianNLLoss

    Args:
        gt_traj: Ground Truth Trajectory
        pred_traj: Prediction Trajectory
        pred_var: Prediction Variance

    Returns:
        Negative Gaussian log likelihood loss
    """
    gt_traj = torch.from_numpy(gt_traj)
    pred_traj = torch.from_numpy(pred_traj)
    pred_var = torch.from_numpy(pred_var)

    loss = torch.nn.functional.gaussian_nll_loss(pred_traj, gt_traj, pred_var)
    return loss.numpy()
