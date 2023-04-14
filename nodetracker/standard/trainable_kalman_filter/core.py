"""
Trainable Kalman Filter using SGD

1. Measurement vector: z = [x, y, h, w]
2. Measurement noise covariance matrix (learnable): R = [
    [xx, xy, xh, xw],
    [yx, yy, yh, yw]
    [hx, hy, hh, hw]
    [wx, wy, wh, ww]
]

State vector: x = [x, y, h, w, x', y', h', w']
State covariance matrix: P = [...] (like noise but 8x8)

Motion model (learnable): A (shape=8x8) - x, y, w, h, x', y', h', w'
Process noise (learnable): Q (shape=8x8) - x, y, w, h, x', y', h', w'

Initial state:
- State vector (learnable)
- State covariance (learnable)
"""
from typing import Tuple, List

import torch
import torch.linalg as LA
from torch import nn
import enum


class TrainingAKFMode(enum.Enum):
    FROZEN = enum.auto()  # Parameters are frozen
    MOTION = enum.auto()  # Uncertainty parameters are frozen
    UNCERTAINTY = enum.auto()  # Motion parameters are frozen
    ALL = enum.auto()  # All parameters are unfrozen

    @property
    def flags(self) -> Tuple[bool, bool]:
        """
        Returns:
            Train motion, train uncertainty flags
        """
        if self == TrainingAKFMode.FROZEN:
            return False, False
        if self == TrainingAKFMode.MOTION:
            return True, False
        if self == TrainingAKFMode.UNCERTAINTY:
            return False, True
        if self == TrainingAKFMode.ALL:
            return False, False

        raise AssertionError('Invalid Program State!')


class TrainableAdaptiveKalmanFilter(nn.Module):
    def __init__(self,
        sigma_p: float = 0.05,
        sigma_p_init_mult: float = 2.0,
        sigma_v: float = 0.00625,
        sigma_v_init_mult: float = 10.0,
        sigma_r: float = 1.0,
        sigma_q: float = 1.0,
        dt: float = 1.0,

        training_mode: TrainingAKFMode = TrainingAKFMode.FROZEN
    ):
        """
        Args:
            sigma_p: Position uncertainty parameter (multiplier)
            sigma_p_init_mult: Position uncertainty parameter for initial cov matrix (multiplier)
            sigma_v: Velocity uncertainty parameter (multiplier)
            sigma_v_init_mult: Velocity uncertainty parameter for initial cov matrix (multiplier)
            sigma_r: Measurement noise multiplier (matrix R)
            sigma_q: Process noise multiplier (matrix Q)
            dt: Step period

            training_mode: Model training mode
        """
        super().__init__()
        self._training_mode = training_mode
        train_motion_parameters, train_uncertainty_parameters = training_mode.flags

        self._sigma_p = nn.Parameter(torch.tensor(sigma_p, dtype=torch.float32),
                                     requires_grad=train_uncertainty_parameters)
        self._sigma_p_init_mult = nn.Parameter(torch.tensor(sigma_p_init_mult, dtype=torch.float32),
                                               requires_grad=train_uncertainty_parameters)
        self._sigma_v = nn.Parameter(torch.tensor(sigma_v, dtype=torch.float32),
                                     requires_grad=train_uncertainty_parameters)
        self._sigma_v_init_mult = nn.Parameter(torch.tensor(sigma_v_init_mult, dtype=torch.float32),
                                               requires_grad=train_uncertainty_parameters)
        self._sigma_r = nn.Parameter(torch.tensor(sigma_r, dtype=torch.float32),
                                     requires_grad=train_uncertainty_parameters)
        self._sigma_q = nn.Parameter(torch.tensor(sigma_q, dtype=torch.float32),
                                     requires_grad=train_uncertainty_parameters)
        self._dt = dt

        self._A = nn.Parameter(torch.tensor([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=torch.float32), requires_grad=train_motion_parameters)

        self._H = nn.Parameter(torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=torch.float32), requires_grad=False)

        self._I = nn.Parameter(torch.eye(8, dtype=torch.float32), requires_grad=False)

    @property
    def training_mode(self) -> TrainingAKFMode:
        """
        Returns:
            Model training model
        """
        return self._training_mode

    @training_mode.setter
    def training_mode(self, mode: TrainingAKFMode) -> None:
        """
        Sets model training mode.

        Args:
            mode: New training model
        """
        self._training_mode = mode

    # noinspection PyMethodMayBeStatic
    def _create_diag_cov_matrix(self, diag: List[float]) -> torch.Tensor:
        """
        Creates diagonal covariance matrix given list of stds.

        Args:
            diag: Diagonal elements

        Returns:
            Covariance matrix
        """
        diag = torch.tensor(diag, dtype=torch.float32)
        diag = torch.square(diag)
        return torch.diag(diag)

    def _estimate_process_noise_heuristic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates process noise matrix using bbox size heuristic. Noise is proportional to the bbox size.

        Args:
            x: State vector (last posterior)

        Returns:
            Process noise (Q) matrix
        """
        h, w = x[2], x[3]
        x_weight = h * self._sigma_p
        y_weight = w * self._sigma_p
        xv_weight = h * self._sigma_v
        yv_weight = w * self._sigma_v
        return self._create_diag_cov_matrix([x_weight, y_weight, x_weight, y_weight,
                                             xv_weight, yv_weight, xv_weight, yv_weight])

    def _estimate_measurement_noise_heuristic(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Estimates measurement noise matrix using bbox size heuristic. Noise is proportional to the bbox size.

        Args:
            x_hat: State vector (last prior)

        Returns:
            Measurement noise (Q) matrix
        """
        h, w = x_hat[2], x_hat[3]
        x_weight = h * self._sigma_p
        y_weight = w * self._sigma_p
        return self._create_diag_cov_matrix([x_weight, y_weight, x_weight, y_weight])

    def initiate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initiate state vector and covariance (uncertainty) matrix

        Args:
            z: Measurement

        Returns:
            Initialized vector and covariance matrix
        """
        x = torch.hstack([z, torch.zeros_like(z, dtype=torch.float32)])
        P = self._create_diag_cov_matrix(4 * [self._sigma_p * self._sigma_p_init_mult]
                                         + 4 * [self._sigma_v * self._sigma_v_init_mult])

        return x, P

    def predict(self, x: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs prediction (a priori, initial) step.

        Args:
            x: Current vector state (position)
            P: Current covariance matrix (uncertainty)

        Returns:
            - x_hat (Predicted vector state)
            - P_hat (Predicted covariance matrix)
        """
        # Predict state
        x_hat = self._A @ x  # 8x1

        # Predict uncertainty
        Q = self._estimate_process_noise_heuristic(x)
        P_hat = self._A @ P @ self._A.T + Q  # 8x8

        return x_hat, P_hat

    def multistep_predict(self, x: torch.Tensor, P: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs prediction (a priori, initial) step `n_steps` ahead.

        Args:
            x: Current vector state (position)
            P: Current covariance matrix (uncertainty)
            n_steps: Number of steps

        Returns:
            - x_hat (Predicted vector state)
            - P_hat (Predicted covariance matrix)
        """
        x_hat, P_hat = x, P

        for _ in range(n_steps):
            x_hat, P_hat = self.predict(x_hat, P_hat)

        return x_hat, P_hat

    def update(self, x_hat: torch.Tensor, P_hat: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs update (a posteriori) step.

        Args:
            x_hat: Predicted vector state (position)
            P_hat: Predicted covariance matrix (uncertainty)
            z: measurement

        Returns:
            Updated object bbox coordinates and uncertainty for those coordinates
        """

        # Update state
        R = self._estimate_measurement_noise_heuristic(x_hat)
        K = (P_hat @ self._H.T) @ LA.inv(self._H @ P_hat @ self._H.T + R)  # 8x4

        # validation: (8x8 @ 8x4) @ inv(4x8 @ 8x8 @ 8x4) = 8x4 @ 4x4 = 8x4
        z_hat = self._H @ x_hat  # 4x1
        innovation = z - z_hat  # 4x1
        x = x_hat + K @ innovation  # 8x1
        # validation: 8x1 + 8x4 @ 4x1 = 8x1 + 8x1

        # Update uncertainty
        P = (self._I - K @ self._H) @ P_hat  # 8x8
        # validation: (8x8 - 8x4 @ 4x8) @ 8x8 = 8x8 @ 8x8 = 8x8

        return x, P

    def project(self, x: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects state vector to measurement space.

        Args:
            x: State vector
            P: State uncertainty covariance matrix

        Returns:
            Projected state vector, projected uncertain covariance matrix
        """
        return self._H @ x, self._H @ P @ self._H.T


# noinspection DuplicatedCode
@torch.no_grad()
def run_kf_test():
    import cv2
    import os
    from nodetracker.common.project import OUTPUTS_PATH
    import numpy as np

    kf = TrainableAdaptiveKalmanFilter()

    state_A = torch.tensor([0.5, 0.5, 0.1, 0.1])
    state_B = torch.tensor([0.2, 0.1, 0.2, 0.3])
    state_C = torch.tensor([0.8, 0.9, 0.05, 0.05])
    time_len = 20
    dt = 1.0
    noise_std = 1e-3

    kf_preds = []
    kf_ests = []
    ground_truth = []
    ts = []

    # KF prediction: A -> B
    first_iteration = True
    x_hat, P_hat, x, P = None, None, None, None
    for i in range(time_len):
        t = dt*i
        gt = (1 - t/time_len) * state_A + t/time_len * state_B
        measurement = gt + torch.randn(*gt.shape) * noise_std
        if first_iteration:
            x, P = kf.initiate(measurement)
            first_iteration = False
            pred, _ = kf.project(x, P)
            pred = pred.clone()
            est = pred
        else:
            x_hat, P_hat = kf.predict(x, P)
            x, P = kf.update(x_hat, P_hat, measurement)
            pred, _ = kf.project(x_hat, P_hat)
            pred = pred.clone()
            est, _ = kf.project(x, P)
            est = pred.clone()

        ts.append(dt*i)
        ground_truth.append(gt)
        kf_preds.append(pred)
        kf_ests.append(est)

    # KF prediction: B -> C
    for i in range(time_len):
        t = dt*i
        gt = (1 - t/time_len) * state_B + t/time_len * state_C
        measurement = gt + torch.randn(*gt.shape) * noise_std

        x_hat, P_hat = kf.predict(x, P)
        x, P = kf.update(x_hat, P_hat, measurement)
        pred, _ = kf.project(x_hat, P_hat)
        pred = pred.clone()
        est, _ = kf.project(x, P)
        est = pred.clone()

        ts.append(dt*i)
        ground_truth.append(gt)
        kf_preds.append(pred)
        kf_ests.append(est)

    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    SIZE = 200
    mp4_path = os.path.join(OUTPUTS_PATH, 'tkf_simulation.mp4')

    print('Path:', mp4_path)
    # noinspection PyUnresolvedReferences
    mp4_writer = cv2.VideoWriter(mp4_path, fourcc, 5, (SIZE, SIZE))

    for kf_pred, gt in zip(kf_preds, ground_truth):
        kf_x, kf_y, kf_w, kf_h = [float(v) for v in kf_pred]
        kf_xmin = round(max(0, kf_x - kf_w) * SIZE)
        kf_ymin = round(max(0, kf_y - kf_h) * SIZE)
        kf_xmax = round(max(0, kf_x + kf_w) * SIZE)
        kf_ymax = round(max(0, kf_y + kf_h) * SIZE)

        gt_x, gt_y, gt_w, gt_h = [float(v) for v in gt]
        gt_xmin = round(max(0, gt_x - gt_w) * SIZE)
        gt_ymin = round(max(0, gt_y - gt_h) * SIZE)
        gt_xmax = round(max(0, gt_x + gt_w) * SIZE)
        gt_ymax = round(max(0, gt_y + gt_h) * SIZE)

        frame = np.ones(shape=(SIZE, SIZE, 3), dtype=np.uint8)
        # noinspection PyUnresolvedReferences
        frame = cv2.rectangle(frame, (gt_ymin, gt_xmin), (gt_ymax, gt_xmax), color=(0, 255, 0), thickness=2)
        # noinspection PyUnresolvedReferences
        frame = cv2.rectangle(frame, (kf_ymin, kf_xmin), (kf_ymax, kf_xmax), color=(255, 0, 0), thickness=1)
        mp4_writer.write(frame)

    mp4_writer.release()


if __name__ == '__main__':
    run_kf_test()
