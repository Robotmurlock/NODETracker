"""
Trainable Kalman Filter using SGD that supports batch operations

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
import enum
from typing import Tuple

import torch
from torch import nn


class TrainingAKFMode(enum.Enum):
    FROZEN = 'frozen'  # Parameters are frozen
    MOTION = 'motion' # Uncertainty parameters are frozen
    UNCERTAINTY = 'uncertainty'  # Motion parameters are frozen
    ALL = 'all'  # All parameters are unfrozen

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
            return True, True

        raise AssertionError('Invalid Program State!')

    @classmethod
    def from_str(cls, value: str) -> 'TrainingAKFMode':
        """
        Creates training mode from string

        Args:
            value: String (raw)

        Returns:
            Training mode
        """
        for item in cls:
            if item.value.lower() == value.lower():
                return item

        raise ValueError(f'Can\'t deduce mode from "{value}"!')


class TrainableAdaptiveKalmanFilter(nn.Module):
    def __init__(self,
        sigma_p: float = 0.05,
        sigma_p_init_mult: float = 2.0,
        sigma_v: float = 0.00625,
        sigma_v_init_mult: float = 10.0,
        sigma_r: float = 1.0,
        sigma_q: float = 1.0,
        dt: float = 1.0,

        training_mode: TrainingAKFMode = TrainingAKFMode.FROZEN,
        positive_motion_mat: bool = True,
        triu_motion_mat: bool = True,
        first_principles_motion_mat: bool = True
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
            positive_motion_mat: Use positive motion matrix `A >= 0` (non-negative)
            triu_motion_mat: Use upper triangular motion matrix
            first_principles_motion_mat: Use first principles motion matrix as initial parameters
        """
        super().__init__()
        self._training_mode = training_mode
        train_motion_parameters, train_uncertainty_parameters = training_mode.flags

        self._positive_motion_mat = positive_motion_mat
        self._triu_motion_mat = triu_motion_mat
        self._first_principles_motion_mat = first_principles_motion_mat

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

        self._initialize_motion_matrix(
            dt=dt,
            train_motion_parameters=train_motion_parameters,
            first_principles_motion_mat=first_principles_motion_mat,
            positive_motion_mat=positive_motion_mat,
            triu_motion_mat=triu_motion_mat
        )

        self._H = nn.Parameter(torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=torch.float32), requires_grad=False)

        self._I = nn.Parameter(torch.eye(8, dtype=torch.float32), requires_grad=False)


    def _initialize_motion_matrix(
        self,
        dt: float,
        train_motion_parameters: bool,
        first_principles_motion_mat: bool,
        positive_motion_mat: bool,
        triu_motion_mat: bool
    ) -> None:
        """
        Initializes motion matrix

        Args:
            dt: Velocity time step hyperparameter
            train_motion_parameters: Set `requires_grad=True` for motion matrix
            first_principles_motion_mat: Use heuristic to initialize motion matrix
            positive_motion_mat: Initialize motion matrix to be positive
            triu_motion_mat: All element of motion matrix below diagonal are zeros
        """
        assert dt >= 0, f'Parameter `dt` can\'t be negative! Got {dt}.'

        if first_principles_motion_mat:
            A = torch.tensor([
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=torch.float32)
        else:
            A = torch.zeros(8, 8, dtype=torch.float32)
            nn.init.xavier_normal_(A)

        if positive_motion_mat:
            A = torch.relu(A)
        if triu_motion_mat:
            A = torch.relu(A)

        self._A = nn.Parameter(A, requires_grad=train_motion_parameters)

    def get_motion_matrix(self) -> torch.Tensor:
        """
        Preprocess motion matrix.

        Returns:
            Processed motion matrix
        """
        A = self._A

        if self._positive_motion_mat:
            A = torch.relu(A)
        if self._triu_motion_mat:
            A = torch.triu(A)

        return A

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
    def _create_diag_cov_matrix(self, diag: torch.Tensor) -> torch.Tensor:
        """
        Creates diagonal covariance matrix given list of stds.

        Args:
            diag: Diagonal elements

        Returns:
            Covariance matrix
        """
        diag = torch.square(diag)
        return torch.diag_embed(diag)

    def _estimate_process_noise_heuristic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates process noise matrix using bbox size heuristic. Noise is proportional to the bbox size.

        Args:
            x: State vector (last posterior)

        Returns:
            Process noise (Q) matrix
        """
        h, w = x[:, 2, 0], x[:, 3, 0]
        x_weight = h * self._sigma_p
        y_weight = w * self._sigma_p
        xv_weight = h * self._sigma_v
        yv_weight = w * self._sigma_v
        diag = torch.stack([x_weight, y_weight, x_weight, y_weight,
                            xv_weight, yv_weight, xv_weight, yv_weight]).T
        return self._create_diag_cov_matrix(diag)

    def _estimate_measurement_noise_heuristic(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Estimates measurement noise matrix using bbox size heuristic. Noise is proportional to the bbox size.

        Args:
            x_hat: State vector (last prior)

        Returns:
            Measurement noise (Q) matrix
        """
        h, w = x_hat[:, 2, 0], x_hat[:, 3, 0]
        x_weight = h * self._sigma_p
        y_weight = w * self._sigma_p
        diag = torch.stack([x_weight, y_weight, x_weight, y_weight]).T
        return self._create_diag_cov_matrix(diag)

    def initiate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initiate state vector and covariance (uncertainty) matrix

        Args:
            z: Measurement

        Returns:
            Initialized vector and covariance matrix
        """
        assert len(z.shape) == 2, f'Expected measurement shape is (batch_size, dim) but found: {z.shape}'
        batch_size, n_dim = z.shape
        z = z.unsqueeze(-1)

        x = torch.hstack([z, torch.zeros_like(z, dtype=torch.float32)])

        diag = torch.tensor([4 * [self._sigma_p * self._sigma_p_init_mult]
                             + 4 * [self._sigma_v * self._sigma_v_init_mult] for _ in range(batch_size)])
        P = self._create_diag_cov_matrix(diag).to(z)

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
        batch_size, *_ = x.shape
        A = self.get_motion_matrix()
        A_expanded = A.unsqueeze(0).expand(batch_size, *A.shape)
        AT_expanded = torch.transpose(A_expanded, dim0=-2, dim1=-1)

        # Predict state
        x_hat = torch.bmm(A_expanded, x)  # 8x1

        # Predict uncertainty
        Q = self._estimate_process_noise_heuristic(x).to(x)
        P_hat = torch.bmm(torch.bmm(A_expanded, P), AT_expanded) + Q  # 8x8

        return x_hat, P_hat

    def forward(self, x: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict(x, P)

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

        x_preds, P_preds = [], []
        for _ in range(n_steps):
            x_hat, P_hat = self.predict(x_hat, P_hat)
            x_proj, P_proj = self.project(x_hat, P_hat)
            x_preds.append(x_proj)
            P_preds.append(P_proj)

        x_preds = torch.stack(x_preds)
        P_preds = torch.stack(P_preds)

        return x_preds, P_preds

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
        batch_size, *_ = x_hat.shape
        z = z.unsqueeze(-1)

        # Update state
        x_proj, P_proj = self.project(x_hat, P_hat, flatten=False)

        H = self._H.to(z)
        H_expanded = H.unsqueeze(0).expand(batch_size, *H.shape)
        HT_expanded = torch.transpose(H_expanded, dim0=-2, dim1=-1)

        # TODO: Use Cholesky batch matrix decomposition instead of `torch.inverse` for better numerical stability
        K = torch.bmm(torch.bmm(P_hat, HT_expanded), torch.inverse(P_proj))  # 8x4

        # validation: (8x8 @ 8x4) @ inv(4x8 @ 8x8 @ 8x4) = 8x4 @ 4x4 = 8x4
        innovation = z - x_proj  # 4x1
        x = x_hat + torch.bmm(K, innovation)  # 8x1
        # validation: 8x1 + 8x4 @ 4x1 = 8x1 + 8x1

        # Update uncertainty
        I = self._I.to(z)
        I_expanded = I.unsqueeze(0).expand(batch_size, *I.shape)
        P = torch.bmm(I_expanded - torch.bmm(K, H_expanded), P_hat)  # 8x8
        # validation: (8x8 - 8x4 @ 4x8) @ 8x8 = 8x8 @ 8x8 = 8x8

        return x, P

    def project(self, x: torch.Tensor, P: torch.Tensor, flatten: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects state vector to measurement space.

        Args:
            x: State vector
            P: State uncertainty covariance matrix
            flatten: Flatten state vector `x`

        Returns:
            Projected state vector, projected uncertain covariance matrix
        """
        batch_size, *_ = x.shape

        R = self._estimate_measurement_noise_heuristic(x).to(x)
        H = self._H.to(x)
        H_expanded = H.unsqueeze(0).expand(batch_size, *H.shape)
        HT_expanded = torch.transpose(H_expanded, dim0=-2, dim1=-1)

        x_proj = torch.bmm(H_expanded, x)
        if flatten:
            x_proj = x_proj.squeeze(-1)
        P_proj = torch.bmm(torch.bmm(H_expanded, P), HT_expanded) + R

        return x_proj, P_proj


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

    kf_preds1, kf_preds2 = [], []
    kf_ests1, kf_ests2 = [], []
    ground_truth1, ground_truth2 = [], []

    # KF prediction: A -> B
    first_iteration = True
    x_hat, P_hat, x, P = None, None, None, None
    for i in range(time_len):
        t = dt*i
        gt1 = ((1 - t/time_len) * state_A + t/time_len * state_B)
        gt2 = ((1 - t / time_len) * state_B + t / time_len * state_C)
        gt = torch.stack([gt1, gt2])
        measurement = gt + torch.randn_like(gt) * noise_std
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

        ground_truth1.append(gt[0])
        kf_preds1.append(pred[0])
        kf_ests1.append(est[0])

        ground_truth2.append(gt[1])
        kf_preds2.append(pred[1])
        kf_ests2.append(est[1])

    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    SIZE = 200
    mp4_path = os.path.join(OUTPUTS_PATH, 'tkf_simulation.mp4')

    # noinspection PyUnresolvedReferences
    mp4_writer = cv2.VideoWriter(mp4_path, fourcc, 5, (SIZE, SIZE))

    kf_preds = kf_preds1 + kf_preds2
    ground_truth = ground_truth1 + ground_truth2

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
