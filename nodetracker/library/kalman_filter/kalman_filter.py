"""
*Algorithm*

Initial State:
    - x, y (zeros)
    - v_x, v_y (zeros)

Initial Uncertainty:
    - std_x, std_y (high values ~ SIZE0 for eye matrix)

Prediction (a priori) Step:
    Generic State Prediction model:
        x_hat[k] = A @ x[k-1] + B @ u[k-1] + w[k-1]

        where:
            k - current state index
            (k-1) - previous state index
            x - state
            x_hat - predicted state
            u - control input
            w - process noise vector (ignored) with covariance matrix Q

            A - state transition matrix
            B - control input matrix

    Constant velocity model (u := 0) with ignored process noise vector:
        x_hat[k] = A @ x[k-1]

    Measurement model (transforming state to measurement space):
        z[k] = H @ x[k]

    Generic Uncertainty Prediction model:
        P_hat[k] = A @ P[k-1] @ A.T + Q

        where:
            P - covariance error matrix
            P_hat - predicted covariance error matrix

Update (a posteriori) Step:
    Update State model:
        x[k] = x_hat[k] + K @ (z[k] - H @ x_hat[k])

        where
            z - measurement
            K - Kalman gain

    Kalman Gain:
        K = (P[k] @ H.T) @ (H @ P[k] @ H.T + R).inv

        where
            H - transformation matrix to measurement format
            R - measurement noise

    Update Uncertainty model:
        P[k] = (I - K @ H) @ P_hat[k]


"""
from typing import Tuple, Optional

import numpy as np
import numpy.linalg as LA


class ConstantVelocityODKalmanFilter:
    """
    Implementation of constant velocity Kalman filter. It is assumed that only x, y, w and h measurements are available
    from object detector but x', y', w', and h' (velocities) are not available.
    """
    def __init__(
        self,
        initial_position_uncertainty: float = 10,
        initial_velocity_uncertainty: float = 1000,
        process_noise_multiplier: float = 1.0,
        measurement_noise_multiplier: float = 1.0,
        z_initial: Optional[np.ndarray] = None
    ):
        """
        Args:
            initial_position_uncertainty: Initial uncertainty for object position (x, y, w, h)
            initial_velocity_uncertainty: Initial uncertainty for object velocities (x', y', w', h')
            process_noise_multiplier: Q multiplier
            measurement_noise_multiplier: R multiplier
        """
        self._initial_position_uncertainty = initial_position_uncertainty
        self._initial_velocity_uncertainty = initial_velocity_uncertainty

        # z ~ x, y, w, h
        self._x_hat = np.zeros(shape=(8, 1))  # x, y, w, h, x', y', w', h'
        self._x = np.zeros(shape=(8, 1))

        self._H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        self._P_hat = np.eye(8, dtype=np.float32)
        self._P_hat[:4, :4] *= self._initial_position_uncertainty
        self._P_hat[4:, 4:] *= self._initial_velocity_uncertainty
        self._P = self._P_hat.copy()

        self._Q = np.eye(8, dtype=np.float32) * process_noise_multiplier
        self._R = np.eye(4, dtype=np.float32) * measurement_noise_multiplier

        self._I = np.eye(8)

        self._predict_performed = False

        if z_initial is not None:
            self.set_z(z_initial)

    def set_z(self, z: np.ndarray) -> None:
        """
        Sets state to some value. Used for initial KF step (first bbox).
        Note: This resets KF state.

        Args:
            z: Measurement
        """
        self.reset_state()
        self._x[:4] = z.reshape(-1, 1)

    def get_z(self) -> np.ndarray:
        """
        Gets current state observable values.

        Returns:
            Observable values state
        """
        return self._H @ self._x

    def reset_state(self) -> None:
        """
        Resets KalmanFilter state.
        """
        self._P_hat = np.eye(8, dtype=np.float32)
        self._P_hat[:4, :4] *= self._initial_position_uncertainty
        self._P_hat[4:, 4:] *= self._initial_velocity_uncertainty
        self._P = self._P_hat.copy()

        self._x_hat = np.zeros(shape=(8, 1))
        self._x = np.zeros(shape=(8, 1))

    # noinspection PyMethodMayBeStatic
    def _get_A(self, dt: float) -> np.ndarray:
        """
        Creates transformation matrix A with time different `dt`.

        Args:
            dt:

        Returns:

        """
        return np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs prediction (a priori, initial) step.

        Args:
            dt: Time difference (time step)

        Returns:
            Predicted object bbox coordinates and uncertainty for those coordinates
        """
        A = self._get_A(dt)
        # Predict state
        self._x_hat = A @ self._x  # 8x1

        # Predict uncertainty
        self._P_hat = A @ self._P @ A.T + self._Q  # 8x8

        self._predict_performed = True

        z_predicted = self._H @ self._x_hat
        Pz_predicted = self._H @ self._P_hat @ self._H.T
        return z_predicted, Pz_predicted

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs update (a posteriori) step.

        Args:
            z: Measurements

        Returns:
            Updated object bbox coordinates and uncertainty for those coordinates
        """
        expected_shape = (4, 1)
        assert z.shape == expected_shape, f'Expected shape {expected_shape} but found {z.shape}!'
        assert self._predict_performed, 'Predict was not performed!'
        self._predict_performed = False

        # Update state
        K = (self._P_hat @ self._H.T) @ LA.inv(self._H @ self._P_hat @ self._H.T + self._R)  # 8x4
        # validation: (8x8 @ 8x4) @ inv(4x8 @ 8x8 @ 8x4) = 8x4 @ 4x4 = 8x4
        z_hat = self._H @ self._x_hat  # 4x1
        innovation = z - z_hat  # 4x1
        self._x = self._x_hat + K @ innovation  # 8x1
        # validation: 8x1 + 8x4 @ 4x1 = 8x1 + 8x1

        # Update uncertainty
        self._P = (self._I - K @ self._H) @ self._P_hat  # 8x8
        # validation: (8x8 - 8x4 @ 4x8) @ 8x8 = 8x8 @ 8x8 = 8x8

        z_updated = self._H @ self._x
        P_updated = self._H @ self._P @ self._H.T
        return z_updated, P_updated


# noinspection DuplicatedCode
def run_kf_test():
    import cv2
    import os
    from nodetracker.common.project import OUTPUTS_PATH

    kf = ConstantVelocityODKalmanFilter()

    state_A = np.array([[0.5, 0.5, 0.1, 0.1]]).T
    state_B = np.array([[0.2, 0.1, 0.2, 0.3]]).T
    state_C = np.array([[0.8, 0.9, 0.05, 0.05]]).T
    time_len = 20
    dt = 1.0
    noise_std = 1e-3

    kf_preds = []
    kf_ests = []
    ground_truth = []
    ts = []

    # KF prediction: A -> B
    first_iteration = True
    for i in range(time_len):
        t = dt*i
        gt = (1 - t/time_len) * state_A + t/time_len * state_B
        measurement = gt + np.random.randn(*gt.shape) * noise_std
        if first_iteration:
            kf.set_z(measurement)
            first_iteration = False
            pred = est = kf.get_z()
        else:
            pred, _ = kf.predict(dt)
            est, _ = kf.update(measurement)

        ts.append(dt*i)
        ground_truth.append(gt)
        kf_preds.append(pred)
        kf_ests.append(est)

    # KF prediction: B -> C
    for i in range(time_len):
        t = dt*i
        gt = (1 - t/time_len) * state_B + t/time_len * state_C
        measurement = gt + np.random.randn(*gt.shape) * noise_std

        pred, _ = kf.predict(dt)
        est, _ = kf.update(measurement)

        ts.append(dt*i)
        ground_truth.append(gt)
        kf_preds.append(pred)
        kf_ests.append(est)

    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    SIZE = 200
    # noinspection PyUnresolvedReferences
    mp4_writer = cv2.VideoWriter(os.path.join(OUTPUTS_PATH, 'simulation.mp4'), fourcc, 5, (SIZE, SIZE))

    for kf_pred, gt in zip(kf_preds, ground_truth):
        kf_x, kf_y, kf_w, kf_h = np.squeeze(kf_pred.T)
        kf_xmin = round(max(0, kf_x - kf_w) * SIZE)
        kf_ymin = round(max(0, kf_y - kf_h) * SIZE)
        kf_xmax = round(max(0, kf_x + kf_w) * SIZE)
        kf_ymax = round(max(0, kf_y + kf_h) * SIZE)

        gt_x, gt_y, gt_w, gt_h = np.squeeze(gt.T)
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
