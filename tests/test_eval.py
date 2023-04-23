import unittest
import numpy as np

from evaluation import sot as sot_eval


# noinspection DuplicatedCode
class TestEvalMetrics(unittest.TestCase):
    def test_mse(self):
        # Arrange
        gt_traj = np.array([
            [1, 1, 1, 1],
            [2, 3, 4, 5],
            [3, 5, 7, 9]
        ], dtype=np.float32)
        pred_traj = np.array([
            [1, 1, 1, 1],
            [2.1, 2.9, 4.1, 4.9],
            [3.1, 4.9, 7.1, 8.9]
        ], dtype=np.float32)

        # Act
        mse = sot_eval.mse(gt_traj, pred_traj)

        # Assert
        self.assertAlmostEqual(8 * 0.01 / 12, mse)

    def test_mse_batch(self):
        # Arrange
        gt_traj = np.array([
            [
                [1, 1, 1, 1],
                [2, 3, 4, 5],
                [3, 5, 7, 9]
            ],
            [
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4]
            ]
        ], dtype=np.float32)
        pred_traj = np.array([
            [
                [1, 1, 1, 1],
                [2.1, 2.9, 4.1, 4.9],
                [3.1, 4.9, 7.1, 8.9]
            ],
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]
            ]
        ], dtype=np.float32)

        # Act
        mse = sot_eval.mse(gt_traj, pred_traj)

        # Assert
        self.assertAlmostEqual((8 * 0.01 + 12) / 24, mse)

    def test_accuracy(self):
        # Arrange
        gt_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.2, 0.3, 0.2, 0.2],
            [0.3, 0.5, 0.4, 0.4]
        ], dtype=np.float32)
        pred_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.21, 0.31, 0.2, 0.2],
            [0.61, 0.69, 0.3, 0.3]
        ], dtype=np.float32)

        # Act
        accuracy = sot_eval.accuracy(gt_traj, pred_traj)

        # Assert
        expected_accuracy = (1.0 + 0.82232346241 + 0.08178277801) / 3
        self.assertAlmostEqual(expected_accuracy, accuracy)

    def test_accuracy_batch(self):
        # Arrange
        gt_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.2, 0.3, 0.2, 0.2],
                [0.3, 0.5, 0.4, 0.4]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)
        pred_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.21, 0.31, 0.2, 0.2],
                [0.31, 0.49, 0.4, 0.4]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)

        # Act
        accuracy = sot_eval.accuracy(gt_traj, pred_traj)

        # Assert
        expected_accuracy = (1.0 + 0.82232346241 + 0.90589636688 + 3 * 1.0) / 6
        self.assertAlmostEqual(expected_accuracy, accuracy)

    def test_success(self):
        # Arrange
        gt_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.2, 0.3, 0.2, 0.2],
            [0.3, 0.5, 0.4, 0.4]
        ], dtype=np.float32)
        pred_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.21, 0.31, 0.2, 0.2],
            [0.61, 0.69, 0.3, 0.3]
        ], dtype=np.float32)

        # Act
        success = sot_eval.success(gt_traj, pred_traj)

        # Assert
        expected_success = 2 / 3
        self.assertAlmostEqual(expected_success, success)

    def test_success_batch(self):
        # Arrange
        gt_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.2, 0.3, 0.2, 0.2],
                [0.3, 0.5, 0.4, 0.4]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)
        pred_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.21, 0.31, 0.2, 0.2],
                [0.61, 0.69, 0.3, 0.3]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)

        # Act
        success = sot_eval.success(gt_traj, pred_traj)

        # Assert
        expected_success = 5 / 6
        self.assertAlmostEqual(expected_success, success)


    def test_precision(self):
        # Arrange
        gt_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.2, 0.3, 0.2, 0.2],
            [0.3, 0.5, 0.4, 0.4]
        ], dtype=np.float32)
        pred_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.21, 0.31, 0.2, 0.2],
            [0.61, 0.69, 0.3, 0.3]
        ], dtype=np.float32)
        imwidth, imheight = 1200, 800

        # Act
        precision = sot_eval.precision(gt_traj, pred_traj, imwidth=imwidth, imheight=imheight)

        # Assert
        expected_precision = 2 / 3
        self.assertAlmostEqual(expected_precision, precision)

    def test_precision_batch(self):
        # Arrange
        gt_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.2, 0.3, 0.2, 0.2],
                [0.3, 0.5, 0.4, 0.4]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)
        pred_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.21, 0.31, 0.2, 0.2],
                [0.61, 0.69, 0.3, 0.3]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)
        imwidth, imheight = 1200, 800

        # Act
        precision = sot_eval.precision(gt_traj, pred_traj, imwidth=imwidth, imheight=imheight)

        # Assert
        expected_precision = 5 / 6
        self.assertAlmostEqual(expected_precision, precision)

    def test_norm_precision(self):
        # Arrange
        gt_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.2, 0.3, 0.2, 0.2],
            [0.3, 0.5, 0.4, 0.4]
        ], dtype=np.float32)
        pred_traj = np.array([
            [0.1, 0.1, 0.05, 0.05],
            [0.21, 0.31, 0.2, 0.2],
            [0.61, 0.69, 0.3, 0.3]
        ], dtype=np.float32)

        # Act
        norm_precision = sot_eval.norm_precision(gt_traj, pred_traj)

        # Assert
        expected_norm_precision = 2 / 3
        self.assertAlmostEqual(expected_norm_precision, norm_precision)

    def test_norm_precision_batch(self):
        # Arrange
        gt_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.2, 0.3, 0.2, 0.2],
                [0.3, 0.5, 0.4, 0.4]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)
        pred_traj = np.array([
            [
                [0.1, 0.1, 0.05, 0.05],
                [0.21, 0.31, 0.2, 0.2],
                [0.61, 0.69, 0.3, 0.3]
            ],
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4]
            ]
        ], dtype=np.float32)

        # Act
        norm_precision = sot_eval.norm_precision(gt_traj, pred_traj)

        # Assert
        expected_norm_precision = 5 / 6
        self.assertAlmostEqual(expected_norm_precision, norm_precision)
