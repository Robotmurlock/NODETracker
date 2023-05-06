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

    def test_iou_same_bboxes(self):
        # Arrange
        bboxes1 = np.array([[0, 0, 1, 1], [2, 2, 4, 4]], dtype=np.float32)
        bboxes2 = np.array([[0, 0, 1, 1], [2, 2, 4, 4]], dtype=np.float32)
        expected_iou_scores = np.array([1.0, 1.0], dtype=np.float32)

        # Act - Assert
        for is_xyhw_format in [True, False]:
            iou_scores = sot_eval.iou(bboxes1, bboxes2, is_xyhw_format=is_xyhw_format)
            self.assertTrue(np.allclose(iou_scores, expected_iou_scores))

    def test_iou_completely_separate_bboxes(self):
        # Arrange
        bboxes1 = np.array([[0, 0, 1, 1], [2, 2, 4, 4]], dtype=np.float32)
        bboxes2 = np.array([[10, 10, 11, 11], [20, 20, 24, 24]], dtype=np.float32)
        expected_iou_scores = np.array([0.0, 0.0], dtype=np.float32)

        # Act - Assert
        iou_scores = sot_eval.iou(bboxes1, bboxes2, is_xyhw_format=False)
        self.assertTrue(np.allclose(iou_scores, expected_iou_scores))

    def test_iou_all_overlapping_bboxes(self):
        # Arrange
        bboxes1 = np.array([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.float32)
        bboxes2 = np.array([[1, 1, 3, 3], [0, 0, 2, 2]], dtype=np.float32)
        expected_iou_scores = np.array([1 / 7, 1 / 7], dtype=np.float32)

        # Act - Assert
        iou_scores = sot_eval.iou(bboxes1, bboxes2, is_xyhw_format=False)
        self.assertTrue(np.allclose(iou_scores, expected_iou_scores))

    def test_iou_some_overlapping_bboxes(self):
        # Arrange
        bboxes1 = np.array([[-1, -1, 0, 0], [1, 1, 3, 3]], dtype=np.float32)
        bboxes2 = np.array([[1, 1, 3, 3], [0, 0, 2, 2]], dtype=np.float32)
        expected_iou_scores = np.array([0, 1 / 7], dtype=np.float32)

        # Act - Assert
        iou_scores = sot_eval.iou(bboxes1, bboxes2, is_xyhw_format=False)
        self.assertTrue(np.allclose(iou_scores, expected_iou_scores))

    def test_iou_empty_bboxes(self):
        # Arrange
        bboxes1 = np.array([], dtype=np.float32)
        bboxes2 = np.array([], dtype=np.float32)
        expected_iou_scores = np.array([], dtype=np.float32)

        # Act - Assert
        iou_scores = sot_eval.iou(bboxes1, bboxes2, is_xyhw_format=False)
        self.assertTrue(np.allclose(iou_scores, expected_iou_scores))

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
        self.assertAlmostEqual(expected_accuracy, accuracy, places=6)

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
        self.assertAlmostEqual(expected_accuracy, accuracy, places=6)

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
        success = sot_eval.success(gt_traj, pred_traj, threshold=0.5)

        # Assert
        expected_success = 2 / 3
        self.assertAlmostEqual(expected_success, success, places=6)

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
        success = sot_eval.success(gt_traj, pred_traj, threshold=0.5)

        # Assert
        expected_success = 5 / 6
        self.assertAlmostEqual(expected_success, success, places=6)


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
        precision = sot_eval.precision(gt_traj, pred_traj, imwidth=imwidth, imheight=imheight, threshold=20)

        # Assert
        expected_precision = 2 / 3
        self.assertAlmostEqual(expected_precision, precision, places=6)

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
        precision = sot_eval.precision(gt_traj, pred_traj, imwidth=imwidth, imheight=imheight, threshold=20)

        # Assert
        expected_precision = 5 / 6
        self.assertAlmostEqual(expected_precision, precision, places=6)

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
        norm_precision = sot_eval.norm_precision(gt_traj, pred_traj, threshold=0.1)

        # Assert
        expected_norm_precision = 2 / 3
        self.assertAlmostEqual(expected_norm_precision, norm_precision, places=6)

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
        norm_precision = sot_eval.norm_precision(gt_traj, pred_traj, threshold=0.1)

        # Assert
        expected_norm_precision = 5 / 6
        self.assertAlmostEqual(expected_norm_precision, norm_precision, places=6)
