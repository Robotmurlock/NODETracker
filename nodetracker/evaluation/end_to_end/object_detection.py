"""
Object Detection support for E2E evaluation.
Supports YOLOv8 realtime inference and Mock object detection with ground truths.
"""
from abc import ABC, abstractmethod

import cv2
import torch
import ultralytics


class ObjectDetectionInference(ABC):
    """
    ObjectDetection inference interface.
    """
    @abstractmethod
    def predict(self, data: dict) -> torch.Tensor:
        """
        Generates list of bbox prediction based on given data (any input)

        Args:
            data: Data

        Returns:
            List of bbox predictions
        """
        pass


class GroundTruthInference(ObjectDetectionInference):
    """
    Object Detection Mock (returns ground truths).
    """
    def __init__(self, **kwargs):
        pass  # Compatibility with factory method

    def predict(self, data: dict) -> torch.Tensor:
        return torch.tensor(data['bbox'], dtype=torch.float32).view(1, 4)


class YOLOv8Inference(ObjectDetectionInference):
    """
    Object Detection - YOLOv8
    """
    def __init__(self, model_path: str, accelerator: str, verbose: bool = False):
        self._yolo = ultralytics.YOLO(model_path)
        self._yolo.to(accelerator)

        self._verbose = verbose

    def predict(self, data: dict) -> torch.Tensor:
        image_path = data['image_path']
        # noinspection PyUnresolvedReferences
        image = cv2.imread(image_path)
        assert image is not None, f'Failed to load image "{image_path}".'

        h, w, _ = image.shape
        prediction_raw = self._yolo.predict(image, verbose=self._verbose)[0]
        prediction = prediction_raw.boxes.xyxy.detach().cpu()

        prediction[:, [0, 2]] /= w
        prediction[:, [1, 3]] /= h
        prediction[:, 2:] -= prediction[:, :2]  # xyxy to xywh

        return prediction


def object_detection_inference_factory(name: str, params: dict) -> ObjectDetectionInference:
    """
    Creates object detection inference for given name and parameters.

    Args:
        name: OD inference name (type)
        params: CTor parameters

    Returns:
        Initialized OD inference object
    """
    catalog = {
        'ground_truth': GroundTruthInference,
        'yolo': YOLOv8Inference
    }

    name = name.lower()
    if name not in catalog:
        raise ValueError(f'Name "{name}" not found in catalog. Available options: {list(catalog.keys())}.')

    return catalog[name](**params)
