"""
Object Detection support for E2E evaluation.
Supports YOLOv8 realtime inference and Mock object detection with ground truths.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import torch
import ultralytics

from nodetracker.utils.lookup import LookupTable


class ObjectDetectionInference(ABC):
    """
    ObjectDetection inference interface.
    """
    def __init__(self, lookup: LookupTable):
        self._lookup = lookup

    @abstractmethod
    def predict(self, data: dict) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
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
    def __init__(self, lookup: LookupTable):
        super().__init__(lookup=lookup)

    def predict(self, data: dict) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        bboxes = torch.tensor(data['bbox'], dtype=torch.float32).view(1, 4)
        classes = [data['category']]
        conf = torch.tensor([1], dtype=torch.float32)
        return bboxes, classes, conf


class YOLOv8Inference(ObjectDetectionInference):
    """
    Object Detection - YOLOv8
    """
    def __init__(
        self,
        lookup: LookupTable,
        model_path: str,
        accelerator: str,
        verbose: bool = False,
        conf: float = 0.25,
        known_class: bool = False
    ):
        super().__init__(lookup=lookup)
        self._yolo = ultralytics.YOLO(model_path)
        self._yolo.to(accelerator)
        self._verbose = verbose
        self._conf = conf
        self._known_class = known_class

    def predict(self, data: dict) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        image_path = data['image_path']
        # noinspection PyUnresolvedReferences
        image = cv2.imread(image_path)
        assert image is not None, f'Failed to load image "{image_path}".'

        h, w, _ = image.shape
        prediction_raw = self._yolo.predict(
            source=image,
            verbose=self._verbose,
            conf=self._conf,
            classes=self._lookup.lookup(data['category']) if self._known_class else None
        )[0]  # Remove batch

        # Process bboxes
        bboxes = prediction_raw.boxes.xyxy.detach().cpu()
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h
        bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to xywh

        # Process classes
        class_indices = prediction_raw.boxes.cls.detach().cpu()
        classes = [self._lookup.inverse_lookup(int(cls_index)) for cls_index in class_indices.view(-1)]

        # Process confidences
        confidences = prediction_raw.boxes.conf.detach().cpu().view(-1)

        return bboxes, classes, confidences


def object_detection_inference_factory(name: str, params: dict, lookup: LookupTable) -> ObjectDetectionInference:
    """
    Creates object detection inference for given name and parameters.

    Args:
        name: OD inference name (type)
        params: CTor parameters
        lookup: Used to decode classes

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

    return catalog[name](**params, lookup=lookup)
