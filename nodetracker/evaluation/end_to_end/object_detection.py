"""
Object Detection support for E2E evaluation.
Supports YOLOv8 realtime inference and Mock object detection with ground truths.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import ultralytics

from nodetracker.datasets import TrajectoryDataset
from nodetracker.datasets.mot.core import SceneInfoIndex, MOTDataset
from nodetracker.library.cv.bbox import PredBBox, BBox
from nodetracker.utils import file_system
from nodetracker.utils.lookup import LookupTable


class ObjectDetectionInference(ABC):
    """
    ObjectDetection inference interface.
    """
    def __init__(self, dataset: TrajectoryDataset, lookup: LookupTable):
        self._dataset = dataset
        self._lookup = lookup

    @abstractmethod
    def predict(
        self,
        scene_name: str,
        frame_index: int,
        object_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Generates list of bbox prediction based on given data (any input)

        Args:
            scene_name: Prediction scene name
            frame_index: Frame id
            object_id: Object id (optional leakage)

        Returns:
            List of bbox predictions
        """
        pass


class GroundTruthInference(ObjectDetectionInference):
    """
    Object Detection Mock (returns ground truths).
    """
    def __init__(self, dataset: TrajectoryDataset, lookup: LookupTable):
        super().__init__(dataset=dataset, lookup=lookup)

    def predict(
        self,
        scene_name: str,
        frame_index: int,
        object_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        # If `object_id` is given then only one bbox is returned (if it is present)
        object_ids = [object_id] if object_id is not None else self._dataset.get_scene_object_ids(scene_name)
        bboxes, classes = [], []

        for object_id in object_ids:
            data = self._dataset.get_object_data_label_by_frame_index(object_id, frame_index)
            if data is None:
                continue

            bboxes.append(torch.tensor(data['bbox'], dtype=torch.float32))
            classes.append(data['category'])

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 \
            else torch.empty(0, 4, dtype=torch.float32)
        confs = torch.ones(bboxes.shape[0], dtype=torch.float32) if bboxes.shape[0] > 0 \
            else torch.empty(0, 4, dtype=torch.float32)

        return bboxes, classes, confs


class YOLOv8Inference(ObjectDetectionInference):
    """
    Object Detection - YOLOv8
    """
    def __init__(
        self,
        dataset: TrajectoryDataset,
        lookup: LookupTable,
        model_path: str,
        accelerator: str,
        verbose: bool = False,
        conf: float = 0.25,
        known_class: bool = False
    ):
        super().__init__(dataset=dataset, lookup=lookup)
        self._yolo = ultralytics.YOLO(model_path)
        self._yolo.to(accelerator)
        self._verbose = verbose
        self._conf = conf
        self._known_class = known_class

    def predict(
        self,
        scene_name: str,
        frame_index: int,
        object_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        image_path = self._dataset.get_scene_image_path(scene_name, frame_index)

        # noinspection PyUnresolvedReferences
        image = cv2.imread(image_path)
        assert image is not None, f'Failed to load image "{image_path}".'

        h, w, _ = image.shape
        prediction_raw = self._yolo.predict(
            source=image,
            verbose=self._verbose,
            conf=self._conf,
            classes=self._lookup.lookup(self._dataset.get_scene_info(scene_name).category)
                if self._known_class else None
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


class YOLOXInference(ObjectDetectionInference):
    """
    Object Detection - YOLOX
    """
    def __init__(
        self,
        dataset: TrajectoryDataset,
        lookup: LookupTable,
        model_path: str,
        accelerator: str,
        conf: float = 0.01,
        min_bbox_area: int = 0,
        cache_path: Optional[str] = None,
        exp_path: Optional[str] = None,
        exp_name: Optional[str] = None,
        legacy: bool = True
    ):
        super().__init__(dataset=dataset, lookup=lookup)
        from nodetracker.object_detection.yolox import YOLOXPredictor, DEFAULT_EXP_PATH, DEFAULT_EXP_NAME
        exp_path = exp_path if exp_path is not None else DEFAULT_EXP_PATH
        exp_name = exp_name if exp_name is not None else DEFAULT_EXP_NAME

        self._yolox = YOLOXPredictor(
            checkpoint_path=model_path,
            accelerator=accelerator,
            conf_threshold=conf,
            exp_path=exp_path,
            exp_name=exp_name,
            legacy=legacy
        )
        self._conf = conf
        self._min_bbox_area = min_bbox_area
        self._cache_path = cache_path

        if self._cache_path is not None:
            Path(self._cache_path).mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, scene_name: str, frame_index: int) -> str:
        frame_name = f'{scene_name}_{frame_index:06d}'
        return os.path.join(self._cache_path, f'{frame_name}.npy')

    def _predict(self, scene_name: str, frame_index: int) -> np.ndarray:
        """
        Performs model image loading + inference with cache support (faster inference in second run).
        In cache the output can be found in cache then image loading and inference is skipped

        Args:
            scene_name: Scene name
            frame_index: frame index

        Returns:
            Model output
        """
        if self._cache_path is not None:
            frame_cache_path = self._get_cache_path(scene_name, frame_index)
            if os.path.exists(frame_cache_path):
                with open(frame_cache_path, 'rb') as f:
                    return np.load(f)

        image_path = self._dataset.get_scene_image_path(scene_name, frame_index)

        # noinspection PyUnresolvedReferences
        image = cv2.imread(image_path)
        assert image is not None, f'Failed to load image "{image_path}".'
        output, _ = self._yolox.predict(image)

        if self._cache_path is not None:
            frame_cache_path = self._get_cache_path(scene_name, frame_index)
            with open(frame_cache_path, 'wb') as f:
                np.save(f, output)

        return output

    def predict(
        self,
        scene_name: str,
        frame_index: int,
        object_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        scene_info = self._dataset.get_scene_info(scene_name)

        h, w = scene_info.imheight, scene_info.imwidth
        output = self._predict(scene_name, frame_index)
        output = torch.from_numpy(output)

        # Filter small bboxes
        bboxes = output[:, :4]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        output = output[areas >= self._min_bbox_area]

        # Process bboxes
        bboxes = output[:, :4]
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h
        bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to xywh

        # Process classes
        class_indices = output[:, 6]
        classes = [self._lookup.inverse_lookup(int(cls_index)) for cls_index in class_indices]

        # Process confidences
        confidences = output[:, 4]

        return bboxes, classes, confidences


MOT20Detections = Dict[str, Dict[str, List[List[float]]]]


class MOT20OfflineInference(ObjectDetectionInference):
    def __init__(self, dataset: TrajectoryDataset, lookup: LookupTable, dataset_path: str):
        super().__init__(dataset=dataset, lookup=lookup)

        # Parse data
        self._scene_info_index = self._create_scene_info_index(dataset_path)
        self._detections = self._parse_detection(self._scene_info_index)

    @staticmethod
    def _create_scene_info_index(dataset_path) -> SceneInfoIndex:
        scene_names = file_system.listdir(dataset_path)
        scene_info_index: SceneInfoIndex = {}

        for scene_name in scene_names:
            scene_filepath = os.path.join(dataset_path, scene_name)
            scene_info_index[scene_name] = MOTDataset.parse_scene_ini_file(scene_filepath, 'det')

        return scene_info_index

    @staticmethod
    def _parse_detection(scene_info_index: SceneInfoIndex) -> MOT20Detections:
        detections: MOT20Detections = {scene_name: {} for scene_name in scene_info_index.keys()}

        for scene_name, scene_info in scene_info_index.items():
            df = pd.read_csv(scene_info.gt_path, header=None)
            df = df.iloc[:, :7]
            df.columns = ['frame_id', 'object_id', 'ymin', 'xmin', 'w', 'h', 'conf']  # format: yxwh
            # df = df[df['conf'] == 1]
            df = df.drop(columns=['object_id', 'conf'], axis=1)
            df['ymin'] /= scene_info.imwidth
            df['xmin'] /= scene_info.imheight
            df['w'] /= scene_info.imwidth
            df['h'] /= scene_info.imheight

            for _, row in df.iterrows():
                frame_id = row.iloc[0]
                bbox = row.iloc[1:].values.tolist()

                if frame_id not in detections[scene_name]:
                    detections[scene_name][frame_id] = []
                detections[scene_name][frame_id].append(bbox)

        return detections

    def predict(
        self,
        scene_name: str,
        frame_index: int,
        object_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        category = self._dataset.get_scene_info(scene_name).category
        frame_id = str(frame_index + 1)

        bboxes = torch.tensor(self._detections[scene_name].get(frame_id, []), dtype=torch.float32)
        classes = [category] * bboxes.shape[0]
        conf = torch.tensor([1] * bboxes.shape[0], dtype=torch.float32)
        return bboxes, classes, conf


def object_detection_inference_factory(
    name: str,
    params: dict,
    dataset: TrajectoryDataset,
    lookup: LookupTable
) -> ObjectDetectionInference:
    """
    Creates object detection inference for single object tracking (filter evaluation) given name and parameters.

    Args:
        name: OD inference name (type)
        params: CTor parameters
        dataset: Dataset info
        lookup: Used to decode classes

    Returns:
        Initialized OD inference object for filter evaluation
    """
    catalog = {
        'ground_truth': GroundTruthInference,
        'yolo': YOLOv8Inference,
        'yolox': YOLOXInference,
        'mot20_offline': MOT20OfflineInference
    }

    name = name.lower()
    if name not in catalog:
        raise ValueError(f'Name "{name}" not found in catalog. Available options: {list(catalog.keys())}.')

    return catalog[name](**params, dataset=dataset, lookup=lookup)


@torch.no_grad()
def create_bbox_objects(
    inf_bboxes: torch.Tensor,
    inf_classes: List[str],
    inf_conf: torch.Tensor,
    clip: bool = False
) -> List[PredBBox]:
    """
    Creates bboxes from raw outputs.

    Args:
        inf_bboxes: Inference bboxes
        inf_classes: Inference bbox classes
        inf_conf: Inference bbox confidences
        clip: Clip bboxes

    Returns:
        List of PredBBox objects
    """
    inf_bboxes = inf_bboxes.detach().cpu().numpy()
    inf_conf = inf_conf.detach().cpu().numpy()
    assert len(inf_bboxes) == len(inf_classes) == len(inf_conf)

    n_bboxes = len(inf_classes)
    bboxes: List[PredBBox] = []

    for i in range(n_bboxes):
        bbox_coords = [float(v) for v in inf_bboxes[i]]
        bbox = PredBBox.create(
            bbox=BBox.from_yxwh(*bbox_coords, clip=clip),
            label=inf_classes[i],
            conf=float(inf_conf[i])
        )
        bboxes.append(bbox)

    return bboxes
