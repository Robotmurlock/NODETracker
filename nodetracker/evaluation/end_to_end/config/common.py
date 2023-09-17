from dataclasses import dataclass


@dataclass
class FilterConfig:
    type: str
    params: dict


@dataclass
class ObjectDetectionInferenceConfig:
    type: str
    params: dict