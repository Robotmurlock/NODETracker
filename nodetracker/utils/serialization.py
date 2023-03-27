from abc import ABC
from dataclasses import dataclass, asdict
from typing import Union, Dict, Any, List

import dacite

JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


@dataclass
class Serializable(ABC):
    def serialize(self) -> JSONType:
        """
        Serializes object into a raw json.

        Returns:
            raw json
        """
        return asdict(self)

    @classmethod
    def deserialize(cls, raw: JSONType) -> Any:
        """
        Parses raw json into an object of the class type.

        Returns:
            parsed object.
        """
        return dacite.from_dict(cls, raw)


SerializableJSONType = Union[JSONType, Serializable]


def serialize_json(data: SerializableJSONType) -> JSONType:
    """
    Serializes Json object that may contain some deserialized Serializable objects.

    Args:
        data: Json

    Returns:
        Raw json (fully serialized)
    """
    if isinstance(data, dict):
        return {k: serialize_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [serialize_json(v) for v in data]
    if isinstance(data, Serializable):
        return data.serialize()

    return data