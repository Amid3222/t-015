from typing import Type, List

from jedi.third_party.typeshed.stdlib.dataclasses import dataclass


@dataclass
class ModelClassInfo:
    """Информация о классе модели"""
    model_name: str
    model_class: Type
    params_keys: List[str]


@dataclass
class ModelObj:
    """Информация о объекте модели"""
    model_name: str
    param_ver: str
    model: Type
