from typing import Optional, Type, Dict
import pandas as pd
from .input_stage_base import InputStage

class InputStageRegistry:
    _registry: Dict[str, Type[InputStage]] = {}

    @classmethod
    def register(cls, name: str, stage_cls: Type[InputStage]) -> None:
        cls._registry[name] = stage_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[InputStage]]:
        return cls._registry.get(name)

    @classmethod
    def load(cls, name: str) -> Optional[InputStage]:
        cls_type = cls.get(name)
        if cls_type is None:
            return None
        return cls_type()
