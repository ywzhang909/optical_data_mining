from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
import pandas as pd

T = TypeVar('T')

class InputSource(Generic[T], ABC):
    """Input source interface: reads data and returns type T"""

    def __init__(self, config: Optional[object] = None):
        self.config = config

    @abstractmethod
    def read(self) -> T:
        """Read data from the input source"""
        raise NotImplementedError

    @abstractmethod
    def validate(self, data: T) -> bool:
        """Validate the read data"""
        raise NotImplementedError
