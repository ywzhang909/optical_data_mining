from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T_co = TypeVar("T_co")


class InputSource(ABC, Generic[T_co]):
    """Input source interface: reads data and returns type T"""

    def __init__(self, config: object | None = None):
        self.config = config

    @abstractmethod
    def read(self) -> T_co:
        """Read data from the input source"""
        raise NotImplementedError

    @abstractmethod
    def validate(self, data: T_co) -> bool:
        """Validate the read data"""
        raise NotImplementedError
