from io import BytesIO, StringIO

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Type, Union

from src.evaluation.metric import Metric


class Evaluator(ABC):
    @classmethod
    @abstractmethod
    def metrics(cls) -> Tuple[Type[Metric], ...]:
        pass

    @abstractmethod
    def evaluate(self, filepath: Path) -> Tuple[Metric, ...]:
        pass

    @abstractmethod
    def validate_submission(self, io_stream: Union[StringIO, BytesIO]) -> bool:
        pass
