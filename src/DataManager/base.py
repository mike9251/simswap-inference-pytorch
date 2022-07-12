from abc import ABC, abstractmethod
import numpy as np


class BaseDataManager(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get(self) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, img: np.ndarray) -> None:
        pass
