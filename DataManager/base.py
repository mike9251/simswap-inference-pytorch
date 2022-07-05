from abc import ABC, abstractmethod


class BaseDataManager(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get(self):
        pass
