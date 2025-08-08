from abc import ABC, abstractmethod


class AbstractTask(ABC):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_description(self):
        pass
