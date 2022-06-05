
from abc import ABCMeta, abstractmethod

from tensorflow import keras


class Model(metaclass=ABCMeta):
    @abstractmethod
    def get_default_param(self):
        pass

    @abstractmethod
    def build(self) -> keras.Model:
        pass
