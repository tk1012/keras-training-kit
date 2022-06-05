from dataclasses import dataclass
from typing import Optional, Tuple

from tensorflow import keras

from .base import Model
from .factory import register


@dataclass
class SimpleDNNParam:
    input_shape: Optional[Tuple[int]] = None
    alpha: float = 1.0
    output: int = 10
    output_activation: Optional[str] = None


@register("simplednn")
class SimpleDNN(Model):
    def get_default_param(self):
        return SimpleDNNParam()

    def build(self, input_shape: Optional[Tuple[int]], alpha: float, output: int, output_activation: Optional[str]) -> keras.Model:
        return keras.Sequential(
            [
                keras.layers.Flatten(input_shape=input_shape),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(output, activation=output_activation),
            ]
        )
