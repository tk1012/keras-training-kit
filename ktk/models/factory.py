import builtins
from logging import getLogger
from typing import Callable, List

from .base import Model

logger = getLogger(__name__)


_registry = {}


def register(name: str) -> Callable:
    def inner_wrapper(wrapped_class: Model) -> Model:
        if name in _registry:
            logger.warning(f"Model for {name} already exists. Will replace it")
        _registry[name] = wrapped_class
        print(_registry.keys())
        return wrapped_class

    return inner_wrapper


def get(name: str, **kwargs) -> Model:
    if name not in _registry:
        raise Exception(f"Model {name} not exists in the registry")
    model_class = _registry[name]
    obj = model_class(**kwargs)
    return obj


def list() -> List[str]:
    return builtins.list(_registry.keys())
