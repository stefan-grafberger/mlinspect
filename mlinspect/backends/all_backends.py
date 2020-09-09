"""
Get all available backends
"""
from typing import List

from .backend import Backend
from .pandas_backend import PandasBackend
from .sklearn_backend import SklearnBackend


def get_all_backends() -> List[Backend]:
    """Get the list of all currently available backends"""
    return [PandasBackend(), SklearnBackend()]
