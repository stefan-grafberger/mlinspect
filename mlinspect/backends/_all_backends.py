"""
Get all available backends
"""
from typing import List

from ._backend import Backend
from ._pandas_backend import PandasBackend
from ._sklearn_backend import SklearnBackend


def get_all_backends() -> List[Backend]:
    """Get the list of all currently available backends"""
    return [PandasBackend(), SklearnBackend()]
