"""
Get all available backends
"""
from typing import List

from mlinspect.instrumentation.backends.backend import Backend
from mlinspect.instrumentation.backends.pandas_backend import PandasBackend
from mlinspect.instrumentation.backends.sklearn_backend import SklearnBackend


def get_all_backends() -> List[Backend]:
    """Get the list of all currently available backends"""
    return [PandasBackend(), SklearnBackend()]
