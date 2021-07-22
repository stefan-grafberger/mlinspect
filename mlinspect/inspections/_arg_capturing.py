"""
A simple inspection to capture important function call arguments like estimator hyperparameters
"""
from typing import Iterable

from ._inspection import Inspection


class ArgumentCapturing(Inspection):
    """
    A simple inspection to capture important function call arguments like estimator hyperparameters
    """

    def __init__(self):
        self._captured_arguments = None

    @property
    def inspection_id(self):
        return None

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        self._captured_arguments = inspection_input.non_data_function_args

        for _ in inspection_input.row_iterator:
            yield None

    def get_operator_annotation_after_visit(self) -> any:
        captured_args = self._captured_arguments
        self._captured_arguments = None
        return captured_args
