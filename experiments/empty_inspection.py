"""
A simple empty inspection
"""
from typing import Iterable

from mlinspect.inspections._inspection import Inspection


class EmptyInspection(Inspection):
    """
    An empty inspection for performance experiments
    """

    def __init__(self, inspection_id):
        self._id = inspection_id

    @property
    def inspection_id(self):
        return self._id

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        for _ in inspection_input.row_iterator:
            yield None

    def get_operator_annotation_after_visit(self) -> any:
        return None
