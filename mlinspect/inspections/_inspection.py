"""
The Interface for the Inspection
"""
import abc
from typing import Union, Iterable

from mlinspect.inspections._inspection_input import InspectionInputDataSource, \
    InspectionInputUnaryOperator, InspectionInputNAryOperator, InspectionInputSinkOperator


class Inspection(metaclass=abc.ABCMeta):
    """
    The Interface for the Inspections
    """

    @property
    def inspection_id(self):
        """The id of the inspection"""
        return None

    @abc.abstractmethod
    def visit_operator(self, inspection_input: Union[InspectionInputDataSource, InspectionInputUnaryOperator,
                                                     InspectionInputNAryOperator, InspectionInputSinkOperator])\
            -> Iterable[any]:
        """Visit an operator in the DAG"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_operator_annotation_after_visit(self) -> any:
        """Get the output to be included in the DAG"""
        raise NotImplementedError

    def __eq__(self, other):
        """Inspections must implement equals"""
        return (isinstance(other, self.__class__) and
                self.inspection_id == other.inspection_id)

    def __hash__(self):
        """Inspections must be hashable"""
        return hash((self.__class__.__name__, self.inspection_id))

    def __repr__(self):
        """Inspections must have a str representation"""
        return "{}({})".format(self.__class__.__name__, self.inspection_id)
