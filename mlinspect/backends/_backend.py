"""
The Interface for the different instrumentation backends
"""
import abc
import dataclasses
from typing import List, Dict

from mlinspect.inspections import Inspection


@dataclasses.dataclass(frozen=True)
class AnnotatedDfObject:
    """ A dataframe-like object and its annotations """
    result_data: any
    result_annotation: any


@dataclasses.dataclass(frozen=True)
class BackendResult:
    """ The annotated dataframe and the annotations for the current DAG node """
    annotated_dfobject: AnnotatedDfObject
    dag_node_annotation: Dict[Inspection, any]


class Backend(metaclass=abc.ABCMeta):
    """
    The Interface for the different instrumentation backends
    """

    @abc.abstractmethod
    def before_call(self, function_info, operator_context, input_infos: List[AnnotatedDfObject]) \
            -> List[AnnotatedDfObject]:
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def after_call(self, function_info, operator_context, input_infos: List[AnnotatedDfObject], return_value) \
            -> BackendResult:
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError
