"""
The Interface for the Constraints
"""
from __future__ import annotations

import abc
import dataclasses
from enum import Enum
from typing import Iterable

from mlinspect.inspections.inspection import Inspection


class ConstraintStatus(Enum):
    """
    The result of the check
    """
    SUCCESS = "Success"
    FAILURE = "Failure"


@dataclasses.dataclass
class ConstraintResult(Enum):
    """
    Does this check cause an error or a warning if it fails?
    """
    constraint: Constraint
    status: ConstraintStatus


class Constraint(metaclass=abc.ABCMeta):
    """
    Constraints like no_bias_introduced
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    @property
    def constraint_id(self):
        """The id of the inspection"""
        return None

    @property
    @abc.abstractmethod
    def required_inspection(self) -> Iterable[Inspection]:
        """The id of the inspection"""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, inspection_result) -> ConstraintResult:
        """The id of the inspection"""
        raise NotImplementedError

    def __eq__(self, other):
        """Inspections must implement equals"""
        return (isinstance(other, self.__class__) and
                self.constraint_id == other.constraint_id)

    def __hash__(self):
        """Inspections must be hashable"""
        return hash((self.__class__.__name__, self.constraint_id))

    def __repr__(self):
        """Inspections must have a str representation"""
        return "{}({})".format(self.__class__.__name__, self.constraint_id)
