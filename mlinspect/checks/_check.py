"""
The Interface for the Checks
"""
from __future__ import annotations

import abc
import dataclasses
from enum import Enum
from typing import Iterable

from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_result import InspectionResult


class CheckStatus(Enum):
    """
    The result of the check
    """
    SUCCESS = "Success"
    FAILURE = "Failure"


@dataclasses.dataclass
class CheckResult:
    """
    Does this check cause an error or a warning if it fails?
    """
    check: Check
    status: CheckStatus
    description: str or None


class Check(metaclass=abc.ABCMeta):
    """
    Checks like no_bias_introduced
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    @property
    def check_id(self):
        """The id of the Check"""
        return None

    @property
    @abc.abstractmethod
    def required_inspections(self) -> Iterable[Inspection]:
        """Inspections required to evaluate this check"""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        raise NotImplementedError

    def __eq__(self, other):
        """Checks must implement equals"""
        return (isinstance(other, self.__class__) and
                self.check_id == other.check_id)

    def __hash__(self):
        """Checks must be hashable"""
        return hash((self.__class__.__name__, self.check_id))

    def __repr__(self):
        """Checks must have a str representation"""
        return "{}({})".format(self.__class__.__name__, self.check_id or "")
