"""
The Interface for the Constraints
"""
import dataclasses
from enum import Enum


class ConstraintStatus(Enum):
    """
    The result of the check
    """
    SUCCESS = "Success"
    FAILURE = "Failure"


class Constraint:
    """
    Constraints like no_bias_introduced
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods
    description = ""


@dataclasses.dataclass
class ConstraintResult(Enum):
    """
    Does this check cause an error or a warning if it fails?
    """
    constraint: Constraint
    status: ConstraintStatus
