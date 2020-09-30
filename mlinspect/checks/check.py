"""
The check
"""
import dataclasses
from enum import Enum
from typing import List

from mlinspect.checks.constraint import ConstraintResult


class CheckLevel(Enum):
    """
    Does this check cause an error or a warning if it fails?
    """
    WARNING = "Warning"
    ERROR = "Error"


class CheckStatus(Enum):
    """
    The result of the check
    """
    SUCCESS = "Success"
    WARNING = "Warning"
    ERROR = "Error"


@dataclasses.dataclass
class Check:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    level = CheckLevel.ERROR
    description = ""
    constraints = []  # List[Constraint]

    def no_illegal_features(self, additional_illegal_feature_names=None):
        """
        Ensure no potentially problematic features like 'race' or 'age' are used directly as feature
        """
        no_illegal_feature_constraint = "TODO {}".format(additional_illegal_feature_names)  # FIXME
        self.constraints.append(no_illegal_feature_constraint)
        return self

    def no_bias_introduced_for(self, column_names):
        """
        Ensure no bias is introduced by operators like joins
        """
        no_bias_introduced_constraint = "TODO {}".format(column_names)  # FIXME
        self.constraints.append(no_bias_introduced_constraint)
        return self


@dataclasses.dataclass
class CheckResult:
    """
    The result of a check
    """
    check: Check
    status: CheckStatus
    constraint_results: List[ConstraintResult]
