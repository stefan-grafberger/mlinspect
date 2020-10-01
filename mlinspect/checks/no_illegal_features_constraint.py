"""
The Interface for the Constraints
"""
from __future__ import annotations

from mlinspect.checks.constraint import Constraint, ConstraintStatus, ConstraintResult
from mlinspect.instrumentation.inspection_result import InspectionResult


class NoIllegalFeaturesConstraint(Constraint):
    """
    Constraints like no_bias_introduced
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, additional_illegal_feature_names):
        self.additional_illegal_feature_names = additional_illegal_feature_names

    required_inspection = []

    def evaluate(self, inspection_result: InspectionResult) -> ConstraintResult:
        """The id of the inspection"""
        # TODO
        return ConstraintResult(self, ConstraintStatus.SUCCESS)
