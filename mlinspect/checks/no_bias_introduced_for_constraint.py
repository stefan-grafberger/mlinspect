"""
The Interface for the Constraints
"""
from __future__ import annotations

from typing import Iterable

from mlinspect.checks.constraint import Constraint, ConstraintStatus, ConstraintResult
from mlinspect.inspections.histogram_inspection import HistogramInspection
from mlinspect.inspections.inspection import Inspection
from mlinspect.instrumentation.inspection_result import InspectionResult


class NoBiasIntroducedForConstraint(Constraint):
    """
    Does the user pipeline introduce bias because of operators like joins and selects?
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, sensitive_columns):
        self.sensitive_columns = sensitive_columns

    @property
    def required_inspection(self) -> Iterable[Inspection]:
        """The id of the constraint"""
        return [HistogramInspection(self.sensitive_columns)]

    def evaluate(self, inspection_result: InspectionResult) -> ConstraintResult:
        """Evaluate the constraint"""
        # TODO
        return ConstraintResult(self, ConstraintStatus.SUCCESS)
