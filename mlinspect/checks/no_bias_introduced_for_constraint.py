"""
The Interface for the Constraints
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, List, Tuple

from werkzeug.routing import Map

from mlinspect.checks.constraint import Constraint, ConstraintStatus, ConstraintResult
from mlinspect.inspections.histogram_inspection import HistogramInspection
from mlinspect.inspections.inspection import Inspection
from mlinspect.instrumentation.dag_node import OperatorType
from mlinspect.instrumentation.inspection_result import InspectionResult


@dataclasses.dataclass
class NoBiasIntroducedforConstraintResult(ConstraintResult):
    """
    Does this check cause an error or a warning if it fails?
    """
    histograms_after_before: List[Tuple[Map]]


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
        dag = inspection_result.dag
        histograms = inspection_result.inspection_to_annotations[HistogramInspection(self.sensitive_columns)]
        relevant_nodes = [node for node in dag.nodes if node.operator_type in {OperatorType.JOIN,
                                                                               OperatorType.SELECTION} or
                          node.module == ('sklearn.impute._base', 'fit_transform')]
        result = ConstraintStatus.SUCCESS
        histograms_after_before = []
        for node in relevant_nodes:
            parents = dag.predecessors(node)
            # TODO
            print(histograms)
            print(parents)
        return NoBiasIntroducedforConstraintResult(self, result, histograms_after_before)
