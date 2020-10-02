"""
The Interface for the Constraints
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Dict

from mlinspect.checks.constraint import Constraint, ConstraintStatus, ConstraintResult
from mlinspect.inspections.histogram_inspection import HistogramInspection
from mlinspect.inspections.inspection import Inspection
from mlinspect.instrumentation.dag_node import OperatorType, DagNode
from mlinspect.instrumentation.inspection_result import InspectionResult


@dataclasses.dataclass(eq=True, frozen=True)
class BiasDistributionChange:
    """
    Did the histogram change too much for one given operation?
    """
    acceptable_change: bool
    max_relative_change: float
    before_map: Dict[str, int]
    after_map: Dict[str, int]


@dataclasses.dataclass
class NoBiasIntroducedForConstraintResult(ConstraintResult):
    """
    Did the histogram change too much for some operations?
    """
    bias_distribution_change: Dict[DagNode, Dict[str, BiasDistributionChange]]


class NoBiasIntroducedForConstraint(Constraint):
    """
    Does the user pipeline introduce bias because of operators like joins and selects?
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, sensitive_columns, max_relative_change=0.3):
        self.sensitive_columns = sensitive_columns
        self.max_relative_change = max_relative_change

    @property
    def constraint_id(self):
        """The id of the Constraints"""
        return tuple(self.sensitive_columns), self.max_relative_change

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
                          (node.module == ('sklearn.impute._base', 'SimpleImputer', 'Pipeline') and
                           node.columns[0] in self.sensitive_columns)]
        constraint_result = ConstraintStatus.SUCCESS
        bias_distribution_change = {}
        for node in relevant_nodes:
            parents = list(dag.predecessors(node))
            column_results = {}
            for column in self.sensitive_columns:
                constraint_result = self.get_histograms_for_node_and_column(column, column_results, constraint_result,
                                                                            histograms, node, parents)
            bias_distribution_change[node] = column_results
        return NoBiasIntroducedForConstraintResult(self, constraint_result, bias_distribution_change)

    def get_histograms_for_node_and_column(self, column, column_results, constraint_result, histograms, node, parents):
        """
        Compute histograms for a dag node like a join and a concrete sensitive column like race
        """
        # pylint: disable=too-many-locals, too-many-arguments
        after_map = histograms[node][column]
        before_map = {}
        for parent in parents:
            parent_histogram = histograms[parent][column]
            before_map = {**before_map, **parent_histogram}
        removed_groups = [group_key for group_key in after_map.keys() if group_key not in before_map and
                          group_key != "None"]
        if removed_groups:
            max_abs_change = 1.0
        else:
            before_count_all = sum(before_map.values())
            after_count_all = sum(after_map.values())
            abs_relative_changes = []
            for group_key in after_map:
                after_count = after_map[group_key]
                after_ratio = after_count / after_count_all
                before_count = before_map.get(group_key, 0)
                before_ratio = before_count / before_count_all or 0
                relative_change = (after_ratio - before_ratio) / after_ratio
                abs_relative_changes.append(abs(relative_change))
            max_abs_change = max(abs_relative_changes)
        all_changes_acceptable = max_abs_change <= self.max_relative_change
        if not all_changes_acceptable:
            constraint_result = ConstraintStatus.FAILURE
        column_results[column] = BiasDistributionChange(all_changes_acceptable, max_abs_change,
                                                        before_map, after_map)
        return constraint_result
