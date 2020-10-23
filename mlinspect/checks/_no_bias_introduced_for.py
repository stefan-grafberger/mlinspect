"""
The NoBiasIntroducedFor check
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, OrderedDict
import collections

from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._histogram_inspection import HistogramInspection
from mlinspect.inspections._inspection import Inspection
from mlinspect.instrumentation._dag_node import OperatorType, DagNode
from mlinspect.inspections._inspection_result import InspectionResult


@dataclasses.dataclass(eq=True, frozen=True)
class BiasDistributionChange:
    """
    Did the histogram change too much for one given operation?
    """
    dag_node: DagNode
    acceptable_change: bool
    max_relative_change: float
    before_map: OrderedDict[str, int]
    after_map: OrderedDict[str, int]


@dataclasses.dataclass
class NoBiasIntroducedForResult(CheckResult):
    """
    Did the histogram change too much for some operations?
    """
    bias_distribution_change: OrderedDict[DagNode, OrderedDict[str, BiasDistributionChange]]


class NoBiasIntroducedFor(Check):
    """
    Does the user pipeline introduce bias because of operators like joins and selects?
    """

    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, sensitive_columns, max_relative_change=0.3):
        self.sensitive_columns = sensitive_columns
        self.max_relative_change = max_relative_change

    @property
    def check_id(self):
        """The id of the Check"""
        return tuple(self.sensitive_columns), self.max_relative_change

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return [HistogramInspection(self.sensitive_columns)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        dag = inspection_result.dag
        histograms = inspection_result.inspection_to_annotations[HistogramInspection(self.sensitive_columns)]
        relevant_nodes = [node for node in dag.nodes if node.operator_type in {OperatorType.JOIN,
                                                                               OperatorType.SELECTION} or
                          (node.module == ('sklearn.impute._base', 'SimpleImputer', 'Pipeline') and
                           node.columns[0] in self.sensitive_columns)]
        check_status = CheckStatus.SUCCESS
        bias_distribution_change = collections.OrderedDict()
        for node in relevant_nodes:
            parents = list(dag.predecessors(node))
            column_results = collections.OrderedDict()
            for column in self.sensitive_columns:
                check_status = self.get_histograms_for_node_and_column(column, column_results, check_status,
                                                                       histograms, node, parents)
            bias_distribution_change[node] = column_results
        return NoBiasIntroducedForResult(self, check_status, bias_distribution_change)

    def get_histograms_for_node_and_column(self, column, column_results, check_result, histograms, node, parents):
        """
        Compute histograms for a dag node like a join and a concrete sensitive column like race
        """
        # pylint: disable=too-many-locals, too-many-arguments
        after_map = histograms[node][column]
        before_map = collections.OrderedDict()
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
            check_result = CheckStatus.FAILURE
        column_results[column] = BiasDistributionChange(node, all_changes_acceptable, max_abs_change,
                                                        before_map, after_map)
        return check_result
