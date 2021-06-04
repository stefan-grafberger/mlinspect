"""
The Interface for the Constraints
"""
from __future__ import annotations

import dataclasses
from typing import List, Iterable

from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import OperatorType
from mlinspect.inspections._inspection_result import InspectionResult

ILLEGAL_FEATURES = {"race", "gender", "age"}


@dataclasses.dataclass
class NoIllegalFeaturesResult(CheckResult):
    """
    Does the pipeline use illegal features?
    """
    illegal_features: List[str]


class NoIllegalFeatures(Check):
    """
    Does the model get sensitive attributes like race as feature?
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, additional_illegal_feature_names=None):
        if additional_illegal_feature_names is None:
            additional_illegal_feature_names = []
        self.additional_illegal_feature_names = additional_illegal_feature_names

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return []

    @property
    def check_id(self):
        """The id of the Constraints"""
        return tuple(self.additional_illegal_feature_names)

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        # TODO: Make this robust and add extensive testing
        dag = inspection_result.dag
        train_data_nodes = [node for node in dag.nodes if node.operator_info.operator == OperatorType.TRAIN_DATA]
        used_columns = []
        for train_data_node in train_data_nodes:
            used_columns.extend(self.get_used_columns(dag, train_data_node))
        forbidden_columns = {*ILLEGAL_FEATURES, *self.additional_illegal_feature_names}
        used_illegal_columns = list(set(used_columns).intersection(forbidden_columns))
        if used_illegal_columns:
            description = "Used illegal columns: {}".format(used_illegal_columns)
            result = NoIllegalFeaturesResult(self, CheckStatus.FAILURE, description, used_illegal_columns)
        else:
            result = NoIllegalFeaturesResult(self, CheckStatus.SUCCESS, None, [])
        return result

    def get_used_columns(self, dag, current_node):
        """
        Get the output column of the current dag node. If the current dag node is, e.g., a concatenation,
        check the parents of the current dag node.
        """
        columns = current_node.details.columns
        if columns is not None and columns != ["array"]:
            result = columns
        else:
            parent_columns = []
            for parent in dag.predecessors(current_node):
                parent_columns.extend(self.get_used_columns(dag, parent))
            result = parent_columns
        return result
