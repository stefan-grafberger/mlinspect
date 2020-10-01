"""
The Interface for the Constraints
"""
from __future__ import annotations

from mlinspect.checks.constraint import Constraint, ConstraintStatus, ConstraintResult
from mlinspect.instrumentation.dag_node import OperatorType
from mlinspect.instrumentation.inspection_result import InspectionResult

ILLEGAL_FEATURES = {"race", "gender", "age"}


class NoIllegalFeaturesConstraint(Constraint):
    """
    Does the model get sensitive attributes like race as feature?
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, additional_illegal_feature_names=None):
        if additional_illegal_feature_names is None:
            additional_illegal_feature_names = []
        self.additional_illegal_feature_names = additional_illegal_feature_names

    required_inspection = []

    def evaluate(self, inspection_result: InspectionResult) -> ConstraintResult:
        """Evaluate the constraint"""
        # TODO: Make this robust and add extensive testing
        dag = inspection_result.dag
        train_data = [node for node in dag.nodes if node.operator_type == OperatorType.ESTIMATOR][0]
        used_columns = self.get_used_columns(dag, train_data)
        forbidden_columns = {*ILLEGAL_FEATURES, *self.additional_illegal_feature_names}
        if set(used_columns).intersection(forbidden_columns):
            result = ConstraintResult(self, ConstraintStatus.FAILURE)
        else:
            result = ConstraintResult(self, ConstraintStatus.SUCCESS)
        return result

    def get_used_columns(self, dag, current_node):
        """
        Get the output column of the current dag node. If the current dag node is, e.g., a concatenation,
        check the parents of the current dag node.
        """
        columns = current_node.columns
        if columns is not None and columns != ["array"]:
            result = columns
        else:
            parent_columns = []
            for parent in dag.predecessors(current_node):
                parent_columns.extend(self.get_used_columns(dag, parent))
            result = parent_columns
        return result
