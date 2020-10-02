"""
The Interface for the Constraints
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Dict

from demo.healthcare.missing_embeddings_inspection import MissingEmbeddingInspection, MissingEmbeddingsInfo
from mlinspect.checks.constraint import Constraint, ConstraintStatus, ConstraintResult
from mlinspect.inspections.inspection import Inspection
from mlinspect.instrumentation.dag_node import DagNode
from mlinspect.instrumentation.inspection_result import InspectionResult

ILLEGAL_FEATURES = {"race", "gender", "age"}


@dataclasses.dataclass
class NoMissingEmbeddingsConstraintResult(ConstraintResult):
    """
    Does the pipeline use illegal features?
    """
    dag_node_to_missing_embeddings: Dict[DagNode, MissingEmbeddingsInfo]


class NoMissingEmbeddingsConstraint(Constraint):
    """
    Does the model get sensitive attributes like race as feature?
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, example_threshold=10):
        self.example_threshold = example_threshold

    @property
    def constraint_id(self):
        """The id of the Constraints"""
        return self.example_threshold

    @property
    def required_inspection(self) -> Iterable[Inspection]:
        """The id of the constraint"""
        return [MissingEmbeddingInspection(self.example_threshold)]

    def evaluate(self, inspection_result: InspectionResult) -> ConstraintResult:
        """Evaluate the constraint"""
        dag = inspection_result.dag
        embedding_inspection_result = inspection_result.inspection_to_annotations[
            MissingEmbeddingInspection(self.example_threshold)]
        dag_node_to_missing_embeddings = {}
        for dag_node in dag.nodes:
            if dag_node in embedding_inspection_result and embedding_inspection_result[dag_node] is not None:
                missing_embedding_info = embedding_inspection_result[dag_node]
                if missing_embedding_info.missing_embedding_count > 0:
                    dag_node_to_missing_embeddings[dag_node] = missing_embedding_info
        if dag_node_to_missing_embeddings:
            result = NoMissingEmbeddingsConstraintResult(self, ConstraintStatus.FAILURE, dag_node_to_missing_embeddings)
        else:
            result = NoMissingEmbeddingsConstraintResult(self, ConstraintStatus.SUCCESS, {})
        return result
