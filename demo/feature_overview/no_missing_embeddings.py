"""
The Interface for the Constraints
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Dict

from demo.feature_overview.missing_embeddings_inspection import MissingEmbeddingInspection, MissingEmbeddingsInfo
from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._inspection import Inspection
from mlinspect.instrumentation._dag_node import DagNode
from mlinspect.instrumentation._inspection_result import InspectionResult

ILLEGAL_FEATURES = {"race", "gender", "age"}


@dataclasses.dataclass
class NoMissingEmbeddingsResult(CheckResult):
    """
    Does the pipeline use illegal features?
    """
    dag_node_to_missing_embeddings: Dict[DagNode, MissingEmbeddingsInfo]


class NoMissingEmbeddings(Check):
    """
    Does the model get sensitive attributes like race as feature?
    """
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, example_threshold=10):
        self.example_threshold = example_threshold

    @property
    def check_id(self):
        """The id of the Constraints"""
        return self.example_threshold

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The id of the check"""
        return [MissingEmbeddingInspection(self.example_threshold)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
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
            result = NoMissingEmbeddingsResult(self, CheckStatus.FAILURE, dag_node_to_missing_embeddings)
        else:
            result = NoMissingEmbeddingsResult(self, CheckStatus.SUCCESS, {})
        return result
