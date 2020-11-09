"""
The NoMissingEmbeddings Check
"""
import collections
import dataclasses
from typing import Iterable, OrderedDict

from demo.feature_overview.missing_embeddings import MissingEmbeddings, MissingEmbeddingsInfo
from mlinspect import DagNode
from mlinspect.checks import Check, CheckStatus, CheckResult
from mlinspect.inspections import Inspection, InspectionResult


ILLEGAL_FEATURES = {"race", "gender", "age"}


@dataclasses.dataclass
class NoMissingEmbeddingsResult(CheckResult):
    """
    Does the pipeline use illegal features?
    """
    dag_node_to_missing_embeddings: OrderedDict[DagNode, MissingEmbeddingsInfo]


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
        return [MissingEmbeddings(self.example_threshold)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        dag = inspection_result.dag
        embedding_inspection_result = inspection_result.inspection_to_annotations[
            MissingEmbeddings(self.example_threshold)]
        dag_node_to_missing_embeddings = collections.OrderedDict()
        for dag_node in dag.nodes:
            if dag_node in embedding_inspection_result and embedding_inspection_result[dag_node] is not None:
                missing_embedding_info = embedding_inspection_result[dag_node]
                if missing_embedding_info.missing_embedding_count > 0:
                    dag_node_to_missing_embeddings[dag_node] = missing_embedding_info
        if dag_node_to_missing_embeddings:
            description = "Missing embeddings were found!"
            result = NoMissingEmbeddingsResult(self, CheckStatus.FAILURE, description, dag_node_to_missing_embeddings)
        else:
            result = NoMissingEmbeddingsResult(self, CheckStatus.SUCCESS, None, collections.OrderedDict())
        return result
