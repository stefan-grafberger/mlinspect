"""
The NoMissingEmbeddings Check
"""
import collections
import dataclasses
from typing import Iterable, Dict

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
        return [MissingEmbeddings(self.example_threshold)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        dag_node_to_missing_embeddings = dict()
        for dag_node, dag_node_inspection_result in inspection_result.dag_node_to_inspection_results.items():
            if MissingEmbeddings(self.example_threshold) in dag_node_inspection_result:
                missing_embedding_info = dag_node_inspection_result[MissingEmbeddings(self.example_threshold)]
                assert missing_embedding_info is None or isinstance(missing_embedding_info, MissingEmbeddingsInfo)
                if missing_embedding_info is not None and missing_embedding_info.missing_embedding_count > 0:
                    dag_node_to_missing_embeddings[dag_node] = missing_embedding_info
        if dag_node_to_missing_embeddings:
            description = "Missing embeddings were found!"
            result = NoMissingEmbeddingsResult(self, CheckStatus.FAILURE, description, dag_node_to_missing_embeddings)
        else:
            result = NoMissingEmbeddingsResult(self, CheckStatus.SUCCESS, None, collections.OrderedDict())
        return result
