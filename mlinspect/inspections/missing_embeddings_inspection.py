"""
A simple example inspection
"""
from typing import Iterable

from mlinspect.inspections.inspection import Inspection
from mlinspect.inspections.inspection_input import InspectionInputUnaryOperator


class MissingEmbeddingInspection(Inspection):
    """
    A simple example inspection
    """

    def __init__(self, example_threshold=10):
        self._is_embedding_operator = False
        self._missing_embedding_count = 0
        self._missing_embeddings_examples = []
        self.example_threshold = example_threshold

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        # pylint: disable=too-many-branches, too-many-statement
        if isinstance(inspection_input, InspectionInputUnaryOperator) and \
                inspection_input.operator_context.function_info == ('demo.healthcare.demo_utils', 'fit_transform'):
            # TODO: Are there existing word embedding transformers for sklearn we can use this for?
            self._is_embedding_operator = True
            for row in inspection_input.row_iterator:
                # Count missing embeddings
                embedding_array = row.output[0]
                is_zero_vector = not embedding_array.any()
                if is_zero_vector:
                    self._missing_embedding_count += 1
                    if len(self._missing_embeddings_examples) < self.example_threshold:
                        self._missing_embeddings_examples.append(row.input[0])
            yield None
        else:
            for _ in inspection_input.row_iterator:
                yield None

    def get_operator_annotation_after_visit(self) -> any:
        if self._is_embedding_operator:
            assert self._missing_embedding_count is not None  # May only be called after the operator visit is finished
            result = {"missing_embedding_count": self._missing_embedding_count,
                      "missing_embeddings_examples": self._missing_embeddings_examples}
            self._missing_embedding_count = None
            self._is_embedding_operator = False
            self._missing_embeddings_examples = []
            return result
        return None

    @property
    def inspection_id(self):
        return self.example_threshold
