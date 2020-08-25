"""
Preprocess Sklearn WIR nodes to enable DAG extraction
"""

from ..instrumentation.dag_node import DagNodeIdentifier
from ..utils import traverse_graph_and_process_nodes


class SklearnDagPostprocessor:
    """
    Preprocess Sklearn WIR nodes to enable DAG extraction
    """
    # pylint: disable=too-few-public-methods

    @staticmethod
    def postprocess_dag(graph, wir_post_processing_map):
        """Associate DAG nodes with the correct inspection output from sklearn pipelines"""
        new_code_references = {}

        def process_node(node, _):
            dag_node_identifier = DagNodeIdentifier(node.operator_type, node.code_reference, node.description)
            if node.code_reference not in wir_post_processing_map:
                return
            if node.module in {('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                               ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                               ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                               ('sklearn.demo.healthcare.demo_utils', 'MyW2VTransformer', 'Pipeline'),
                               ('sklearn.impute._base', 'SimpleImputer', 'Pipeline')
                               }:
                annotations_for_all_associated_dag_nodes = wir_post_processing_map[node.code_reference]
                annotation = annotations_for_all_associated_dag_nodes[node.description]
                new_code_references[dag_node_identifier] = annotation
            elif node.module == ('sklearn.pipeline', 'fit', 'Train Data'):
                annotations_for_all_associated_dag_nodes = wir_post_processing_map[node.code_reference]
                annotations_x = annotations_for_all_associated_dag_nodes['fit X']
                new_code_references[dag_node_identifier] = annotations_x
            elif node.module == ('sklearn.pipeline', 'fit', 'Train Labels'):
                annotations_for_all_associated_dag_nodes = wir_post_processing_map[node.code_reference]
                annotations_y = annotations_for_all_associated_dag_nodes['fit y']
                new_code_references[dag_node_identifier] = annotations_y
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'):
                annotations_for_all_associated_dag_nodes = wir_post_processing_map[node.code_reference]
                annotations = annotations_for_all_associated_dag_nodes['concat']
                new_code_references[dag_node_identifier] = annotations
            elif node.module in {('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'),
                                 ('sklearn.tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier',
                                  'Pipeline')}:
                annotations_for_all_associated_dag_nodes = wir_post_processing_map[node.code_reference]
                annotations = annotations_for_all_associated_dag_nodes['fit']
                new_code_references[dag_node_identifier] = annotations
            elif node.module == ('sklearn.pipeline', 'Pipeline'):
                pass  # Nothing to do here

        traverse_graph_and_process_nodes(graph, process_node)
        return new_code_references
