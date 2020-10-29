"""
Preprocess Sklearn WIR nodes to enable DAG extraction
"""

from ..instrumentation._dag_node import DagNodeIdentifier
from ..utils._utils import traverse_graph_and_process_nodes


class SklearnDagPostprocessor:
    """
    Preprocess Sklearn WIR nodes to enable DAG extraction
    """
    # pylint: disable=too-few-public-methods

    @staticmethod
    def process_dag(graph, annotation_post_processing_map):
        """Associate DAG nodes with the correct inspection output from sklearn pipelines"""
        new_code_references_to_inspection_result = {}
        new_code_references_to_columns = {}

        def process_node(node, _):
            dag_node_identifier = DagNodeIdentifier(node.operator_type, node.code_reference, node.description)
            if node.code_reference not in annotation_post_processing_map:
                return
            if node.module in {('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                               ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                               ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                               ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer', 'Pipeline'),
                               ('sklearn.impute._base', 'SimpleImputer', 'Pipeline'),
                               ('sklearn.preprocessing._discretization', 'KBinsDiscretizer', 'Pipeline'),
                               }:
                annotations_for_all_associated_dag_nodes = annotation_post_processing_map[node.code_reference]
                annotation, columns = annotations_for_all_associated_dag_nodes[node.description]
                new_code_references_to_inspection_result[dag_node_identifier] = annotation
                new_code_references_to_columns[dag_node_identifier] = columns
            elif node.module == ('sklearn.pipeline', 'fit', 'Train Data'):
                annotations_for_all_associated_dag_nodes = annotation_post_processing_map[node.code_reference]
                annotations_x, columns_x = annotations_for_all_associated_dag_nodes['fit X']
                new_code_references_to_inspection_result[dag_node_identifier] = annotations_x
                new_code_references_to_columns[dag_node_identifier] = columns_x
            elif node.module == ('sklearn.pipeline', 'fit', 'Train Labels'):
                annotations_for_all_associated_dag_nodes = annotation_post_processing_map[node.code_reference]
                annotations_y, columns_y = annotations_for_all_associated_dag_nodes['fit y']
                new_code_references_to_inspection_result[dag_node_identifier] = annotations_y
                new_code_references_to_columns[dag_node_identifier] = columns_y
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'):
                annotations_for_all_associated_dag_nodes = annotation_post_processing_map[node.code_reference]
                annotations, columns = annotations_for_all_associated_dag_nodes['concat']
                new_code_references_to_inspection_result[dag_node_identifier] = annotations
                new_code_references_to_columns[dag_node_identifier] = columns
            elif node.module in {('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'),
                                 ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier',
                                  'Pipeline'),
                                 ('sklearn.linear_model._logistic', 'LogisticRegression', 'Pipeline')}:
                annotations_for_all_associated_dag_nodes = annotation_post_processing_map[node.code_reference]
                annotations, columns = annotations_for_all_associated_dag_nodes['fit']
                new_code_references_to_inspection_result[dag_node_identifier] = annotations
                new_code_references_to_columns[dag_node_identifier] = columns
            elif node.module == ('sklearn.pipeline', 'Pipeline'):
                pass  # Nothing to do here

        traverse_graph_and_process_nodes(graph, process_node)
        return new_code_references_to_inspection_result, new_code_references_to_columns
