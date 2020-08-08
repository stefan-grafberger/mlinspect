"""
Extract a DAG from the WIR (Workflow Intermediate Representation)
"""
import networkx

from mlinspect.instrumentation.dag_node import DagNode, OperatorType
from mlinspect.instrumentation.sklearn_wir_preprocessor import SklearnWirPreprocessor
from mlinspect.utils import traverse_graph_and_process_nodes


class WirToDagTransformer:
    """
    Extract DAG from the WIR (Workflow Intermediate Representation)
    """

    OPERATOR_MAP = {
        ('pandas.io.parsers', 'read_csv'): OperatorType.DATA_SOURCE,
        ('pandas.core.frame', 'dropna'): OperatorType.SELECTION,
        ('pandas.core.frame', '__getitem__'): OperatorType.PROJECTION,
        ('mlinspect.instrumentation.backends.pandas_backend_frame_wrapper', '__getitem__'): OperatorType.PROJECTION,
        ('sklearn.preprocessing._label', 'label_binarize'): OperatorType.PROJECTION_MODIFY,
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'): OperatorType.PROJECTION,
        ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'): OperatorType.TRANSFORMER,
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'): OperatorType.CONCATENATION,
        ('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'): OperatorType.ESTIMATOR,
        ('sklearn.pipeline', 'fit', 'Pipeline'): OperatorType.FIT,
        ('sklearn.pipeline', 'fit', 'Train Data'): OperatorType.TRAIN_DATA,
        ('sklearn.pipeline', 'fit', 'Train Labels'): OperatorType.TRAIN_LABELS
    }

    @staticmethod
    def extract_dag(wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Extract the final DAG
        """
        preprocessed_wir = SklearnWirPreprocessor().sklearn_wir_preprocessing(wir)
        cleaned_wir = WirToDagTransformer.remove_all_nodes_but_calls_and_subscripts(preprocessed_wir)
        dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

        return dag

    @staticmethod
    def remove_all_nodes_but_calls_and_subscripts(graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Removes all nodes that can not be a operator we might care about
        """
        def process_node(node, _):
            if node.operation in {"Import", "Constant"}:
                graph.remove_node(node)
            elif node.operation in {"Assign", "Keyword", "List", "Tuple"}:
                parents = list(graph.predecessors(node))
                children = list(graph.successors(node))
                for parent_node in parents:
                    for child_node in children:
                        graph.add_edge(parent_node, child_node)
                graph.remove_node(node)
            elif node.operation in {"Call", "Subscript"}:
                pass
            else:
                print("Unknown WIR Node Type: {}".format(node))
                assert False

        traverse_graph_and_process_nodes(graph, process_node)

        # By modifying edges, most labels are lost, so we remove the rest of them too
        for (_, _, edge_attributes) in graph.edges(data=True):
            edge_attributes.clear()

        return graph

    @staticmethod
    def remove_all_non_operators_and_update_names(graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Removes all nodes that can not be a operator we might care about
        """
        def process_node(node, processed_nodes):
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))
            if node.module in WirToDagTransformer.OPERATOR_MAP:
                new_dag_vertex = DagNode(node.node_id, WirToDagTransformer.OPERATOR_MAP[node.module],
                                         node.code_reference, node.module, node.dag_operator_description)
                for parent in parents:
                    graph.add_edge(parent, new_dag_vertex)
                for child in children:
                    graph.add_edge(new_dag_vertex, child)
                graph.remove_node(node)
                processed_nodes.add(new_dag_vertex)
            else:
                for parent in parents:
                    for child in children:
                        graph.add_edge(parent, child)
                graph.remove_node(node)

        traverse_graph_and_process_nodes(graph, process_node)

        return graph
