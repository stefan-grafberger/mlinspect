"""
Extract a DAG from the WIR (Workflow Intermediate Representation)
"""
import networkx

from mlinspect.instrumentation.dag_vertex import DagVertex
from mlinspect.utils import traverse_graph_and_process_nodes


class WirToDagTransformer:
    """
    Extract DAG from the WIR (Workflow Intermediate Representation)
    """

    OPERATOR_MAP = {
        ('pandas.io.parsers', 'read_csv'): "Data Source",
        ('pandas.core.frame', 'dropna'): "Selection",
        ('pandas.core.frame', '__getitem__'): "Projection",
        ('sklearn.preprocessing._label', 'label_binarize'): "Projection (Modify)",
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'): "Projection",
        ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'): "Transformer",
        ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'): "Transformer",
        ('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'): "Concatenation",
        ('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'): "Estimator",
        ('sklearn.pipeline', 'fit', 'Pipeline'): "Fit Transformers and Estimators"
    }

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
                new_dag_vertex = DagVertex(node.node_id, WirToDagTransformer.OPERATOR_MAP[node.module], node.lineno,
                                           node.col_offset, node.module)
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

    def get_parent_operator_identifier_for_operator(self, lineno, col_offset):
        """
        If we store the annotations of analyzers in a map until the next operator needs it, we need some
        way to identify them and know when we can delete them. When e.g., there is a raw_data.dropna, it needs
        to know that the previous operator was the pd.read_csv. Then it can load the annotations from a map.
        Afterwards, the annotations need to be deleted from the map to avoid unnecessary overhead.
        While we might store the annotations directly in the data frame in the case of pandas, in the case of
        sklearn that is probably not easily possible.
        """
        raise NotImplementedError
