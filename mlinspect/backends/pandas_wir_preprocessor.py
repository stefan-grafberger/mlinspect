"""
Preprocess pandas WIR nodes to enable DAG extraction
"""

from ..instrumentation.wir_node import WirNode
from ..utils import traverse_graph_and_process_nodes, get_sorted_node_parents


class PandasWirPreprocessor:
    """
    Preprocess Pandas WIR nodes to enable DAG extraction
    """
    # pylint: disable=too-few-public-methods

    @staticmethod
    def preprocess_wir(graph, wir_post_processing_map):
        """Associate DAG nodes with the correct inspection output from sklearn pipelines"""
        def process_node(node, _):
            if node.module == ('pandas.core.frame', '__getitem__'):
                operator_type = wir_post_processing_map[node.code_reference]
                new_module = (node.module[0], node.module[1], operator_type)

                new_node = WirNode(node.node_id, node.name, node.operation, node.code_reference, new_module,
                                   node.dag_operator_description)

                parents = get_sorted_node_parents(graph, node)
                graph.add_edge(parents[0], new_node)

                children = list(graph.successors(node))
                for child in children:
                    child_edge_data = graph.get_edge_data(node, child)
                    graph.add_edge(new_node, child, **child_edge_data)

                graph.remove_node(node)

        traverse_graph_and_process_nodes(graph, process_node)
