"""
Extract a DAG from the WIR (Workflow Intermediate Representation)
"""
import networkx

from ..backends.all_backends import get_all_backends
from ..instrumentation.dag_node import DagNode
from ..instrumentation.wir_extractor import WirExtractor
from ..utils import traverse_graph_and_process_nodes


class WirToDagTransformer:
    """
    Extract DAG from the WIR (Workflow Intermediate Representation)
    """

    OPERATOR_MAP = dict(backend_item for backend in get_all_backends() for backend_item in backend.operator_map.items())

    @staticmethod
    def extract_dag(wir: networkx.DiGraph) -> networkx.DiGraph:
        """
        Extract the final DAG
        """
        cleaned_wir = WirToDagTransformer.remove_all_nodes_but_calls_and_subscripts(wir)
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
            elif node.operation in {"Call", "Subscript", "Subscript-Assign"}:
                pass
            elif node == WirExtractor.NOT_FOUND_WIR:
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
                graph.remove_node(node)
                graph.add_node(new_dag_vertex)
                for parent in parents:
                    graph.add_edge(parent, new_dag_vertex)
                for child in children:
                    graph.add_edge(new_dag_vertex, child)

                processed_nodes.add(new_dag_vertex)
            else:
                for parent in parents:
                    for child in children:
                        graph.add_edge(parent, child)
                graph.remove_node(node)

        traverse_graph_and_process_nodes(graph, process_node)

        return graph
