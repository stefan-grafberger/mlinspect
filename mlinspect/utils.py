"""
Some useful utils for the project
"""
import ast
from pathlib import Path

import networkx


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


# Apparently python AST nodes have no equals, so we need some workarounds to identify ast nodes
def simplify_ast_call_nodes(node: ast):
    """
    Apparently python AST nodes have no equals.
    """
    if not isinstance(node, ast.Call):
        assert False

    id_tuple = (node.lineno, node.col_offset)
    return id_tuple


def get_sorted_node_parents(graph, node_with_parents):
    """
    Get the parent nodes of a WIR node sorted by argument index.
    """
    node_parents = list(graph.predecessors(node_with_parents))
    node_parents_with_arg_index = [(node_parent, graph.get_edge_data(node_parent, node_with_parents))
                                   for node_parent in node_parents]
    sorted_node_parents_with_arg_index = sorted(node_parents_with_arg_index, key=lambda x: x[1]['arg_index'])
    sorted_node_parents = [node_parent[0] for node_parent in sorted_node_parents_with_arg_index]
    return sorted_node_parents


def traverse_graph_and_process_nodes(graph: networkx.DiGraph, func, start_nodes=None, end_node=None, child_filter=None):
    """
    Traverse the WIR node by node from top to bottom
    """
    if not start_nodes:
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
    else:
        current_nodes = start_nodes
    processed_nodes = set()
    while len(current_nodes) != 0:
        node = current_nodes.pop(0)
        processed_nodes.add(node)
        children = list(graph.successors(node))
        if child_filter:
            children = [child for child in children if child_filter(child)]

        # Nodes can have multiple parents, only want to process them once we processed all parents
        if not end_node or node != end_node:
            for child in children:
                if child not in processed_nodes:
                    predecessors = graph.predecessors(child)
                    if child_filter or processed_nodes.issuperset(predecessors):
                        current_nodes.append(child)

        func(node, processed_nodes)
    return graph
