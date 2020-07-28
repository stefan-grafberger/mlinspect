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


def traverse_graph_and_process_nodes(graph: networkx.DiGraph, func):
    """
    Traverse the WIR node by node from top to bottom
    """
    current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
    processed_nodes = set()
    while len(current_nodes) != 0:
        node = current_nodes.pop(0)
        processed_nodes.add(node)
        children = list(graph.successors(node))

        # Nodes can have multiple parents, only want to process them once we processed all parents
        for child in children:
            if child not in processed_nodes:
                if processed_nodes.issuperset(graph.predecessors(child)):
                    current_nodes.append(child)

        func(node, processed_nodes)
    return graph
