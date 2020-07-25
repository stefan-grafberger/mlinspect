"""
Some useful utils for the project
"""
import ast
from pathlib import Path


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
