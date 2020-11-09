"""
Tests whether the utils work
"""
import ast
from pathlib import Path

import pytest

from mlinspect.utils._utils import get_project_root, simplify_ast_call_nodes


def test_get_project_root():
    """
    Tests whether get_project_root works
    """
    assert get_project_root() == Path(__file__).parent.parent.parent


def test_simplify_ast_call_nodes_throws_if_not_call():
    """
    Tests whether simplify_ast_call_nodes throws errors for non-call nodes
    """
    constant_ast = ast.Constant(n=2, kind=None)
    with pytest.raises(Exception):
        simplify_ast_call_nodes(constant_ast)


def test_simplify_ast_call_nodes():
    """
    Tests whether simplify_ast_call_nodes works
    """
    call_ast = ast.Call(n=2, kind=None, lineno=5, col_offset=10)
    assert simplify_ast_call_nodes(call_ast) == (5, 10)
