"""
Tests whether the utils work
"""
import ast
from inspect import cleandoc
from pathlib import Path
import pytest
from mlinspect.utils import get_project_root, simplify_ast_call_nodes


def test_get_project_root():
    """
    Tests whether get_project_root works
    """
    assert get_project_root() == Path(__file__).parent.parent


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


def get_test_df_creation_str(size):
    """
    Get a complete code str that creates a DF with random value
    """
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint

        array = randint(0,100,size=({}, 4))
        df = pd.DataFrame(array, columns=['A', 'B', 'C', 'D'])
        """.format(size))
    return test_code


def get_test_projection_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        test = df[['A']]
        """)
    return test_code
