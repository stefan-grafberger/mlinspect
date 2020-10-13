"""
Tests whether the adult_easy test pipeline works
"""
import ast

from example_pipelines.pipelines import COMPAS_PY


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    with open(COMPAS_PY) as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))
