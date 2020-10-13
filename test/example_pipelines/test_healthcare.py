"""
Tests whether the healthcare demo works
"""
import ast

from example_pipelines.pipelines import HEALTHCARE_PY


def test_py_pipeline_runs():
    """
    Tests whether the pipeline works without instrumentation
    """
    with open(HEALTHCARE_PY) as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))
