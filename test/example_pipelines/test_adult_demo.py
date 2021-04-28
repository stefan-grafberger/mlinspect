"""
Tests whether the adult_easy test pipeline works
"""
import ast

from test.testing_helper_utils import run_and_assert_all_op_outputs_inspected
from example_pipelines._pipelines import ADULT_DEMO_PY, ADULT_DEMO_PNG


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    with open(ADULT_DEMO_PY) as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    run_and_assert_all_op_outputs_inspected(ADULT_DEMO_PY, ["race"], ADULT_DEMO_PNG)
