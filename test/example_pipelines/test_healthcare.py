"""
Tests whether the healthcare demo works
"""
import ast
from test.testing_helper_utils import run_and_assert_all_op_outputs_inspected
from example_pipelines import HEALTHCARE_PY, HEALTHCARE_PNG


def test_py_pipeline_runs():
    """
    Tests whether the pipeline works without instrumentation
    """
    with open(HEALTHCARE_PY) as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    run_and_assert_all_op_outputs_inspected(HEALTHCARE_PY, ["age_group", "race"], HEALTHCARE_PNG)
