"""
Tests whether the adult_easy test pipeline works
"""
import ast
from mlinspect.testing._testing_helper_utils import run_and_assert_all_op_outputs_inspected
from example_pipelines import COMPAS_PY, COMPAS_PNG


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    with open(COMPAS_PY) as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(COMPAS_PY, ['sex', 'race'], COMPAS_PNG)
    assert len(dag) == 39
