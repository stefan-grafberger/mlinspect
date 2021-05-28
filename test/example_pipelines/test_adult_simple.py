"""
Tests whether the adult_easy test pipeline works
"""
import ast

import nbformat
from nbconvert import PythonExporter

from mlinspect.testing._testing_helper_utils import run_and_assert_all_op_outputs_inspected
from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB, ADULT_SIMPLE_PNG


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    with open(ADULT_SIMPLE_PY) as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_nb_pipeline_runs():
    """
    Tests whether the .ipynb version of the pipeline works
    """
    with open(ADULT_SIMPLE_IPYNB) as file:
        notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
        exporter = PythonExporter()

        code, _ = exporter.from_notebook_node(notebook)
        parsed_ast = ast.parse(code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(ADULT_SIMPLE_PY, ["race"], ADULT_SIMPLE_PNG)
    assert len(dag) == 12
