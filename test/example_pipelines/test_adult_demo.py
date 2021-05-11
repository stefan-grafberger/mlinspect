"""
Tests whether the adult_easy test pipeline works
"""
import ast
import os

import matplotlib
from importnb import Notebook

from mlinspect.utils import get_project_root
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


ADULT_DEMO_NB = os.path.join(str(get_project_root()), "experiments", "user_interviews",
                             "example-testing-adult-pipeline.ipynb")


def test_demo_nb():
    """
    Tests whether the demo notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(ADULT_DEMO_NB)
