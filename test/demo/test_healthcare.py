"""
Tests whether the healthcare demo works
"""
import ast
import os

from importnb import Notebook

from mlinspect.utils import get_project_root

ADULT_EASY_FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


PIPELINE_FILE_PY = os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare.py")
DEMO_NB_FILE = os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare_demo.ipynb")


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    with open(PIPELINE_FILE_PY) as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_demo_nb():
    """
    Tests whether the .py version of the inspector works
    """
    Notebook.load(DEMO_NB_FILE)
