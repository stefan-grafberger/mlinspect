"""
Tests whether the adult_easy test pipeline works
"""
import ast
import os
import nbformat
from nbconvert import PythonExporter
from mlinspect.utils import get_project_root

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "healthcare.py")


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    with open(FILE_PY) as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))
