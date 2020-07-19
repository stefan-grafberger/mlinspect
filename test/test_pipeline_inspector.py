"""
Tests whether the fluent API works
"""
import os
from mlinspect.utils import get_project_root
from mlinspect.pipeline_inspector import PipelineInspector

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def test_py_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    result = PipelineInspector\
        .on_python_pipeline(FILE_PY)\
        .add_analyzer("test")\
        .execute()
    assert result == "test"


def test_nb_pipeline_runs():
    """
    Tests whether the .py version of the pipeline works
    """
    result = PipelineInspector\
        .on_jupyter_pipeline(FILE_NB)\
        .add_analyzer("test")\
        .execute()
    assert result == "test"
