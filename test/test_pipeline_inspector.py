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
    Tests whether the .py version of the inspector works
    """
    result = PipelineInspector\
        .on_pipeline_from_py_file(FILE_PY)\
        .add_analyzer("test")\
        .execute()
    assert result == "test"


def test_nb_pipeline_runs():
    """
    Tests whether the .ipynb version of the inspector works
    """
    result = PipelineInspector\
        .on_pipeline_from_ipynb_file(FILE_NB)\
        .add_analyzer("test")\
        .execute()
    assert result == "test"


def test_str_pipeline_runs():
    """
    Tests whether the str version of the inspector works
    """
    with open(FILE_PY) as file:
        code = file.read()

        result = PipelineInspector\
            .on_pipeline_from_string(code)\
            .add_analyzer("test")\
            .execute()
        assert result == "test"
