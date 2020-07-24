"""
Tests whether the adult_easy test pipeline works
"""
import os
from mlinspect.utils import get_project_root
from mlinspect.instrumentation.pipeline_executor import pipeline_executor

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def test_pipeline_executor_py_file():
    """
    Tests whether the PipelineExecutor works for .py files
    """
    extracted_dag = pipeline_executor.run(None, FILE_PY, None)
    assert extracted_dag == "test"


def test_pipeline_executor_nb_file():
    """
    Tests whether the PipelineExecutor works for .ipynb files
    """
    extracted_dag = pipeline_executor.run(FILE_NB, None, None)
    assert extracted_dag == "test"
