"""
Tests whether the visualisation of the resulting DAG works
"""
import os

from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.utils import get_project_root
from mlinspect.visualisation import save_fig_to_path

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_inspector_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    # pylint: disable=duplicate-code
    extracted_dag = PipelineInspector\
        .on_pipeline_from_py_file(FILE_PY)\
        .add_analyzer("test")\
        .execute()

    filename = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.png")
    save_fig_to_path(extracted_dag, filename)

    assert os.path.isfile(filename)
