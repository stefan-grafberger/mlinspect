"""
Tests whether the visualisation of the resulting DAG works
"""
import os

from mlinspect.pipeline_inspector import PipelineInspector
from test.utils import get_expected_dag_adult_easy_py
from mlinspect.utils import get_project_root
from mlinspect.visualisation import save_fig_to_path, get_dag_as_pretty_string

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_save_fig_to_path():
    """
    Tests whether the .py version of the inspector works
    """
    # extracted_dag = get_expected_dag_adult_easy_py()

    extracted_dag = PipelineInspector \
        .on_pipeline_from_py_file(FILE_PY) \
        .add_analyzer("test") \
        .execute()

    filename = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.png")
    save_fig_to_path(extracted_dag, filename)

    assert os.path.isfile(filename)


def test_get_dag_as_pretty_string():
    """
    Tests whether the .py version of the inspector works
    """
    extracted_dag = get_expected_dag_adult_easy_py()

    pretty_string = get_dag_as_pretty_string(extracted_dag)

    print(pretty_string)
