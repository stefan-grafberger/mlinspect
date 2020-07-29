"""
Tests whether the visualisation of the resulting DAG works
"""
import os

import networkx

from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.utils import get_project_root
from mlinspect.visualisation import save_fig_to_path
from .utils import get_expected_dag_adult_easy_py

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_inspector_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    extracted_dag = PipelineInspector\
        .on_pipeline_from_py_file(FILE_PY)\
        .add_analyzer("test")\
        .execute()
    expected_dag = get_expected_dag_adult_easy_py()
    assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)

    filename = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.png")
    save_fig_to_path(extracted_dag, filename)
