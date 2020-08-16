"""
Tests whether the fluent API works
"""
import os

import networkx
from testfixtures import compare

from mlinspect.instrumentation.analyzers.materialize_first_rows_analyzer import MaterializeFirstRowsAnalyzer
from mlinspect.utils import get_project_root
from mlinspect.pipeline_inspector import PipelineInspector
from .utils import get_expected_dag_adult_easy_ipynb, get_expected_dag_adult_easy_py

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def test_inspector_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    inspection_result = PipelineInspector\
        .on_pipeline_from_py_file(FILE_PY)\
        .add_analyzer(MaterializeFirstRowsAnalyzer(5))\
        .execute()
    extracted_dag = inspection_result.dag
    expected_dag = get_expected_dag_adult_easy_py()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_ipynb_pipeline():
    """
    Tests whether the .ipynb version of the inspector works
    """
    inspection_result = PipelineInspector\
        .on_pipeline_from_ipynb_file(FILE_NB)\
        .add_analyzer(MaterializeFirstRowsAnalyzer(5))\
        .execute()
    extracted_dag = inspection_result.dag
    expected_dag = get_expected_dag_adult_easy_ipynb()
    assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)


def test_inspector_str_pipeline():
    """
    Tests whether the str version of the inspector works
    """
    with open(FILE_PY) as file:
        code = file.read()

        inspection_result = PipelineInspector\
            .on_pipeline_from_string(code)\
            .add_analyzer(MaterializeFirstRowsAnalyzer(5))\
            .execute()
        extracted_dag = inspection_result.dag
        expected_dag = get_expected_dag_adult_easy_py()
        assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)
