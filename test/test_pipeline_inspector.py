"""
Tests whether the fluent API works
"""
import os

import networkx
from testfixtures import compare

from mlinspect.checks.check import Check, CheckStatus
from mlinspect.inspections.histogram_inspection import HistogramInspection
from mlinspect.inspections.materialize_first_rows_inspection import MaterializeFirstRowsInspection
from mlinspect.utils import get_project_root
from mlinspect.pipeline_inspector import PipelineInspector
from .utils import get_expected_dag_adult_easy_ipynb, get_expected_dag_adult_easy_py

ADULT_EASY_FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")

check = Check()\
        .no_bias_introduced_for(['race'])\
        .no_illegal_features()


def test_inspector_adult_easy_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_py_file(ADULT_EASY_FILE_PY)\
        .add_required_inspection(MaterializeFirstRowsInspection(5))\
        .add_check(check)\
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy_py()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert HistogramInspection(['race']) in inspector_result.inspection_to_annotations
    assert check in inspector_result.check_to_check_results
    check_result = inspector_result.check_to_check_results[check]
    assert check_result.status == CheckStatus.ERROR


def test_inspector_adult_easy_ipynb_pipeline():
    """
    Tests whether the .ipynb version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_ipynb_file(FILE_NB)\
        .add_required_inspection(MaterializeFirstRowsInspection(5)) \
        .add_check(check) \
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy_ipynb()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert HistogramInspection(['race']) in inspector_result.inspection_to_annotations
    assert check in inspector_result.check_to_check_results
    check_result = inspector_result.check_to_check_results[check]
    assert check_result.status == CheckStatus.ERROR


def test_inspector_adult_easy_str_pipeline():
    """
    Tests whether the str version of the inspector works
    """
    with open(ADULT_EASY_FILE_PY) as file:
        code = file.read()

        inspector_result = PipelineInspector\
            .on_pipeline_from_string(code)\
            .add_required_inspection(MaterializeFirstRowsInspection(5)) \
            .add_check(check) \
            .execute()
        extracted_dag = inspector_result.dag
        expected_dag = get_expected_dag_adult_easy_py()
        compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

        assert HistogramInspection(['race']) in inspector_result.inspection_to_annotations
        assert check in inspector_result.check_to_check_results
        check_result = inspector_result.check_to_check_results[check]
        assert check_result.status == CheckStatus.ERROR
