"""
Tests whether the fluent API works
"""

import networkx
from testfixtures import compare

from mlinspect import PipelineInspector
from mlinspect.checks import CheckStatus, NoBiasIntroducedFor, NoIllegalFeatures
from mlinspect.inspections import HistogramForColumns, MaterializeFirstOutputRows
from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB
from .testing_helper_utils import get_expected_dag_adult_easy_ipynb, get_expected_dag_adult_easy_py


def test_inspector_adult_easy_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY)\
        .add_required_inspection(MaterializeFirstOutputRows(5))\
        .add_check(NoBiasIntroducedFor(['race']))\
        .add_check(NoIllegalFeatures())\
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy_py()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert HistogramForColumns(['race']) in inspector_result.inspection_to_annotations
    check_to_check_results = inspector_result.check_to_check_results
    assert check_to_check_results[NoBiasIntroducedFor(['race'])].status == CheckStatus.SUCCESS
    assert check_to_check_results[NoIllegalFeatures()].status == CheckStatus.FAILURE


def test_inspector_adult_easy_py_pipeline_without_inspections():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY)\
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy_py()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_ipynb_pipeline():
    """
    Tests whether the .ipynb version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_ipynb_file(ADULT_SIMPLE_IPYNB)\
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .add_check(NoBiasIntroducedFor(['race'])) \
        .add_check(NoIllegalFeatures()) \
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy_ipynb()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert HistogramForColumns(['race']) in inspector_result.inspection_to_annotations
    check_to_check_results = inspector_result.check_to_check_results
    assert check_to_check_results[NoBiasIntroducedFor(['race'])].status == CheckStatus.SUCCESS
    assert check_to_check_results[NoIllegalFeatures()].status == CheckStatus.FAILURE


def test_inspector_adult_easy_str_pipeline():
    """
    Tests whether the str version of the inspector works
    """
    with open(ADULT_SIMPLE_PY) as file:
        code = file.read()

        inspector_result = PipelineInspector\
            .on_pipeline_from_string(code)\
            .add_required_inspection(MaterializeFirstOutputRows(5)) \
            .add_check(NoBiasIntroducedFor(['race'])) \
            .add_check(NoIllegalFeatures()) \
            .execute()
        extracted_dag = inspector_result.dag
        expected_dag = get_expected_dag_adult_easy_py()
        compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

        assert HistogramForColumns(['race']) in inspector_result.inspection_to_annotations
        check_to_check_results = inspector_result.check_to_check_results
        assert check_to_check_results[NoBiasIntroducedFor(['race'])].status == CheckStatus.SUCCESS
        assert check_to_check_results[NoIllegalFeatures()].status == CheckStatus.FAILURE
