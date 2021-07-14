"""
Tests whether the fluent API works
"""

import networkx
from testfixtures import compare

from example_pipelines.healthcare import custom_monkeypatching
from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB, HEALTHCARE_PY
from mlinspect import PipelineInspector, OperatorType
from mlinspect.checks import CheckStatus, NoBiasIntroducedFor, NoIllegalFeatures
from mlinspect.inspections import HistogramForColumns, MaterializeFirstOutputRows
from mlinspect.testing._testing_helper_utils import get_expected_dag_adult_easy


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
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert HistogramForColumns(['race']) in list(inspector_result.dag_node_to_inspection_results.values())[0]
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
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
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
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_IPYNB, 6)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert HistogramForColumns(['race']) in list(inspector_result.dag_node_to_inspection_results.values())[0]
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
        expected_dag = get_expected_dag_adult_easy("<string-source>")
        compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

        assert HistogramForColumns(['race']) in list(inspector_result.dag_node_to_inspection_results.values())[0]
        check_to_check_results = inspector_result.check_to_check_results
        assert check_to_check_results[NoBiasIntroducedFor(['race'])].status == CheckStatus.SUCCESS
        assert check_to_check_results[NoIllegalFeatures()].status == CheckStatus.FAILURE


def test_inspector_additional_module():
    """
    Tests whether the str version of the inspector works
    """
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .execute()

    assert_healthcare_pipeline_output_complete(inspector_result)


def test_inspector_additional_modules():
    """
    Tests whether the str version of the inspector works
    """
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .execute()

    assert_healthcare_pipeline_output_complete(inspector_result)


def assert_healthcare_pipeline_output_complete(inspector_result):
    """ Assert that the healthcare DAG was extracted completely """
    for dag_node, inspection_result in inspector_result.dag_node_to_inspection_results.items():
        assert dag_node.operator_info.operator != OperatorType.MISSING_OP
        assert MaterializeFirstOutputRows(5) in inspection_result
        if dag_node.operator_info.operator is not OperatorType.ESTIMATOR:
            assert inspection_result[MaterializeFirstOutputRows(5)] is not None
        else:
            assert inspection_result[MaterializeFirstOutputRows(5)] is None
    assert len(inspector_result.dag) == 37
