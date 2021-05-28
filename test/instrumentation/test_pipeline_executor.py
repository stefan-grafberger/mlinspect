"""
Tests whether the PipelineExecutor works
"""

import networkx
from testfixtures import compare

from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB
from mlinspect.instrumentation import _pipeline_executor
from ..testing_helper_utils import get_expected_dag_adult_easy


def test_pipeline_executor_py_file(mocker):
    """
    Tests whether the PipelineExecutor works for .py files
    """
    _pipeline_executor.singleton = _pipeline_executor.PipelineExecutor()

    before_call_used_value_spy = mocker.spy(_pipeline_executor, 'before_call_used_value')
    before_call_used_args_spy = mocker.spy(_pipeline_executor, 'before_call_used_args')
    before_call_used_kwargs_spy = mocker.spy(_pipeline_executor, 'before_call_used_kwargs')
    after_call_used_spy = mocker.spy(_pipeline_executor, 'after_call_used')

    extracted_dag = _pipeline_executor.singleton.run(None, ADULT_SIMPLE_PY, None, [], []).dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)

    assert before_call_used_value_spy.call_count == 11
    assert before_call_used_args_spy.call_count == 15
    assert before_call_used_kwargs_spy.call_count == 14
    assert after_call_used_spy.call_count == 15


def test_pipeline_executor_nb_file(mocker):
    """
    Tests whether the PipelineExecutor works for .ipynb files
    """
    _pipeline_executor.singleton = _pipeline_executor.PipelineExecutor()

    before_call_used_value_spy = mocker.spy(_pipeline_executor, 'before_call_used_value')
    before_call_used_args_spy = mocker.spy(_pipeline_executor, 'before_call_used_args')
    before_call_used_kwargs_spy = mocker.spy(_pipeline_executor, 'before_call_used_kwargs')
    after_call_used_spy = mocker.spy(_pipeline_executor, 'after_call_used')

    extracted_dag = _pipeline_executor.singleton.run(ADULT_SIMPLE_IPYNB, None, None, [], []).dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_IPYNB)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert before_call_used_value_spy.call_count == 11
    assert before_call_used_args_spy.call_count == 15
    assert before_call_used_kwargs_spy.call_count == 14
    assert after_call_used_spy.call_count == 15
