"""
Tests whether the PipelineExecutor works
"""
from inspect import cleandoc

import networkx
from testfixtures import compare

from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import CodeReference
from ..testing_helper_utils import get_pandas_read_csv_and_dropna_code, get_expected_dag_adult_easy_py, \
    get_expected_dag_adult_easy_ipynb


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
    expected_dag = get_expected_dag_adult_easy_py()
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
    expected_dag = get_expected_dag_adult_easy_ipynb()
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert before_call_used_value_spy.call_count == 11
    assert before_call_used_args_spy.call_count == 15
    assert before_call_used_kwargs_spy.call_count == 14
    assert after_call_used_spy.call_count == 15


def test_pipeline_executor_function_call_info_extraction():
    """
    Tests whether the capturing of module information works
    """
    test_code = get_pandas_read_csv_and_dropna_code()

    _pipeline_executor.singleton = _pipeline_executor.PipelineExecutor()
    _pipeline_executor.singleton.run(None, None, test_code, [], [])
    expected_module_info = {CodeReference(6, 11, 6, 34): ('pandas.io.parsers', 'read_csv'),
                            CodeReference(7, 7, 7, 24): ('pandas.core.frame', 'dropna'),
                            CodeReference(8, 16, 8, 55): ('pandas.core.frame', '__getitem__')}

    pandas_backend = [backend for backend in _pipeline_executor.singleton.backends
                      if isinstance(backend, PandasBackend)][0]
    compare(pandas_backend.code_reference_to_module, expected_module_info)


def test_pipeline_executor_function_subscript_index_info_extraction():
    """
    Tests whether the capturing of module information works
    """
    test_code = cleandoc("""
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
            raw_data = pd.read_csv(train_file, na_values='?', index_col=0)
            data = raw_data.dropna()
            data['income-per-year']
            """)

    _pipeline_executor.singleton = _pipeline_executor.PipelineExecutor()
    _pipeline_executor.singleton.run(None, None, test_code, [], [])
    expected_module_info = {CodeReference(6, 11, 6, 62): ('pandas.io.parsers', 'read_csv'),
                            CodeReference(7, 7, 7, 24): ('pandas.core.frame', 'dropna'),
                            CodeReference(8, 0, 8, 23): ('pandas.core.frame', '__getitem__')}

    pandas_backend = [backend for backend in _pipeline_executor.singleton.backends
                      if isinstance(backend, PandasBackend)][0]
    compare(pandas_backend.code_reference_to_module, expected_module_info)
