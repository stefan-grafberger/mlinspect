"""
Tests whether the PipelineExecutor works
"""
import os
from inspect import cleandoc

import networkx
from testfixtures import compare

from mlinspect.instrumentation.dag_node import CodeReference
from mlinspect.utils import get_project_root
from mlinspect.instrumentation import pipeline_executor
from ..utils import get_expected_dag_adult_easy_py, get_expected_dag_adult_easy_ipynb, \
    get_pandas_read_csv_and_dropna_code

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def test_pipeline_executor_py_file(mocker):
    """
    Tests whether the PipelineExecutor works for .py files
    """
    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()

    before_call_used_value_spy = mocker.spy(pipeline_executor, 'before_call_used_value')
    before_call_used_args_spy = mocker.spy(pipeline_executor, 'before_call_used_args')
    before_call_used_kwargs_spy = mocker.spy(pipeline_executor, 'before_call_used_kwargs')
    after_call_used_spy = mocker.spy(pipeline_executor, 'after_call_used')

    extracted_dag = pipeline_executor.singleton.run(None, FILE_PY, None, []).dag
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
    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()

    before_call_used_value_spy = mocker.spy(pipeline_executor, 'before_call_used_value')
    before_call_used_args_spy = mocker.spy(pipeline_executor, 'before_call_used_args')
    before_call_used_kwargs_spy = mocker.spy(pipeline_executor, 'before_call_used_kwargs')
    after_call_used_spy = mocker.spy(pipeline_executor, 'after_call_used')

    extracted_dag = pipeline_executor.singleton.run(FILE_NB, None, None, []).dag
    expected_dag = get_expected_dag_adult_easy_ipynb()
    assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)

    assert before_call_used_value_spy.call_count == 11
    assert before_call_used_args_spy.call_count == 15
    assert before_call_used_kwargs_spy.call_count == 14
    assert after_call_used_spy.call_count == 15


def test_pipeline_executor_function_call_info_extraction():
    """
    Tests whether the capturing of module information works
    """
    test_code = get_pandas_read_csv_and_dropna_code()

    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()
    pipeline_executor.singleton.run(None, None, test_code, [])
    expected_module_info = {CodeReference(5, 13, 5, 85): ('posixpath', 'join'),
                            CodeReference(5, 26, 5, 49): ('builtins', 'str'),
                            CodeReference(5, 30, 5, 48): ('mlinspect.utils', 'get_project_root'),
                            CodeReference(6, 11, 6, 34): ('pandas.io.parsers', 'read_csv'),
                            CodeReference(7, 7, 7, 24): ('pandas.core.frame', 'dropna')}

    compare(pipeline_executor.singleton.code_reference_to_module, expected_module_info)


def test_pipeline_executor_function_subscript_index_info_extraction():
    """
    Tests whether the capturing of module information works
    """
    test_code = cleandoc("""
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            train_file = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")
            raw_data = pd.read_csv(train_file, na_values='?', index_col=0)
            data = raw_data.dropna()
            data['income-per-year']
            """)

    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()
    pipeline_executor.singleton.run(None, None, test_code, [])
    expected_module_info = {CodeReference(5, 13, 5, 85): ('posixpath', 'join'),
                            CodeReference(5, 26, 5, 49): ('builtins', 'str'),
                            CodeReference(5, 30, 5, 48): ('mlinspect.utils', 'get_project_root'),
                            CodeReference(6, 11, 6, 62): ('pandas.io.parsers', 'read_csv'),
                            CodeReference(7, 7, 7, 24): ('pandas.core.frame', 'dropna'),
                            CodeReference(8, 0, 8, 23): ('pandas.core.frame', '__getitem__')}

    compare(pipeline_executor.singleton.code_reference_to_module, expected_module_info)
