"""
Tests whether the PipelineExecutor works
"""

import networkx
from testfixtures import compare

from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.instrumentation import _pipeline_executor
from ..testing_helper_utils import get_expected_dag_adult_easy


def test_pipeline_executor_py_file(mocker):
    """
    Tests whether the PipelineExecutor works for .py files
    """
    before_call_pandas_spy = mocker.spy(PandasBackend, 'before_call')
    after_call_pandas_spy = mocker.spy(PandasBackend, 'after_call')
    before_call_sklearn_spy = mocker.spy(SklearnBackend, 'before_call')
    after_call_sklearn_spy = mocker.spy(SklearnBackend, 'after_call')

    extracted_dag = _pipeline_executor.singleton.run(python_path=ADULT_SIMPLE_PY).dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)

    assert before_call_pandas_spy.call_count == 5
    assert after_call_pandas_spy.call_count == 5
    assert before_call_sklearn_spy.call_count == 7
    assert after_call_sklearn_spy.call_count == 7


def test_pipeline_executor_nb_file(mocker):
    """
    Tests whether the PipelineExecutor works for .ipynb files
    """
    _pipeline_executor.singleton = _pipeline_executor.PipelineExecutor()

    before_call_pandas_spy = mocker.spy(PandasBackend, 'before_call')
    after_call_pandas_spy = mocker.spy(PandasBackend, 'after_call')
    before_call_sklearn_spy = mocker.spy(SklearnBackend, 'before_call')
    after_call_sklearn_spy = mocker.spy(SklearnBackend, 'after_call')

    extracted_dag = _pipeline_executor.singleton.run(notebook_path=ADULT_SIMPLE_IPYNB).dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_IPYNB, 6)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

    assert before_call_pandas_spy.call_count == 5
    assert after_call_pandas_spy.call_count == 5
    assert before_call_sklearn_spy.call_count == 7
    assert after_call_sklearn_spy.call_count == 7
