"""
Tests whether the PipelineExecutor works
"""
from inspect import cleandoc

import networkx
from testfixtures import compare

from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB
from mlinspect import OperatorType
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import CodeReference, DagNode
from mlinspect.testing._testing_helper_utils import get_expected_dag_adult_easy


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


def test_pipeline_executor_py_file_without_code_reference_tracking(mocker):
    """
    Tests whether the PipelineExecutor works for .py files
    """
    before_call_pandas_spy = mocker.spy(PandasBackend, 'before_call')
    after_call_pandas_spy = mocker.spy(PandasBackend, 'after_call')
    before_call_sklearn_spy = mocker.spy(SklearnBackend, 'before_call')
    after_call_sklearn_spy = mocker.spy(SklearnBackend, 'after_call')

    extracted_dag = _pipeline_executor.singleton.run(python_path=ADULT_SIMPLE_PY, track_code_references=False).dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY, with_code_references=False)
    assert networkx.to_dict_of_dicts(extracted_dag) == networkx.to_dict_of_dicts(expected_dag)

    assert before_call_pandas_spy.call_count == 5
    assert after_call_pandas_spy.call_count == 5
    assert before_call_sklearn_spy.call_count == 7
    assert after_call_sklearn_spy.call_count == 7


def test_pipeline_executor_nb_file(mocker):
    """
    Tests whether the PipelineExecutor works for .ipynb files
    """
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


def test_func_defs_and_loops():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = cleandoc("""
        import pandas as pd

        def black_box_df_op():
            df = pd.DataFrame([0, 1], columns=['A'])
            return df
        df = black_box_df_op()
        for _ in range(2):
            df = df.dropna()
        """)

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True).dag

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0, "<string-source>", 4, OperatorType.DATA_SOURCE,
                                   ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                   optional_code_reference=CodeReference(4, 9, 4, 44),
                                   optional_source_code="pd.DataFrame([0, 1], columns=['A'])")
    expected_select_1 = DagNode(1, "<string-source>", 8, OperatorType.SELECTION,
                                module=('pandas.core.frame', 'dropna'), description='dropna', columns=['A'],
                                optional_code_reference=CodeReference(8, 9, 8, 20),
                                optional_source_code='df.dropna()')
    expected_dag.add_edge(expected_data_source, expected_select_1)
    expected_select_2 = DagNode(2, "<string-source>", 8, OperatorType.SELECTION,
                                module=('pandas.core.frame', 'dropna'), description='dropna', columns=['A'],
                                optional_code_reference=CodeReference(8, 9, 8, 20),
                                optional_source_code='df.dropna()')
    expected_dag.add_edge(expected_select_1, expected_select_2)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_func_defs_and_loops_without_code_reference_tracking():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = cleandoc("""
        import pandas as pd
        
        def black_box_df_op():
            df = pd.DataFrame([0, 1], columns=['A'])
            return df
        df = black_box_df_op()
        for _ in range(2):
            df = df.dropna()
        """)

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=False).dag

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0, "<string-source>", 4, OperatorType.DATA_SOURCE,
                                   ('pandas.core.frame', 'DataFrame'), description='', columns=['A'])
    expected_select_1 = DagNode(1, "<string-source>", 8, OperatorType.SELECTION,
                                module=('pandas.core.frame', 'dropna'), description='dropna', columns=['A'])
    expected_dag.add_edge(expected_data_source, expected_select_1)
    expected_select_2 = DagNode(2, "<string-source>", 8, OperatorType.SELECTION,
                                module=('pandas.core.frame', 'dropna'), description='dropna', columns=['A'])
    expected_dag.add_edge(expected_select_1, expected_select_2)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_annotation_storage():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = cleandoc("""
        import pandas
        df = pandas.DataFrame([["x", "y"], ["2", "3"]], columns=["a", "b"])
        assert df._mlinspect_annotation is not None
        """)

    _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)


def test_black_box_operation():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = cleandoc("""
        import pandas
        from mlinspect.testing._testing_helper_utils import black_box_df_op
        
        df = black_box_df_op()
        df = df.dropna()
        print("df")
        """)

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True).dag

    expected_dag = networkx.DiGraph()
    expected_missing_op = DagNode(-1, "<string-source>", 5, OperatorType.MISSING_OP,
                                  description='Warning! Operator <string-source>:5 (df.dropna()) encountered a '
                                              'DataFrame resulting from an operation without mlinspect support!',
                                  columns=['A'], optional_code_reference=CodeReference(5, 5, 5, 16),
                                  optional_source_code='df.dropna()')
    expected_select = DagNode(0, "<string-source>", 5, OperatorType.SELECTION, module=('pandas.core.frame', 'dropna'),
                              description='dropna', columns=['A'], optional_code_reference=CodeReference(5, 5, 5, 16),
                              optional_source_code='df.dropna()')
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))
