"""
Tests whether the PipelineExecutor works
"""
import ast
from inspect import cleandoc

import astunparse
import networkx
from testfixtures import compare

from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB
from mlinspect import OperatorType, OperatorContext, FunctionInfo
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import CodeReference, DagNode, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.testing._testing_helper_utils import get_expected_dag_adult_easy, \
    get_test_code_with_function_def_and_for_loop


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
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

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
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))

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
    test_code = get_test_code_with_function_def_and_for_loop()

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True).dag

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 4),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(4, 9, 4, 44), "pd.DataFrame([0, 1], columns=['A'])"))
    expected_select_1 = DagNode(1,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A']),
                                OptionalCodeInfo(CodeReference(8, 9, 8, 20), 'df.dropna()'))
    expected_dag.add_edge(expected_data_source, expected_select_1)
    expected_select_2 = DagNode(2,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A']),
                                OptionalCodeInfo(CodeReference(8, 9, 8, 20), 'df.dropna()'))
    expected_dag.add_edge(expected_select_1, expected_select_2)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_func_defs_and_loops_without_code_reference_tracking():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = get_test_code_with_function_def_and_for_loop()

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=False).dag

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 4),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']))
    expected_select_1 = DagNode(1,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A']))
    expected_dag.add_edge(expected_data_source, expected_select_1)
    expected_select_2 = DagNode(2,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A']))
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
    expected_missing_op = DagNode(-1,
                                  BasicCodeLocation("<string-source>", 5),
                                  OperatorContext(OperatorType.MISSING_OP, None),
                                  DagNodeDetails('Warning! Operator <string-source>:5 (df.dropna()) encountered a '
                                                 'DataFrame resulting from an operation without mlinspect support!',
                                                 ['A']),
                                  OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'))
    expected_select = DagNode(0,
                              BasicCodeLocation("<string-source>", 5),
                              OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                              DagNodeDetails('dropna', ['A']),
                              OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'))
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_instrument_pipeline_with_code_reference_tracking():
    """
    Tests whether the instrumentation modifies user code as expected with code reference tracking
    """
    test_code = get_test_code_with_function_def_and_for_loop()
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, True)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlinspect.instrumentation._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd
            
            def black_box_df_op():
                df = pd.DataFrame([0, 1], **set_code_reference_call(4, 9, 4, 44, columns=['A']))
                return df
            df = black_box_df_op(**set_code_reference_call(6, 5, 6, 22))
            for _ in range(2, **set_code_reference_call(7, 9, 7, 17)):
                df = df.dropna(**set_code_reference_call(8, 9, 8, 20))
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)


def test_instrument_pipeline_without_code_reference_tracking():
    """
    Tests whether the instrumentation modifies user code as expected without code reference tracking
    """
    test_code = get_test_code_with_function_def_and_for_loop()
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, False)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlinspect.instrumentation._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd

            def black_box_df_op():
                df = pd.DataFrame([0, 1], columns=['A'])
                return df
            df = black_box_df_op()
            for _ in range(2):
                df = df.dropna()
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)
