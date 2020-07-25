"""
Tests whether the adult_easy test pipeline works
"""
import os
import ast
from inspect import cleandoc
from mlinspect.utils import get_project_root
from mlinspect.instrumentation import pipeline_executor
import astpretty

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def test_pipeline_executor_py_file():
    """
    Tests whether the PipelineExecutor works for .py files
    """
    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()
    extracted_dag = pipeline_executor.singleton.run(None, FILE_PY, None)
    assert extracted_dag == "test"


def test_pipeline_executor_nb_file():
    """
    Tests whether the PipelineExecutor works for .ipynb files
    """
    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()
    extracted_dag = pipeline_executor.singleton.run(FILE_NB, None, None)
    assert extracted_dag == "test"


def test_pipeline_executor_function_call_info_extraction():
    """
    Tests whether the capturing of module information works
    """
    test_code = cleandoc("""
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root
            
            train_file = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")
            raw_data = pd.read_csv(train_file)
            data = raw_data.dropna()
            """)

    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()
    pipeline_executor.singleton.run(None, None, test_code)
    expected_module_info = {(5, 13): ('posixpath', 'join'),
                            (5, 26): ('builtins', 'str'),
                            (5, 30): ('mlinspect.utils', 'get_project_root'),
                            (6, 11): ('pandas.io.parsers', 'read_csv'),
                            (7, 7): ('pandas.core.frame', 'dropna')}

    assert pipeline_executor.singleton.ast_call_node_id_to_module == expected_module_info


def test_pipeline_executor_function_subscript_index_info_extraction():
    """
    Tests whether the capturing of module information works
    """
    test_code = cleandoc("""
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            train_file = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")
            raw_data = pd.read_csv(train_file)
            data = raw_data.dropna()
            data['income-per-year']
            """)

    pipeline_executor.singleton = pipeline_executor.PipelineExecutor()
    pipeline_executor.singleton.run(None, None, test_code)
    expected_module_info = {(5, 13): ('posixpath', 'join'),
                            (5, 26): ('builtins', 'str'),
                            (5, 30): ('mlinspect.utils', 'get_project_root'),
                            (6, 11): ('pandas.io.parsers', 'read_csv'),
                            (7, 7): ('pandas.core.frame', 'dropna'),
                            (8, 0): ('pandas.core.frame', '__getitem__')}

    assert pipeline_executor.singleton.ast_call_node_id_to_module == expected_module_info


def test_impl_stuff():
    """
    Tests whether the capturing of module information works
    """
    test_code = cleandoc("""
            def capture_args(*args):
                print(args)
                return args
        
            def capture_kwargs(**kwargs):
                print(kwargs)
                return kwargs
        
            print(*capture_args("test", "comma"), **capture_kwargs(sep=', '))
            """)

    parsed_ast = ast.parse(test_code)
    astpretty.pprint(parsed_ast)
    print("exec")
    exec(compile(parsed_ast, filename="<ast>", mode="exec"), {})
