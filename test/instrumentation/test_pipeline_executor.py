"""
Tests whether the adult_easy test pipeline works
"""
import os
from inspect import cleandoc
from mlinspect.utils import get_project_root
from mlinspect.instrumentation import pipeline_executor

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


def test_pipeline_executor_module_information_extraction():
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
