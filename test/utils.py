"""
Some util functions used in other tests
"""
import os
import ast
import networkx

from mlinspect.instrumentation.dag_vertex import DagVertex
from mlinspect.utils import get_project_root

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def get_expected_dag_adult_easy_py():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    expected_graph = networkx.DiGraph()

    expected_data_source = DagVertex(18, "Data Source", 12, 11, ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_node(expected_data_source)

    expected_select = DagVertex(20, "Selection", 14, 7, ('pandas.core.frame', 'dropna'))
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_project = DagVertex(23, "Projection", 16, 38, ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagVertex(28, "Projection (Modify)", 16, 9,
                                        ('sklearn.preprocessing._label', 'label_binarize'))
    expected_graph.add_edge(expected_project, expected_project_modify)

    return expected_graph


def get_expected_dag_adult_easy_ipynb():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    expected_graph = networkx.DiGraph()

    expected_data_source = DagVertex(18, "Data Source", 18, 11, ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_node(expected_data_source)

    expected_select = DagVertex(20, "Selection", 20, 7, ('pandas.core.frame', 'dropna'))
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_project = DagVertex(23, "Projection", 22, 38, ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagVertex(28, "Projection (Modify)", 22, 9,
                                        ('sklearn.preprocessing._label', 'label_binarize'))
    expected_graph.add_edge(expected_project, expected_project_modify)

    return expected_graph


def get_module_info():
    """
    Get the module info for the adult_easy pipeline
    """
    module_info = {(10, 0): ('builtins', 'print'),
                   (11, 13): ('posixpath', 'join'),
                   (11, 26): ('builtins', 'str'),
                   (11, 30): ('mlinspect.utils', 'get_project_root'),
                   (12, 11): ('pandas.io.parsers', 'read_csv'),
                   (14, 7): ('pandas.core.frame', 'dropna'),
                   (16, 9): ('sklearn.preprocessing._label', 'label_binarize'),
                   (16, 38): ('pandas.core.frame', '__getitem__'),
                   (18, 25): ('sklearn.compose._column_transformer', 'ColumnTransformer'),
                   (19, 20): ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                   (20, 16): ('sklearn.preprocessing._data', 'StandardScaler'),
                   (24, 18): ('sklearn.pipeline', 'Pipeline'),
                   (26, 19): ('sklearn.tree._classes', 'DecisionTreeClassifier'),
                   (28, 0): ('sklearn.pipeline', 'fit'),
                   (31, 0): ('builtins', 'print')}

    return module_info


def get_adult_easy_py_ast():
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
    return test_ast
