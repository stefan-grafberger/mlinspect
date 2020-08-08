"""
Some util functions used in other tests
"""
import os
import ast
from inspect import cleandoc

import networkx

from mlinspect.instrumentation.dag_node import DagNode, OperatorType, CodeReference
from mlinspect.instrumentation.wir_extractor import WirExtractor
from mlinspect.utils import get_project_root

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


def get_expected_dag_adult_easy_py():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(18, OperatorType.DATA_SOURCE, CodeReference(12, 11),
                                   ('pandas.io.parsers', 'read_csv'),
                                   "adult_train.csv")
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(20, OperatorType.SELECTION, CodeReference(14, 7), ('pandas.core.frame', 'dropna'),
                              "dropna")
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_train_data = DagNode(56, OperatorType.TRAIN_DATA, CodeReference(28, 0),
                                  ('sklearn.pipeline', 'fit', 'Train Data'))
    expected_graph.add_edge(expected_select, expected_train_data)

    expected_pipeline_project_one = DagNode(34, OperatorType.PROJECTION, CodeReference(19, 75),
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                             'Projection'),
                                            "to ['education']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_one)
    expected_pipeline_project_two = DagNode(35, OperatorType.PROJECTION, CodeReference(19, 88),
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                             'Projection'),
                                            "to ['workclass']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_two)
    expected_pipeline_project_three = DagNode(40, OperatorType.PROJECTION, CodeReference(20, 49),
                                              ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                               'Projection'),
                                              "to ['age']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_three)
    expected_pipeline_project_four = DagNode(41, OperatorType.PROJECTION, CodeReference(20, 56),
                                             ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                              'Projection'),
                                             "to ['hours-per-week']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_four)

    expected_pipeline_transformer_one = DagNode(34, OperatorType.TRANSFORMER, CodeReference(19, 20),
                                                ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder)")
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one)
    expected_pipeline_transformer_two = DagNode(35, OperatorType.TRANSFORMER, CodeReference(19, 20),
                                                ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder)")
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two)
    expected_pipeline_transformer_three = DagNode(40, OperatorType.TRANSFORMER, CodeReference(20, 16),
                                                  ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                                                  "Numerical Encoder (StandardScaler)")
    expected_graph.add_edge(expected_pipeline_project_three, expected_pipeline_transformer_three)
    expected_pipeline_transformer_four = DagNode(41, OperatorType.TRANSFORMER, CodeReference(20, 16),
                                                 ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                                                 "Numerical Encoder (StandardScaler)")
    expected_graph.add_edge(expected_pipeline_project_four, expected_pipeline_transformer_four)

    expected_pipeline_concatenation = DagNode(46, OperatorType.CONCATENATION, CodeReference(18, 25),
                                              ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                               'Concatenation'))
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_three, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_four, expected_pipeline_concatenation)

    expected_estimator = DagNode(51, OperatorType.ESTIMATOR, CodeReference(26, 19),
                                 ('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'),
                                 "Decision Tree")
    expected_graph.add_edge(expected_pipeline_concatenation, expected_estimator)

    expected_pipeline_fit = DagNode(56, OperatorType.FIT, CodeReference(28, 0), ('sklearn.pipeline', 'fit', 'Pipeline'))
    expected_graph.add_edge(expected_estimator, expected_pipeline_fit)

    expected_project = DagNode(23, OperatorType.PROJECTION, CodeReference(16, 38), ('pandas.core.frame', '__getitem__'),
                               "to ['income-per-year']")
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagNode(28, OperatorType.PROJECTION_MODIFY, CodeReference(16, 9),
                                      ('sklearn.preprocessing._label', 'label_binarize'),
                                      "label_binarize, classes: ['>50K', '<=50K']")
    expected_graph.add_edge(expected_project, expected_project_modify)

    expected_train_labels = DagNode(56, OperatorType.TRAIN_LABELS, CodeReference(28, 0),
                                    ('sklearn.pipeline', 'fit', 'Train Labels'))
    expected_graph.add_edge(expected_project_modify, expected_train_labels)
    expected_graph.add_edge(expected_train_labels, expected_pipeline_fit)

    return expected_graph


def get_expected_dag_adult_easy_ipynb():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(18, OperatorType.DATA_SOURCE, CodeReference(18, 11),
                                   ('pandas.io.parsers', 'read_csv'),
                                   "adult_train.csv")
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(20, OperatorType.SELECTION, CodeReference(20, 7), ('pandas.core.frame', 'dropna'),
                              "dropna")
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_train_data = DagNode(56, OperatorType.TRAIN_DATA, CodeReference(34, 0),
                                  ('sklearn.pipeline', 'fit', 'Train Data'))
    expected_graph.add_edge(expected_select, expected_train_data)

    expected_pipeline_project_one = DagNode(34, OperatorType.PROJECTION, CodeReference(25, 75),
                                            ('sklearn.compose._column_transformer',
                                             'ColumnTransformer', 'Projection'),
                                            "to ['education']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_one)
    expected_pipeline_project_two = DagNode(35, OperatorType.PROJECTION, CodeReference(25, 88),
                                            ('sklearn.compose._column_transformer',
                                             'ColumnTransformer', 'Projection'),
                                            "to ['workclass']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_two)
    expected_pipeline_project_three = DagNode(40, OperatorType.PROJECTION, CodeReference(26, 49),
                                              ('sklearn.compose._column_transformer',
                                               'ColumnTransformer', 'Projection'),
                                              "to ['age']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_three)
    expected_pipeline_project_four = DagNode(41, OperatorType.PROJECTION, CodeReference(26, 56),
                                             ('sklearn.compose._column_transformer',
                                              'ColumnTransformer', 'Projection'),
                                             "to ['hours-per-week']")
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_four)

    expected_pipeline_transformer_one = DagNode(34, OperatorType.TRANSFORMER, CodeReference(25, 20),
                                                ('sklearn.preprocessing._encoders',
                                                 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder)")
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one)
    expected_pipeline_transformer_two = DagNode(35, OperatorType.TRANSFORMER, CodeReference(25, 20),
                                                ('sklearn.preprocessing._encoders',
                                                 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder)")
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two)
    expected_pipeline_transformer_three = DagNode(40, OperatorType.TRANSFORMER, CodeReference(26, 16),
                                                  ('sklearn.preprocessing._data',
                                                   'StandardScaler', 'Pipeline'),
                                                  "Numerical Encoder (StandardScaler)")
    expected_graph.add_edge(expected_pipeline_project_three, expected_pipeline_transformer_three)
    expected_pipeline_transformer_four = DagNode(41, OperatorType.TRANSFORMER, CodeReference(26, 16),
                                                 ('sklearn.preprocessing._data',
                                                  'StandardScaler', 'Pipeline'),
                                                 "Numerical Encoder (StandardScaler)")
    expected_graph.add_edge(expected_pipeline_project_four, expected_pipeline_transformer_four)

    expected_pipeline_concatenation = DagNode(46, OperatorType.CONCATENATION, CodeReference(24, 25),
                                              ('sklearn.compose._column_transformer',
                                               'ColumnTransformer', 'Concatenation'))
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_three, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_four, expected_pipeline_concatenation)

    expected_estimator = DagNode(51, OperatorType.ESTIMATOR, CodeReference(32, 19),
                                 ('sklearn.tree._classes', 'DecisionTreeClassifier',
                                  'Pipeline'),
                                 "Decision Tree")
    expected_graph.add_edge(expected_pipeline_concatenation, expected_estimator)

    expected_pipeline_fit = DagNode(56, OperatorType.FIT, CodeReference(34, 0), ('sklearn.pipeline', 'fit',
                                                                                 'Pipeline'))
    expected_graph.add_edge(expected_estimator, expected_pipeline_fit)

    expected_project = DagNode(23, OperatorType.PROJECTION, CodeReference(22, 38), ('pandas.core.frame', '__getitem__'),
                               "to ['income-per-year']")
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagNode(28, OperatorType.PROJECTION_MODIFY, CodeReference(22, 9),
                                      ('sklearn.preprocessing._label', 'label_binarize'),
                                      "label_binarize, classes: ['>50K', '<=50K']")
    expected_graph.add_edge(expected_project, expected_project_modify)

    expected_train_labels = DagNode(56, OperatorType.TRAIN_LABELS, CodeReference(34, 0),
                                    ('sklearn.pipeline', 'fit', 'Train Labels'))
    expected_graph.add_edge(expected_project_modify, expected_train_labels)
    expected_graph.add_edge(expected_train_labels, expected_pipeline_fit)

    return expected_graph


def get_module_info():
    """
    Get the module info for the adult_easy pipeline
    """
    module_info = {CodeReference(10, 0): ('builtins', 'print'),
                   CodeReference(11, 13): ('posixpath', 'join'),
                   CodeReference(11, 26): ('builtins', 'str'),
                   CodeReference(11, 30): ('mlinspect.utils', 'get_project_root'),
                   CodeReference(12, 11): ('pandas.io.parsers', 'read_csv'),
                   CodeReference(14, 7): ('pandas.core.frame', 'dropna'),
                   CodeReference(16, 9): ('sklearn.preprocessing._label', 'label_binarize'),
                   CodeReference(16, 38): ('pandas.core.frame', '__getitem__'),
                   CodeReference(18, 25): ('sklearn.compose._column_transformer', 'ColumnTransformer'),
                   CodeReference(19, 20): ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                   CodeReference(20, 16): ('sklearn.preprocessing._data', 'StandardScaler'),
                   CodeReference(24, 18): ('sklearn.pipeline', 'Pipeline'),
                   CodeReference(26, 19): ('sklearn.tree._classes', 'DecisionTreeClassifier'),
                   CodeReference(28, 0): ('sklearn.pipeline', 'fit'),
                   CodeReference(31, 0): ('builtins', 'print')}

    return module_info


def get_call_description_info():
    """
    Get the module info for the adult_easy pipeline
    """
    call_description_info = {
        CodeReference(12, 11): 'adult_train.csv',
        CodeReference(14, 7): 'dropna',
        CodeReference(16, 38): 'to [\'income-per-year\']',
        CodeReference(16, 9): 'label_binarize, classes: [\'>50K\', \'<=50K\']',
        CodeReference(19, 20): 'Categorical Encoder (OneHotEncoder)',
        CodeReference(20, 16): 'Numerical Encoder (StandardScaler)',
        CodeReference(26, 19): 'Decision Tree'
    }

    return call_description_info


def get_adult_easy_py_ast():
    """
    Get the parsed AST for the adult_easy pipeline
    """
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
    return test_ast


def get_test_wir():
    """
    Get the extracted WIR for the adult_easy pipeline with runtime info
    """
    test_ast = get_adult_easy_py_ast()
    extractor = WirExtractor(test_ast)
    extractor.extract_wir()
    wir = extractor.add_runtime_info(get_module_info(), get_call_description_info())

    return wir


def get_pandas_read_csv_and_dropna_code():
    """
    Get a simple code snipped that loads the adult_easy data and runs dropna
    """
    code = cleandoc("""
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root

            train_file = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")
            raw_data = pd.read_csv(train_file)
            data = raw_data.dropna()
            """)
    return code
