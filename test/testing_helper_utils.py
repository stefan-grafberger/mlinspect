"""
Some util functions used in other tests
"""
import ast
import os
from inspect import cleandoc
from test.backends.random_annotation_testing_inspection import RandomAnnotationTestingInspection
import networkx
from demo.feature_overview.missing_embeddings import MissingEmbeddings
from example_pipelines._pipelines import ADULT_SIMPLE_PY
from mlinspect.checks._no_bias_introduced_for import NoBiasIntroducedFor
from mlinspect.checks._no_illegal_features import NoIllegalFeatures
from mlinspect.visualisation._visualisation import save_fig_to_path
from mlinspect.inspections._lineage import RowLineage
from mlinspect.inspections._materialize_first_output_rows import MaterializeFirstOutputRows
from mlinspect.instrumentation._dag_node import DagNode, OperatorType, CodeReference
from mlinspect.instrumentation._wir_extractor import WirExtractor
from mlinspect._pipeline_inspector import PipelineInspector


def get_expected_dag_adult_easy_py_without_columns():
    """
    Get the expected DAG for the adult_easy pipeline without column runtime info
    """
    dag = get_expected_dag_adult_easy_py()
    for node in dag:
        node.columns = None
    return dag


def get_expected_dag_adult_easy_ipynb_without_columns():
    """
    Get the expected DAG for the adult_easy pipeline without column runtime info
    """
    dag = get_expected_dag_adult_easy_ipynb()
    for node in dag:
        node.columns = None
    return dag


def get_expected_dag_adult_easy_py():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(18, OperatorType.DATA_SOURCE, CodeReference(12, 11, 12, 62),
                                   ('pandas.io.parsers', 'read_csv'),
                                   "adult_train.csv",
                                   ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                    'hours-per-week', 'native-country', 'income-per-year'])
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(20, OperatorType.SELECTION, CodeReference(14, 7, 14, 24), ('pandas.core.frame', 'dropna'),
                              "dropna", ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                         'hours-per-week', 'native-country', 'income-per-year'])
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_train_data = DagNode(56, OperatorType.TRAIN_DATA, CodeReference(24, 18, 26, 51),
                                  ('sklearn.pipeline', 'fit', 'Train Data'), None,
                                  ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                   'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                   'capital-loss', 'hours-per-week', 'native-country', 'income-per-year'])
    expected_graph.add_edge(expected_select, expected_train_data)

    expected_pipeline_project_one = DagNode(34, OperatorType.PROJECTION, CodeReference(18, 25, 21, 2),
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                             'Projection'),
                                            "to ['education'] (ColumnTransformer)", ['education'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_one)
    expected_pipeline_project_two = DagNode(35, OperatorType.PROJECTION, CodeReference(18, 25, 21, 2),
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                             'Projection'),
                                            "to ['workclass'] (ColumnTransformer)", ['workclass'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_two)
    expected_pipeline_project_three = DagNode(40, OperatorType.PROJECTION, CodeReference(18, 25, 21, 2),
                                              ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                               'Projection'),
                                              "to ['age'] (ColumnTransformer)", ['age'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_three)
    expected_pipeline_project_four = DagNode(41, OperatorType.PROJECTION, CodeReference(18, 25, 21, 2),
                                             ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                              'Projection'),
                                             "to ['hours-per-week'] (ColumnTransformer)", ['hours-per-week'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_four)

    expected_pipeline_transformer_one = DagNode(34, OperatorType.TRANSFORMER, CodeReference(19, 20, 19, 72),
                                                ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder), Column: 'education'",
                                                ['education'])
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one)
    expected_pipeline_transformer_two = DagNode(35, OperatorType.TRANSFORMER, CodeReference(19, 20, 19, 72),
                                                ('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder), Column: 'workclass'",
                                                ['workclass'])
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two)
    expected_pipeline_transformer_three = DagNode(40, OperatorType.TRANSFORMER, CodeReference(20, 16, 20, 46),
                                                  ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                                                  "Numerical Encoder (StandardScaler), Column: 'age'", ['age'])
    expected_graph.add_edge(expected_pipeline_project_three, expected_pipeline_transformer_three)
    expected_pipeline_transformer_four = DagNode(41, OperatorType.TRANSFORMER, CodeReference(20, 16, 20, 46),
                                                 ('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                                                 "Numerical Encoder (StandardScaler), Column: 'hours-per-week'",
                                                 ['hours-per-week'])
    expected_graph.add_edge(expected_pipeline_project_four, expected_pipeline_transformer_four)

    expected_pipeline_concatenation = DagNode(46, OperatorType.CONCATENATION, CodeReference(18, 25, 21, 2),
                                              ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                               'Concatenation'), None, ['array'])
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_three, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_four, expected_pipeline_concatenation)

    expected_estimator = DagNode(51, OperatorType.ESTIMATOR, CodeReference(26, 19, 26, 48),
                                 ('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'),
                                 "Decision Tree")
    expected_graph.add_edge(expected_pipeline_concatenation, expected_estimator)

    expected_pipeline_fit = DagNode(56, OperatorType.FIT, CodeReference(24, 18, 26, 51),
                                    ('sklearn.pipeline', 'fit', 'Pipeline'))
    expected_graph.add_edge(expected_estimator, expected_pipeline_fit)

    expected_project = DagNode(23, OperatorType.PROJECTION, CodeReference(16, 38, 16, 61),
                               ('pandas.core.frame', '__getitem__', 'Projection'), "to ['income-per-year']",
                               ['income-per-year'])
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagNode(28, OperatorType.PROJECTION_MODIFY, CodeReference(16, 9, 16, 89),
                                      ('sklearn.preprocessing._label', 'label_binarize'),
                                      "label_binarize, classes: ['>50K', '<=50K']", ['array'])
    expected_graph.add_edge(expected_project, expected_project_modify)

    expected_train_labels = DagNode(56, OperatorType.TRAIN_LABELS, CodeReference(24, 18, 26, 51),
                                    ('sklearn.pipeline', 'fit', 'Train Labels'), None, ['array'])
    expected_graph.add_edge(expected_project_modify, expected_train_labels)
    expected_graph.add_edge(expected_train_labels, expected_pipeline_fit)

    return expected_graph


def get_expected_dag_adult_easy_ipynb():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(18, OperatorType.DATA_SOURCE, CodeReference(18, 11, 18, 62),
                                   ('pandas.io.parsers', 'read_csv'),
                                   "adult_train.csv",
                                   ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                    'hours-per-week', 'native-country', 'income-per-year'])
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(20, OperatorType.SELECTION, CodeReference(20, 7, 20, 24), ('pandas.core.frame', 'dropna'),
                              "dropna", ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                         'hours-per-week', 'native-country', 'income-per-year'])
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_train_data = DagNode(56, OperatorType.TRAIN_DATA, CodeReference(30, 18, 32, 51),
                                  ('sklearn.pipeline', 'fit', 'Train Data'), None,
                                  ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                   'hours-per-week', 'native-country', 'income-per-year'])
    expected_graph.add_edge(expected_select, expected_train_data)

    expected_pipeline_project_one = DagNode(34, OperatorType.PROJECTION, CodeReference(24, 25, 27, 2),
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                             'Projection'),
                                            "to ['education'] (ColumnTransformer)", ['education'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_one)
    expected_pipeline_project_two = DagNode(35, OperatorType.PROJECTION, CodeReference(24, 25, 27, 2),
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer',
                                             'Projection'),
                                            "to ['workclass'] (ColumnTransformer)", ['workclass'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_two)
    expected_pipeline_project_three = DagNode(40, OperatorType.PROJECTION, CodeReference(24, 25, 27, 2),
                                              ('sklearn.compose._column_transformer',
                                               'ColumnTransformer', 'Projection'),
                                              "to ['age'] (ColumnTransformer)", ['age'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_three)
    expected_pipeline_project_four = DagNode(41, OperatorType.PROJECTION, CodeReference(24, 25, 27, 2),
                                             ('sklearn.compose._column_transformer',
                                              'ColumnTransformer', 'Projection'),
                                             "to ['hours-per-week'] (ColumnTransformer)", ['hours-per-week'])
    expected_graph.add_edge(expected_train_data, expected_pipeline_project_four)

    expected_pipeline_transformer_one = DagNode(34, OperatorType.TRANSFORMER, CodeReference(25, 20, 25, 72),
                                                ('sklearn.preprocessing._encoders',
                                                 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder), Column: 'education'",
                                                ['education'])
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one)
    expected_pipeline_transformer_two = DagNode(35, OperatorType.TRANSFORMER, CodeReference(25, 20, 25, 72),
                                                ('sklearn.preprocessing._encoders',
                                                 'OneHotEncoder', 'Pipeline'),
                                                "Categorical Encoder (OneHotEncoder), Column: 'workclass'",
                                                ['workclass'])
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two)
    expected_pipeline_transformer_three = DagNode(40, OperatorType.TRANSFORMER, CodeReference(26, 16, 26, 46),
                                                  ('sklearn.preprocessing._data',
                                                   'StandardScaler', 'Pipeline'),
                                                  "Numerical Encoder (StandardScaler), Column: 'age'", ['age'])
    expected_graph.add_edge(expected_pipeline_project_three, expected_pipeline_transformer_three)
    expected_pipeline_transformer_four = DagNode(41, OperatorType.TRANSFORMER, CodeReference(26, 16, 26, 46),
                                                 ('sklearn.preprocessing._data',
                                                  'StandardScaler', 'Pipeline'),
                                                 "Numerical Encoder (StandardScaler), Column: 'hours-per-week'",
                                                 ['hours-per-week'])
    expected_graph.add_edge(expected_pipeline_project_four, expected_pipeline_transformer_four)

    expected_pipeline_concatenation = DagNode(46, OperatorType.CONCATENATION, CodeReference(24, 25, 27, 2),
                                              ('sklearn.compose._column_transformer',
                                               'ColumnTransformer', 'Concatenation'), None, ["array"])
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_three, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_four, expected_pipeline_concatenation)

    expected_estimator = DagNode(51, OperatorType.ESTIMATOR, CodeReference(32, 19, 32, 48),
                                 ('sklearn.tree._classes', 'DecisionTreeClassifier',
                                  'Pipeline'),
                                 "Decision Tree")
    expected_graph.add_edge(expected_pipeline_concatenation, expected_estimator)

    expected_pipeline_fit = DagNode(56, OperatorType.FIT, CodeReference(30, 18, 32, 51), ('sklearn.pipeline', 'fit',
                                                                                          'Pipeline'))
    expected_graph.add_edge(expected_estimator, expected_pipeline_fit)

    expected_project = DagNode(23, OperatorType.PROJECTION, CodeReference(22, 38, 22, 61),
                               ('pandas.core.frame', '__getitem__', 'Projection'),
                               "to ['income-per-year']", ['income-per-year'])
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagNode(28, OperatorType.PROJECTION_MODIFY, CodeReference(22, 9, 22, 89),
                                      ('sklearn.preprocessing._label', 'label_binarize'),
                                      "label_binarize, classes: ['>50K', '<=50K']", ["array"])
    expected_graph.add_edge(expected_project, expected_project_modify)

    expected_train_labels = DagNode(56, OperatorType.TRAIN_LABELS, CodeReference(30, 18, 32, 51),
                                    ('sklearn.pipeline', 'fit', 'Train Labels'), None, ['array'])
    expected_graph.add_edge(expected_project_modify, expected_train_labels)
    expected_graph.add_edge(expected_train_labels, expected_pipeline_fit)

    return expected_graph


def get_module_info():
    """
    Get the module info for the adult_easy pipeline
    """
    module_info = {CodeReference(lineno=10, col_offset=0, end_lineno=10, end_col_offset=23):
                   ('builtins', 'print'),
                   CodeReference(lineno=11, col_offset=30, end_lineno=11, end_col_offset=48):
                   ('mlinspect.utils', 'get_project_root'),
                   CodeReference(lineno=11, col_offset=26, end_lineno=11, end_col_offset=49):
                   ('builtins', 'str'),
                   CodeReference(lineno=11, col_offset=13, end_lineno=11, end_col_offset=107):
                   ('posixpath', 'join'),
                   CodeReference(lineno=12, col_offset=11, end_lineno=12, end_col_offset=62):
                   ('pandas.io.parsers', 'read_csv'),
                   CodeReference(lineno=14, col_offset=7, end_lineno=14, end_col_offset=24):
                   ('pandas.core.frame', 'dropna'),
                   CodeReference(lineno=16, col_offset=38, end_lineno=16, end_col_offset=61):
                   ('pandas.core.frame', '__getitem__', 'Projection'),
                   CodeReference(lineno=16, col_offset=9, end_lineno=16, end_col_offset=89):
                   ('sklearn.preprocessing._label', 'label_binarize'),
                   CodeReference(lineno=19, col_offset=20, end_lineno=19, end_col_offset=72):
                   ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                   CodeReference(lineno=20, col_offset=16, end_lineno=20, end_col_offset=46):
                   ('sklearn.preprocessing._data', 'StandardScaler'),
                   CodeReference(lineno=18, col_offset=25, end_lineno=21, end_col_offset=2):
                   ('sklearn.compose._column_transformer', 'ColumnTransformer'),
                   CodeReference(lineno=26, col_offset=19, end_lineno=26, end_col_offset=48):
                   ('sklearn.tree._classes', 'DecisionTreeClassifier'),
                   CodeReference(lineno=24, col_offset=18, end_lineno=26, end_col_offset=51):
                   ('sklearn.pipeline', 'Pipeline'),
                   CodeReference(lineno=28, col_offset=0, end_lineno=28, end_col_offset=33):
                   ('sklearn.pipeline', 'fit'),
                   CodeReference(lineno=31, col_offset=0, end_lineno=31, end_col_offset=26):
                   ('builtins', 'print')}

    return module_info


def get_call_description_info():
    """
    Get the module info for the adult_easy pipeline
    """
    call_description_info = {CodeReference(lineno=12, col_offset=11, end_lineno=12, end_col_offset=62):
                             'adult_train.csv',
                             CodeReference(lineno=14, col_offset=7, end_lineno=14, end_col_offset=24):
                             'dropna',
                             CodeReference(lineno=16, col_offset=38, end_lineno=16, end_col_offset=61):
                             "to ['income-per-year']",
                             CodeReference(lineno=16, col_offset=9, end_lineno=16, end_col_offset=89):
                             "label_binarize, classes: ['>50K', '<=50K']",
                             CodeReference(lineno=19, col_offset=20, end_lineno=19, end_col_offset=72):
                             'Categorical Encoder (OneHotEncoder)',
                             CodeReference(lineno=20, col_offset=16, end_lineno=20, end_col_offset=46):
                             'Numerical Encoder (StandardScaler)',
                             CodeReference(lineno=26, col_offset=19, end_lineno=26, end_col_offset=48):
                             'Decision Tree'}

    return call_description_info


def get_adult_simple_py_ast():
    """
    Get the parsed AST for the adult_easy pipeline
    """
    with open(ADULT_SIMPLE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
    return test_ast


def get_test_wir():
    """
    Get the extracted WIR for the adult_easy pipeline with runtime info
    """
    test_ast = get_adult_simple_py_ast()
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

            train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
            raw_data = pd.read_csv(train_file)
            data = raw_data.dropna()
            relevant_data = data[['age', 'workclass', 'education']]
            """)
    return code


def run_random_annotation_testing_analyzer(code):
    """
    An utility function to test backends
    """
    result = PipelineInspector \
        .on_pipeline_from_string(code) \
        .add_required_inspection(RandomAnnotationTestingInspection(10)) \
        .execute()
    inspection_results = result.inspection_to_annotations
    assert RandomAnnotationTestingInspection(10) in inspection_results
    random_annotation_analyzer_result = inspection_results[RandomAnnotationTestingInspection(10)]
    return random_annotation_analyzer_result


def run_row_index_annotation_testing_analyzer(code):
    """
    An utility function to test backends
    """
    result = PipelineInspector \
        .on_pipeline_from_string(code) \
        .add_required_inspection(RowLineage(10)) \
        .execute()
    inspection_results = result.inspection_to_annotations
    assert RowLineage(10) in inspection_results
    result = inspection_results[RowLineage(10)]
    return result


def run_multiple_test_analyzers(code):
    """
   An utility function to test backends.
   Also useful to debug annotation propagation.
   """
    analyzers = [RandomAnnotationTestingInspection(2), MaterializeFirstOutputRows(5),
                 RowLineage(2)]
    result = PipelineInspector \
        .on_pipeline_from_string(code) \
        .add_required_inspections(analyzers) \
        .execute()
    inspection_results = result.inspection_to_annotations
    return inspection_results, analyzers


def run_and_assert_all_op_outputs_inspected(py_file_path, sensitive_columns, dag_png_path):
    """
    Execute the pipeline with a few checks and inspections.
    Assert that mlinspect properly lets inspections inspect all DAG nodes
    """

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(py_file_path) \
        .add_check(NoBiasIntroducedFor(sensitive_columns)) \
        .add_check(NoIllegalFeatures()) \
        .add_required_inspection(MissingEmbeddings(20)) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .execute()
    materialize_output = inspector_result.inspection_to_annotations[MaterializeFirstOutputRows(5)]
    assert len(materialize_output) == (len(inspector_result.dag.nodes) - 1)  # Estimator does not have output

    save_fig_to_path(inspector_result.dag, dag_png_path)
    assert os.path.isfile(dag_png_path)
