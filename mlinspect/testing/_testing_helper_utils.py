"""
Some util functions used in other tests
"""
import os
from inspect import cleandoc

import networkx
from pandas import DataFrame

from demo.feature_overview.missing_embeddings import MissingEmbeddings
from mlinspect import OperatorContext, FunctionInfo, OperatorType
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.checks import SimilarRemovalProbabilitiesFor
from mlinspect.checks._no_bias_introduced_for import NoBiasIntroducedFor
from mlinspect.checks._no_illegal_features import NoIllegalFeatures
from mlinspect.inspections import HistogramForColumns, IntersectionalHistogramForColumns, CompletenessOfColumns, \
    CountDistinctOfColumns, ColumnPropagation
from mlinspect.inspections._lineage import RowLineage
from mlinspect.inspections._materialize_first_output_rows import MaterializeFirstOutputRows
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.testing._random_annotation_testing_inspection import RandomAnnotationTestingInspection
from mlinspect.visualisation._visualisation import save_fig_to_path


def get_expected_dag_adult_easy(caller_filename: str, line_offset: int = 0, with_code_references=True):
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    # The line numbers differ slightly between the .py file and the.ipynb file
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(0,
                                   BasicCodeLocation(caller_filename, 12 + line_offset),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.io.parsers', 'read_csv')),
                                   DagNodeDetails('adult_train.csv',
                                                  ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                                   'marital-status', 'occupation', 'relationship', 'race',
                                                   'sex',
                                                   'capital-gain', 'capital-loss', 'hours-per-week',
                                                   'native-country',
                                                   'income-per-year']),
                                   OptionalCodeInfo(CodeReference(12 + line_offset, 11, 12 + line_offset, 62),
                                                    "pd.read_csv(train_file, na_values='?', index_col=0)"))
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(1,
                              BasicCodeLocation(caller_filename, 14 + line_offset),
                              OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                              DagNodeDetails('dropna',
                                             ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                              'marital-status',
                                              'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                              'capital-loss',
                                              'hours-per-week', 'native-country', 'income-per-year']),
                              OptionalCodeInfo(CodeReference(14 + line_offset, 7, 14 + line_offset, 24),
                                               'raw_data.dropna()'))
    expected_graph.add_edge(expected_data_source, expected_select)

    pipeline_str = "compose.ColumnTransformer(transformers=[\n" \
                   "    ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), " \
                   "['education', 'workclass']),\n" \
                   "    ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])\n" \
                   "])"
    expected_pipeline_project_one = DagNode(4,
                                            BasicCodeLocation(caller_filename, 18 + line_offset),
                                            OperatorContext(OperatorType.PROJECTION,
                                                            FunctionInfo('sklearn.compose._column_transformer',
                                                                         'ColumnTransformer')),
                                            DagNodeDetails("to ['education', 'workclass']", ['education', 'workclass']),
                                            OptionalCodeInfo(CodeReference(18 + line_offset, 25, 21 + line_offset, 2),
                                                             pipeline_str))
    expected_graph.add_edge(expected_select, expected_pipeline_project_one)
    expected_pipeline_project_two = DagNode(6,
                                            BasicCodeLocation(caller_filename, 18 + line_offset),
                                            OperatorContext(OperatorType.PROJECTION,
                                                            FunctionInfo('sklearn.compose._column_transformer',
                                                                         'ColumnTransformer')),
                                            DagNodeDetails("to ['age', 'hours-per-week']", ['age', 'hours-per-week']),
                                            OptionalCodeInfo(CodeReference(18 + line_offset, 25, 21 + line_offset, 2),
                                                             pipeline_str))
    expected_graph.add_edge(expected_select, expected_pipeline_project_two)

    expected_pipeline_transformer_one = DagNode(5,
                                                BasicCodeLocation(caller_filename, 19 + line_offset),
                                                OperatorContext(OperatorType.TRANSFORMER,
                                                                FunctionInfo('sklearn.preprocessing._encoders',
                                                                             'OneHotEncoder')),
                                                DagNodeDetails('One-Hot Encoder: fit_transform', ['array']),
                                                OptionalCodeInfo(CodeReference(19 + line_offset, 20, 19 + line_offset,
                                                                               72),
                                                                 "preprocessing.OneHotEncoder(handle_unknown='ignore')")
                                                )
    expected_pipeline_transformer_two = DagNode(7,
                                                BasicCodeLocation(caller_filename, 20 + line_offset),
                                                OperatorContext(OperatorType.TRANSFORMER,
                                                                FunctionInfo('sklearn.preprocessing._data',
                                                                             'StandardScaler')),
                                                DagNodeDetails('Standard Scaler: fit_transform', ['array']),
                                                OptionalCodeInfo(CodeReference(20 + line_offset, 16, 20 + line_offset,
                                                                               46),
                                                                 'preprocessing.StandardScaler()'))
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one)
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two)

    expected_pipeline_concatenation = DagNode(8,
                                              BasicCodeLocation(caller_filename, 18 + line_offset),
                                              OperatorContext(OperatorType.CONCATENATION,
                                                              FunctionInfo('sklearn.compose._column_transformer',
                                                                           'ColumnTransformer')),
                                              DagNodeDetails(None, ['array']),
                                              OptionalCodeInfo(CodeReference(18 + line_offset, 25, 21 + line_offset, 2),
                                                               pipeline_str))
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation)

    expected_train_data = DagNode(9,
                                  BasicCodeLocation(caller_filename, 26 + line_offset),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails(None, ['array']),
                                  OptionalCodeInfo(CodeReference(26 + line_offset, 19, 26 + line_offset, 48),
                                                   'tree.DecisionTreeClassifier()'))
    expected_graph.add_edge(expected_pipeline_concatenation, expected_train_data)

    expected_project = DagNode(2,
                               BasicCodeLocation(caller_filename, 16 + line_offset),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['income-per-year']", ['income-per-year']),
                               OptionalCodeInfo(CodeReference(16 + line_offset, 38, 16 + line_offset, 61),
                                                "data['income-per-year']"))
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagNode(3,
                                      BasicCodeLocation(caller_filename, 16 + line_offset),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                      FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                      DagNodeDetails("label_binarize, classes: ['>50K', '<=50K']", ['array']),
                                      OptionalCodeInfo(CodeReference(16 + line_offset, 9, 16 + line_offset, 89),
                                                       "preprocessing.label_binarize(data['income-per-year'], "
                                                       "classes=['>50K', '<=50K'])"))
    expected_graph.add_edge(expected_project, expected_project_modify)

    expected_train_labels = DagNode(10,
                                    BasicCodeLocation(caller_filename, 26 + line_offset),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                    DagNodeDetails(None, ['array']),
                                    OptionalCodeInfo(CodeReference(26 + line_offset, 19, 26 + line_offset, 48),
                                                     'tree.DecisionTreeClassifier()'))
    expected_graph.add_edge(expected_project_modify, expected_train_labels)

    expected_estimator = DagNode(11,
                                 BasicCodeLocation(caller_filename, 26 + line_offset),
                                 OperatorContext(OperatorType.ESTIMATOR,
                                                 FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                 DagNodeDetails('Decision Tree', []),
                                 OptionalCodeInfo(CodeReference(26 + line_offset, 19, 26 + line_offset, 48),
                                                  'tree.DecisionTreeClassifier()'))
    expected_graph.add_edge(expected_train_data, expected_estimator)
    expected_graph.add_edge(expected_train_labels, expected_estimator)

    if not with_code_references:
        for dag_node in expected_graph.nodes:
            dag_node.optional_code_info = None

    return expected_graph


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
    inspection_results = result.dag_node_to_inspection_results
    dag_node_to_random_inspection = {}
    for dag_node, inspection_result in inspection_results.items():
        assert RandomAnnotationTestingInspection(10) in inspection_result
        dag_node_to_random_inspection[dag_node] = inspection_result[RandomAnnotationTestingInspection(10)]
    return dag_node_to_random_inspection


def run_row_index_annotation_testing_analyzer(code):
    """
    An utility function to test backends
    """
    result = PipelineInspector \
        .on_pipeline_from_string(code) \
        .add_required_inspection(RowLineage(10)) \
        .execute()
    inspection_results = result.dag_node_to_inspection_results
    dag_node_to_lineage_inspection = {}
    for dag_node, inspection_result in inspection_results.items():
        assert RowLineage(10) in inspection_result
        dag_node_to_lineage_inspection[dag_node] = inspection_result[RowLineage(10)]
    return dag_node_to_lineage_inspection


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
    dag_node_to_inspection_results = result.dag_node_to_inspection_results

    return dag_node_to_inspection_results, analyzers


def run_and_assert_all_op_outputs_inspected(py_file_path, sensitive_columns, dag_png_path, custom_monkey_patching=None):
    """
    Execute the pipeline with a few checks and inspections.
    Assert that mlinspect properly lets inspections inspect all DAG nodes
    """
    if custom_monkey_patching is None:
        custom_monkey_patching = []

    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(py_file_path) \
        .add_check(NoBiasIntroducedFor(sensitive_columns)) \
        .add_check(NoIllegalFeatures()) \
        .add_check(SimilarRemovalProbabilitiesFor(sensitive_columns)) \
        .add_required_inspection(MissingEmbeddings(20)) \
        .add_required_inspection(RowLineage(5)) \
        .add_required_inspection(MaterializeFirstOutputRows(5)) \
        .add_required_inspection(IntersectionalHistogramForColumns(sensitive_columns)) \
        .add_required_inspection(ColumnPropagation(sensitive_columns, 5)) \
        .add_required_inspection(CompletenessOfColumns(sensitive_columns)) \
        .add_required_inspection(CountDistinctOfColumns(sensitive_columns)) \
        .add_custom_monkey_patching_modules(custom_monkey_patching) \
        .execute()

    save_fig_to_path(inspector_result.dag, dag_png_path)
    assert os.path.isfile(dag_png_path)

    for dag_node, inspection_result in inspector_result.dag_node_to_inspection_results.items():
        assert dag_node.operator_info.operator != OperatorType.MISSING_OP
        assert MaterializeFirstOutputRows(5) in inspection_result
        assert RowLineage(5) in inspection_result
        assert MissingEmbeddings(20) in inspection_result
        assert HistogramForColumns(sensitive_columns) in inspection_result
        # Estimator and score do not have output
        if dag_node.operator_info.operator is not OperatorType.ESTIMATOR:
            assert inspection_result[MaterializeFirstOutputRows(5)] is not None
            assert inspection_result[RowLineage(5)] is not None
            assert inspection_result[HistogramForColumns(sensitive_columns)] is not None
            assert inspection_result[ColumnPropagation(sensitive_columns, 5)] is not None
            assert inspection_result[IntersectionalHistogramForColumns(sensitive_columns)] is not None
            assert inspection_result[CompletenessOfColumns(sensitive_columns)] is not None
            assert inspection_result[CountDistinctOfColumns(sensitive_columns)] is not None
        else:
            assert inspection_result[MaterializeFirstOutputRows(5)] is None
            assert inspection_result[RowLineage(5)] is not None
            assert inspection_result[ColumnPropagation(sensitive_columns, 5)] is not None
            assert inspection_result[HistogramForColumns(sensitive_columns)] is None
            assert inspection_result[IntersectionalHistogramForColumns(sensitive_columns)] is None
            assert inspection_result[CompletenessOfColumns(sensitive_columns)] is None
            assert inspection_result[CountDistinctOfColumns(sensitive_columns)] is None

    return inspector_result.dag


def black_box_df_op():
    """
    Black box operation returning a dataframe
    """
    pandas_df = DataFrame([0, 1, 2, 3, 4], columns=['A'])
    return pandas_df


def get_test_code_with_function_def_and_for_loop():
    """
    A simple code snippet with a pandas operation in a function def and then pandas calls in a loop
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
    return test_code
