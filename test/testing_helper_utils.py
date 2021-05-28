"""
Some util functions used in other tests
"""
import ast
import os
from inspect import cleandoc

import networkx
from testfixtures import RangeComparison

from demo.feature_overview.missing_embeddings import MissingEmbeddings
from example_pipelines._pipelines import ADULT_SIMPLE_PY
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.checks._no_bias_introduced_for import NoBiasIntroducedFor
from mlinspect.checks._no_illegal_features import NoIllegalFeatures
from mlinspect.inspections._lineage import RowLineage
from mlinspect.inspections._materialize_first_output_rows import MaterializeFirstOutputRows
from mlinspect.instrumentation._dag_node import DagNode, OperatorType, CodeReference
from mlinspect.visualisation._visualisation import save_fig_to_path
from test.backends.random_annotation_testing_inspection import RandomAnnotationTestingInspection


def get_expected_dag_adult_easy(caller_filename: str, line_offset: int = 0):
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    # The line numbers differ slightly between the .py file and the.ipynb file
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(node_id=0, caller_filename=caller_filename, lineno=RangeComparison(12, 18),
                                   operator_type=OperatorType.DATA_SOURCE,
                                   module=('pandas.io.parsers', 'read_csv'),
                                   description='adult_train.csv',
                                   columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                            'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                                            'income-per-year'],
                                   optional_code_reference=CodeReference(12 + line_offset, 11,
                                                                         12 + line_offset, 62),
                                   optional_source_code="pd.read_csv(train_file, na_values='?', index_col=0)")
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(node_id=1, caller_filename=caller_filename, lineno=14 + line_offset,
                              operator_type=OperatorType.SELECTION, module=('pandas.core.frame', 'dropna'),
                              description='dropna',
                              columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                       'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                       'hours-per-week', 'native-country', 'income-per-year'],
                              optional_code_reference=CodeReference(lineno=14 + line_offset, col_offset=7,
                                                                    end_lineno=14 + line_offset,
                                                                    end_col_offset=24),
                              optional_source_code='raw_data.dropna()')
    expected_graph.add_edge(expected_data_source, expected_select)

    pipeline_str = "compose.ColumnTransformer(transformers=[\n" \
                   "    ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), " \
                   "['education', 'workclass']),\n" \
                   "    ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])\n" \
                   "])"
    expected_pipeline_project_one = DagNode(node_id=8, caller_filename=caller_filename,
                                            lineno=18 + line_offset, operator_type=OperatorType.PROJECTION,
                                            module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                            description="to ['education', 'workclass']",
                                            columns=['education', 'workclass'],
                                            optional_code_reference=CodeReference(lineno=18 + line_offset,
                                                                                  col_offset=25,
                                                                                  end_lineno=21 + line_offset,
                                                                                  end_col_offset=2),
                                            optional_source_code=pipeline_str)
    expected_graph.add_edge(expected_select, expected_pipeline_project_one)
    expected_pipeline_project_two = DagNode(node_id=9, caller_filename=caller_filename,
                                            lineno=18 + line_offset, operator_type=OperatorType.PROJECTION,
                                            module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                            description="to ['age', 'hours-per-week']",
                                            columns=['age', 'hours-per-week'],
                                            optional_code_reference=CodeReference(lineno=18 + line_offset,
                                                                                  col_offset=25,
                                                                                  end_lineno=21 + line_offset,
                                                                                  end_col_offset=2),
                                            optional_source_code=pipeline_str)
    expected_graph.add_edge(expected_select, expected_pipeline_project_two)

    expected_pipeline_transformer_one = DagNode(node_id=4, caller_filename=caller_filename,
                                                lineno=19 + line_offset, operator_type=OperatorType.TRANSFORMER,
                                                module=('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                                                description='One-Hot Encoder', columns=['array'],
                                                optional_code_reference=CodeReference(lineno=19 + line_offset,
                                                                                      col_offset=20,
                                                                                      end_lineno=19 + line_offset,
                                                                                      end_col_offset=72),
                                                optional_source_code="preprocessing."
                                                                     "OneHotEncoder(handle_unknown='ignore')")
    expected_pipeline_transformer_two = DagNode(node_id=5, caller_filename=caller_filename,
                                                lineno=20 + line_offset, operator_type=OperatorType.TRANSFORMER,
                                                module=('sklearn.preprocessing._data', 'StandardScaler'),
                                                description='Standard Scaler', columns=['array'],
                                                optional_code_reference=CodeReference(lineno=20 + line_offset,
                                                                                      col_offset=16,
                                                                                      end_lineno=20 + line_offset,
                                                                                      end_col_offset=46),
                                                optional_source_code='preprocessing.StandardScaler()')
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one)
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two)

    expected_pipeline_concatenation = DagNode(node_id=6, caller_filename=caller_filename,
                                              lineno=18 + line_offset, operator_type=OperatorType.CONCATENATION,
                                              module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                              description='', columns=['array'],
                                              optional_code_reference=CodeReference(lineno=18 + line_offset,
                                                                                    col_offset=25,
                                                                                    end_lineno=21 + line_offset,
                                                                                    end_col_offset=2),
                                              optional_source_code=pipeline_str)
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation)

    expected_train_data = DagNode(node_id=10, caller_filename=caller_filename, lineno=26 + line_offset,
                                  operator_type=OperatorType.TRAIN_DATA,
                                  module=('sklearn.tree._classes', 'DecisionTreeClassifier'), description='Train Data',
                                  columns=['array'],
                                  optional_code_reference=CodeReference(lineno=26 + line_offset, col_offset=19,
                                                                        end_lineno=26 + line_offset,
                                                                        end_col_offset=48),
                                  optional_source_code='tree.DecisionTreeClassifier()')
    expected_graph.add_edge(expected_pipeline_concatenation, expected_train_data)

    expected_project = DagNode(node_id=2, caller_filename=caller_filename, lineno=16 + line_offset,
                               operator_type=OperatorType.PROJECTION, module=('pandas.core.frame', '__getitem__'),
                               description="to ['income-per-year']", columns=['income-per-year'],
                               optional_code_reference=CodeReference(lineno=16 + line_offset, col_offset=38,
                                                                     end_lineno=16 + line_offset,
                                                                     end_col_offset=61),
                               optional_source_code="data['income-per-year']")
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagNode(node_id=3, caller_filename=caller_filename,
                                      lineno=16 + line_offset,
                                      operator_type=OperatorType.PROJECTION_MODIFY,
                                      module=('sklearn.preprocessing._label', 'label_binarize'),
                                      description="label_binarize, classes: ['>50K', '<=50K']", columns=['array'],
                                      optional_code_reference=CodeReference(lineno=16 + line_offset,
                                                                            col_offset=9,
                                                                            end_lineno=16 + line_offset,
                                                                            end_col_offset=89),
                                      optional_source_code="preprocessing.label_binarize(data['income-per-year'], "
                                                           "classes=['>50K', '<=50K'])")
    expected_graph.add_edge(expected_project, expected_project_modify)

    expected_train_labels = DagNode(node_id=11, caller_filename=caller_filename,
                                    lineno=26 + line_offset,
                                    operator_type=OperatorType.TRAIN_LABELS,
                                    module=('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                    description='Train Labels', columns=['array'],
                                    optional_code_reference=CodeReference(lineno=26 + line_offset,
                                                                          col_offset=19,
                                                                          end_lineno=26 + line_offset,
                                                                          end_col_offset=48),
                                    optional_source_code='tree.DecisionTreeClassifier()')
    expected_graph.add_edge(expected_project_modify, expected_train_labels)

    expected_estimator = DagNode(node_id=7, caller_filename=caller_filename,
                                 lineno=26 + line_offset, operator_type=OperatorType.ESTIMATOR,
                                 module=('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                 description='Decision Tree', columns=[],
                                 optional_code_reference=CodeReference(lineno=26 + line_offset, col_offset=19,
                                                                       end_lineno=26 + line_offset,
                                                                       end_col_offset=48),
                                 optional_source_code='tree.DecisionTreeClassifier()')
    expected_graph.add_edge(expected_train_data, expected_estimator)
    expected_graph.add_edge(expected_train_labels, expected_estimator)

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


def get_call_source_code_info():
    """
    Get the source code info for the adult_easy pipeline
    """
    call_source_code_info = {
        CodeReference(lineno=12, col_offset=11, end_lineno=12, end_col_offset=62):
            "pd.read_csv(train_file, na_values='?', index_col=0)",
        CodeReference(lineno=14, col_offset=7, end_lineno=14, end_col_offset=24):
            'raw_data.dropna()',
        CodeReference(lineno=16, col_offset=38, end_lineno=16, end_col_offset=61): "data['income-per-year']",
        CodeReference(lineno=16, col_offset=9, end_lineno=16, end_col_offset=89):
            "preprocessing.label_binarize(data['income-per-year'], classes=['>50K', '<=50K'])",
        CodeReference(lineno=19, col_offset=20, end_lineno=19, end_col_offset=72):
            "preprocessing.OneHotEncoder(handle_unknown='ignore')",
        CodeReference(lineno=20, col_offset=16, end_lineno=20, end_col_offset=46):
            'preprocessing.StandardScaler()',
        CodeReference(lineno=18, col_offset=25, end_lineno=21, end_col_offset=2):
            "compose.ColumnTransformer(transformers=[\n"
            "    ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), ['education', 'workclass']),\n"
            "    ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])\n])",
        CodeReference(lineno=26, col_offset=19, end_lineno=26, end_col_offset=48): 'tree.DecisionTreeClassifier()',
        CodeReference(lineno=24, col_offset=18, end_lineno=26, end_col_offset=51):
            "pipeline.Pipeline([\n"
            "    ('features', feature_transformation),\n"
            "    ('classifier', tree.DecisionTreeClassifier())])",
        CodeReference(lineno=28, col_offset=0, end_lineno=28, end_col_offset=33): 'income_pipeline.fit(data, labels)'}

    return call_source_code_info


def get_adult_simple_py_ast():
    """
    Get the parsed AST for the adult_easy pipeline
    """
    with open(ADULT_SIMPLE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
    return test_ast


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
