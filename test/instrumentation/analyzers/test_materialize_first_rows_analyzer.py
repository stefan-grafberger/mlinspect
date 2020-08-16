"""
Tests whether the MaterializeFirstRowsAnalyzer works
"""

import os

from testfixtures import compare, RangeComparison
from numpy.ma import array

from mlinspect.instrumentation.analyzers.analyzer_input import AnalyzerInputRow
from mlinspect.instrumentation.analyzers.materialize_first_rows_analyzer import MaterializeFirstRowsAnalyzer
from mlinspect.instrumentation.dag_node import DagNode, OperatorType, CodeReference
from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.utils import get_project_root

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_materialize_first_rows_analyzer():
    """
    Tests whether the MaterializeFirstRowsAnalyzer works
    """
    inspection_result = PipelineInspector \
        .on_pipeline_from_py_file(FILE_PY) \
        .add_analyzer(MaterializeFirstRowsAnalyzer(2)) \
        .execute()
    analyzer_results = inspection_result.analyzer_to_annotations
    assert MaterializeFirstRowsAnalyzer(2) in analyzer_results
    result = analyzer_results[MaterializeFirstRowsAnalyzer(2)]

    compare(result, get_expected_result())


def get_expected_result():
    """
    Get the expected PrintFirstRowsAnalyzer(2) result for the adult_easy example
    """
    expected_result = {
        DagNode(node_id=18, operator_type=OperatorType.DATA_SOURCE, module=('pandas.io.parsers', 'read_csv'),
                code_reference=CodeReference(lineno=12, col_offset=11), description='adult_train.csv'): [
                    AnalyzerInputRow(
                        values=[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                                'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                        fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                'hours-per-week', 'native-country', 'income-per-year']),
                    AnalyzerInputRow(
                        values=[29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                                'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K'],
                        fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                'marital-status', 'occupation', 'relationship', 'race',
                                'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                'native-country', 'income-per-year'])],
        DagNode(node_id=20, operator_type=OperatorType.SELECTION, module=('pandas.core.frame', 'dropna'),
                code_reference=CodeReference(lineno=14, col_offset=7), description='dropna'): [
                    AnalyzerInputRow(
                        values=[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                                'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                        fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                'capital-gain', 'capital-loss', 'hours-per-week',
                                'native-country', 'income-per-year']),
                    AnalyzerInputRow(
                        values=[29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                                'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K'],
                        fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                'capital-gain', 'capital-loss', 'hours-per-week',
                                'native-country', 'income-per-year'])],
        DagNode(node_id=23, operator_type=OperatorType.PROJECTION, module=('pandas.core.frame', '__getitem__'),
                code_reference=CodeReference(lineno=16, col_offset=38), description="to ['income-per-year']"): [
                    AnalyzerInputRow(values=['<=50K'], fields=['income-per-year']),
                    AnalyzerInputRow(values=['<=50K'], fields=['income-per-year'])],
        DagNode(node_id=28, operator_type=OperatorType.PROJECTION_MODIFY,
                module=('sklearn.preprocessing._label', 'label_binarize'),
                code_reference=CodeReference(lineno=16, col_offset=9),
                description="label_binarize, classes: ['>50K', '<=50K']"): [
                    AnalyzerInputRow(values=[array(1)], fields=['array']),
                    AnalyzerInputRow(values=[array(1)], fields=['array'])],
        DagNode(node_id=56, operator_type=OperatorType.TRAIN_DATA, module=('sklearn.pipeline', 'fit', 'Train Data'),
                code_reference=CodeReference(lineno=24, col_offset=18), description=None): [
                    AnalyzerInputRow(
                        values=[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                                'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K', 1],
                        fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                'hours-per-week', 'native-country', 'income-per-year', 'mlinspect_index']),
                    AnalyzerInputRow(
                        values=[29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                                'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K', 2],
                        fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                'hours-per-week', 'native-country', 'income-per-year', 'mlinspect_index'])],
        DagNode(node_id=56, operator_type=OperatorType.TRAIN_LABELS, module=('sklearn.pipeline', 'fit', 'Train Labels'),
                code_reference=CodeReference(lineno=24, col_offset=18), description=None): [
                    AnalyzerInputRow(values=[array(1)], fields=['array']),
                    AnalyzerInputRow(values=[array(1)], fields=['array'])],
        DagNode(node_id=40, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['age']"): [
                    AnalyzerInputRow(values=[46], fields=['age']), AnalyzerInputRow(values=[29], fields=['age'])],
        DagNode(node_id=34, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['education']"): [
                    AnalyzerInputRow(values=['Some-college'], fields=['education']),
                    AnalyzerInputRow(values=['Some-college'], fields=['education'])],
        DagNode(node_id=41, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['hours-per-week']"): [
                    AnalyzerInputRow(values=[40], fields=['hours-per-week']),
                    AnalyzerInputRow(values=[50], fields=['hours-per-week'])],
        DagNode(node_id=35, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['workclass']"): [
                    AnalyzerInputRow(values=['Private'], fields=['workclass']),
                    AnalyzerInputRow(values=['Local-gov'], fields=['workclass'])],
        DagNode(node_id=40, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(20, 16),
                module=('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                description="Numerical Encoder (StandardScaler), Column: 'age'"): [
                    AnalyzerInputRow(values=[array(RangeComparison(0.5, 0.6))], fields=['array']),
                    AnalyzerInputRow(values=[array(RangeComparison(-0.8, -0.7))], fields=['array'])],
        DagNode(node_id=41, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(20, 16),
                module=('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                description="Numerical Encoder (StandardScaler), Column: 'hours-per-week'"): [
                    AnalyzerInputRow(values=[array(RangeComparison(-0.09, -0.08))], fields=['array']),
                    AnalyzerInputRow(values=[array(RangeComparison(0.7, 0.8))], fields=['array'])],
        DagNode(node_id=34, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(19, 20),
                module=('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                description="Categorical Encoder (OneHotEncoder), Column: 'education'"): [
                    AnalyzerInputRow(
                        values=[array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])],
                        fields=['array']),
                    AnalyzerInputRow(
                        values=[array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])],
                        fields=['array'])],
        DagNode(node_id=35, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(19, 20),
                module=('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                description="Categorical Encoder (OneHotEncoder), Column: 'workclass'"): [
                    AnalyzerInputRow(values=[array([0., 0., 1., 0., 0., 0., 0.])], fields=['array']),
                    AnalyzerInputRow(values=[array([0., 1., 0., 0., 0., 0., 0.])], fields=['array'])],
        DagNode(node_id=46, operator_type=OperatorType.CONCATENATION, code_reference=CodeReference(18, 25),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'),
                description=None): [
                    AnalyzerInputRow(values=[array([0., 0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.,
                                                    0., 0., 0., 0., 0., 1.,  0.,  0.,  1.,  0.,
                                                    0., 0., 0., 0.57272928, -0.08076598])],
                                     fields=['array']),
                    AnalyzerInputRow(values=[array([0., 0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.,
                                                    0., 0., 0., 0., 0., 1.,  0.,  1.,  0.,  0.,
                                                    0., 0., 0., -0.72248831, 0.75246815])],
                                     fields=['array'])]

    }
    return expected_result
