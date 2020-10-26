"""
Tests whether the MaterializeFirstOutputRows works
"""

from numpy.ma import array
from pandas import DataFrame
from pandas._testing import assert_frame_equal
from testfixtures import compare, RangeComparison

from example_pipelines import ADULT_SIMPLE_PY
from mlinspect.inspections._materialize_first_output_rows import MaterializeFirstOutputRows
from mlinspect.instrumentation._dag_node import DagNode, OperatorType, CodeReference
from mlinspect._pipeline_inspector import PipelineInspector


def test_materialize_first_rows_inspection():
    """
    Tests whether the MaterializeFirstOutputRows works
    """
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY) \
        .add_required_inspection(MaterializeFirstOutputRows(2)) \
        .execute()
    inspection_result = inspector_result.inspection_to_annotations
    assert MaterializeFirstOutputRows(2) in inspection_result
    result = inspection_result[MaterializeFirstOutputRows(2)]

    assert_df_dicts_equal(result, get_expected_result())


def assert_df_dicts_equal(dict1, dict2):
    """
    Tests whether the two dicts are equal. Data frame equality testing sadly requires this extra function.
    """
    key1 = dict1.keys()
    key2 = dict2.keys()
    compare(key1, key2)
    for key, df1 in dict1.items():
        df2 = dict2[key]
        if isinstance(df1, DataFrame) and isinstance(df2, DataFrame):
            assert_frame_equal(df1, df2)
        else:
            compare(df1, df2)


def get_expected_result():
    """
    Get the expected PrintFirstRowsAnalyzer(2) result for the adult_easy example
    """
    expected_result = {
        DagNode(node_id=18, operator_type=OperatorType.DATA_SOURCE, module=('pandas.io.parsers', 'read_csv'),
                code_reference=CodeReference(12, 11, 12, 62), description='adult_train.csv',
                columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                         'hours-per-week', 'native-country', 'income-per-year']):
            DataFrame([[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                        'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                       [29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                        'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K']],
                      columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                               'hours-per-week', 'native-country', 'income-per-year']),
        DagNode(node_id=20, operator_type=OperatorType.SELECTION, module=('pandas.core.frame', 'dropna'),
                code_reference=CodeReference(14, 7, 14, 24), description='dropna',
                columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                         'marital-status', 'occupation', 'relationship', 'race', 'sex',
                         'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'income-per-year']
                ):
            DataFrame([[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                        'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                       [29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                        'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K']],
                      columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                               'marital-status', 'occupation', 'relationship', 'race', 'sex',
                               'capital-gain', 'capital-loss', 'hours-per-week',
                               'native-country', 'income-per-year']),
        DagNode(node_id=23, operator_type=OperatorType.PROJECTION, module=('pandas.core.frame', '__getitem__',
                                                                           'Projection'),
                code_reference=CodeReference(16, 38, 16, 61), description="to ['income-per-year']",
                columns=['income-per-year']):
            DataFrame([['<=50K'], ['<=50K']], columns=['array']),
        DagNode(node_id=28, operator_type=OperatorType.PROJECTION_MODIFY,
                module=('sklearn.preprocessing._label', 'label_binarize'),
                code_reference=CodeReference(16, 9, 16, 89),
                description="label_binarize, classes: ['>50K', '<=50K']", columns=['array']):
            DataFrame([[array(1)], [array(1)]], columns=['array']),
        DagNode(node_id=56, operator_type=OperatorType.TRAIN_DATA, module=('sklearn.pipeline', 'fit', 'Train Data'),
                code_reference=CodeReference(24, 18, 26, 51), description=None,
                columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                         'hours-per-week', 'native-country', 'income-per-year']):
            DataFrame([
                [46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                 'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                [29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                 'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K']],
                      columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                               'hours-per-week', 'native-country', 'income-per-year']),
        DagNode(node_id=56, operator_type=OperatorType.TRAIN_LABELS, module=('sklearn.pipeline', 'fit', 'Train Labels'),
                code_reference=CodeReference(24, 18, 26, 51), description=None, columns=['array']):
            DataFrame([[array(1)], [array(1)]], columns=['array']),
        DagNode(node_id=40, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25, 21, 2),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['age'] (ColumnTransformer)", columns=['age']):
            DataFrame([[46], [29]], columns=['age']),
        DagNode(node_id=34, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25, 21, 2),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['education'] (ColumnTransformer)", columns=['education']):
            DataFrame([['Some-college'], ['Some-college']], columns=['education']),
        DagNode(node_id=41, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25, 21, 2),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['hours-per-week'] (ColumnTransformer)", columns=['hours-per-week']):
            DataFrame([[40], [50]], columns=['hours-per-week']),
        DagNode(node_id=35, operator_type=OperatorType.PROJECTION, code_reference=CodeReference(18, 25, 21, 2),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Projection'),
                description="to ['workclass'] (ColumnTransformer)", columns=['workclass']):
            DataFrame([['Private'], ['Local-gov']], columns=['workclass']),
        DagNode(node_id=40, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(20, 16, 20, 46),
                module=('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                description="Numerical Encoder (StandardScaler), Column: 'age'", columns=['age']):
            DataFrame([[array(RangeComparison(0.5, 0.6))], [array(RangeComparison(-0.8, -0.7))]], columns=['age']),
        DagNode(node_id=41, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(20, 16, 20, 46),
                module=('sklearn.preprocessing._data', 'StandardScaler', 'Pipeline'),
                description="Numerical Encoder (StandardScaler), Column: 'hours-per-week'", columns=['hours-per-week']):
            DataFrame([[array(RangeComparison(-0.09, -0.08))], [array(RangeComparison(0.7, 0.8))]],
                      columns=['hours-per-week']),
        DagNode(node_id=34, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(19, 20, 19, 72),
                module=('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                description="Categorical Encoder (OneHotEncoder), Column: 'education'", columns=['education']):
            DataFrame([[array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])],
                       [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])]],
                      columns=['education']),
        DagNode(node_id=35, operator_type=OperatorType.TRANSFORMER, code_reference=CodeReference(19, 20, 19, 72),
                module=('sklearn.preprocessing._encoders', 'OneHotEncoder', 'Pipeline'),
                description="Categorical Encoder (OneHotEncoder), Column: 'workclass'", columns=['workclass']):
            DataFrame([[array([0., 0., 1., 0., 0., 0., 0.])], [array([0., 1., 0., 0., 0., 0., 0.])]],
                      columns=['workclass']),
        DagNode(node_id=46, operator_type=OperatorType.CONCATENATION, code_reference=CodeReference(18, 25, 21, 2),
                module=('sklearn.compose._column_transformer', 'ColumnTransformer', 'Concatenation'),
                description=None, columns=['array']):
            DataFrame([[array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
                               0., 0., 0., RangeComparison(0.5, 0.6),
                               RangeComparison(-0.09, -0.08)])],
                       [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
                               0., 0., 0., RangeComparison(-0.8, -0.7),
                               RangeComparison(0.7, 0.8)])]],
                      columns=['array']),
        DagNode(node_id=51, operator_type=OperatorType.ESTIMATOR, code_reference=CodeReference(26, 19, 26, 48),
                module=('sklearn.tree._classes', 'DecisionTreeClassifier', 'Pipeline'),
                description='Decision Tree'): None
    }
    return expected_result
