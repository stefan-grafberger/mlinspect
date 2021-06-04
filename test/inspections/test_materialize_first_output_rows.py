"""
Tests whether MaterializeFirstOutputRows works
"""
import pandas
from numpy.ma import array
from pandas import DataFrame
from testfixtures import RangeComparison

from example_pipelines import ADULT_SIMPLE_PY
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections._materialize_first_output_rows import MaterializeFirstOutputRows


def test_materialize_first_rows_inspection():
    """
    Tests whether the MaterializeFirstOutputRows works
    """
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY) \
        .add_required_inspection(MaterializeFirstOutputRows(2)) \
        .execute()

    dag_node_to_inspection_results = list(inspector_result.dag_node_to_inspection_results.items())

    assert_output_looks_as_expected(dag_node_to_inspection_results)


def assert_output_looks_as_expected(dag_node_to_inspection_results):
    """
    Tests whether the output of MaterializeFirstOutputRows looks as expected for the adult_simple pipeline
    """
    assert dag_node_to_inspection_results[0][0].optional_code_info.source_code == \
           "pd.read_csv(train_file, na_values='?', index_col=0)"
    actual_df = dag_node_to_inspection_results[0][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                              'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                             [29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                              'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K']],
                            columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                     'hours-per-week', 'native-country', 'income-per-year'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))
    assert dag_node_to_inspection_results[1][0].optional_code_info.source_code == 'raw_data.dropna()'
    actual_df = dag_node_to_inspection_results[1][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                              'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                             [29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                              'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K']],
                            columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                     'capital-gain', 'capital-loss', 'hours-per-week',
                                     'native-country', 'income-per-year'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[2][0].optional_code_info.source_code == "data['income-per-year']"
    actual_df = dag_node_to_inspection_results[2][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([['<=50K'], ['<=50K']], columns=['income-per-year'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[3][0].optional_code_info.source_code == \
           "preprocessing.label_binarize(data['income-per-year'], classes=['>50K', '<=50K'])"
    actual_df = dag_node_to_inspection_results[3][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[array(1)], [array(1)]], columns=['array'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[4][0].code_location.lineno == 18
    actual_df = dag_node_to_inspection_results[4][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([['Some-college', 'Private'], ['Some-college', 'Local-gov']],
                            columns=['education', 'workclass'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[5][0].optional_code_info.source_code == \
           "preprocessing.OneHotEncoder(handle_unknown='ignore')"
    actual_df = dag_node_to_inspection_results[5][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[([array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
                                       0., 0., 0.])])],
                             [[array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
                                      0., 0., 0.])]]],
                            columns=['array'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[6][0].code_location.lineno == 18
    actual_df = dag_node_to_inspection_results[6][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[46, 40], [29, 50]], columns=['age', 'hours-per-week'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[7][0].optional_code_info.source_code == 'preprocessing.StandardScaler()'
    actual_df = dag_node_to_inspection_results[7][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[array([RangeComparison(0.5, 0.6), RangeComparison(-0.1, -0.05)])],
                             [array([RangeComparison(-0.8, -0.7), RangeComparison(0.7, 0.8)])]],
                            columns=['array'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[8][0].code_location.lineno == 18
    actual_df = dag_node_to_inspection_results[8][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, RangeComparison(0.5, 0.6),
                                     RangeComparison(-0.1, -0.05)])],
                             [array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, RangeComparison(-0.8, -0.7),
                                     RangeComparison(0.7, 0.8)])]],
                            columns=['array'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[9][0].optional_code_info.source_code == 'tree.DecisionTreeClassifier()'
    actual_df = dag_node_to_inspection_results[9][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, RangeComparison(0.5, 0.6),
                                     RangeComparison(-0.1, -0.05)])],
                             [array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, RangeComparison(-0.8, -0.7),
                                     RangeComparison(0.7, 0.8)])]],
                            columns=['array'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[10][0].optional_code_info.source_code == 'tree.DecisionTreeClassifier()'
    actual_df = dag_node_to_inspection_results[10][1][MaterializeFirstOutputRows(2)]
    expected_df = DataFrame([[array([1])], [array([1])]], columns=['array'])
    pandas.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    assert dag_node_to_inspection_results[11][0].optional_code_info.source_code == 'tree.DecisionTreeClassifier()'
    actual_df = dag_node_to_inspection_results[11][1][MaterializeFirstOutputRows(2)]
    assert actual_df is None
