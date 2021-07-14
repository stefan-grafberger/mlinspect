"""
Tests whether ColumnPropagation works
"""
from inspect import cleandoc

import pandas
from pandas import DataFrame

from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections import ColumnPropagation


def test_propagation_merge():
    """
    Tests whether ColumnPropagation works for joins
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c', 'cat_b'], 'B': [1, 2, 4, 5, 7]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(ColumnPropagation(["A"], 2)) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    propagation_output = inspection_results[0][ColumnPropagation(["A"], 2)]
    expected_df = DataFrame([['cat_a', 1, 'cat_a'], ['cat_b', 2, 'cat_b']], columns=['A', 'B', 'mlinspect_A'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))

    propagation_output = inspection_results[1][ColumnPropagation(["A"], 2)]
    expected_df = DataFrame([[1, 1., None], [2, 5., None]], columns=['B', 'C', 'mlinspect_A'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))

    propagation_output = inspection_results[2][ColumnPropagation(["A"], 2)]
    expected_df = DataFrame([['cat_a', 1, 1., 'cat_a'], ['cat_b', 2, 5., 'cat_b']],
                            columns=['A', 'B', 'C', 'mlinspect_A'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_propagation_projection():
    """
    Tests whether ColumnPropagation works for projections
    """
    test_code = cleandoc("""
            import pandas as pd

            pandas_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c', 'cat_b'], 
                'B': [1, 2, 4, 5, 7], 'C': [2, 2, 10, 5, 7]})
            pandas_df = pandas_df[['B', 'C']]
            pandas_df = pandas_df[['C']]
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(ColumnPropagation(["A"], 2)) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    propagation_output = inspection_results[0][ColumnPropagation(["A"], 2)]
    expected_df = DataFrame([['cat_a', 1, 2, 'cat_a'], ['cat_b', 2, 2, 'cat_b']], columns=['A', 'B', 'C', 'mlinspect_A'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))

    propagation_output = inspection_results[1][ColumnPropagation(["A"], 2)]
    expected_df = DataFrame([[1, 2, 'cat_a'], [2, 2, 'cat_b']], columns=['B', 'C', 'mlinspect_A'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))

    propagation_output = inspection_results[2][ColumnPropagation(["A"], 2)]
    expected_df = DataFrame([[2, 'cat_a'], [2, 'cat_b']], columns=['C', 'mlinspect_A'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_propagation_score():
    """
    Tests whether ColumnPropagation works for projections
    """
    test_code = cleandoc("""
            import pandas as pd
            from sklearn.preprocessing import label_binarize, StandardScaler
            from sklearn.tree import DecisionTreeClassifier
            import numpy as np

            df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'cat_col': ['cat_a', 'cat_b', 'cat_a', 'cat_a'], 
                'target': ['no', 'no', 'yes', 'yes']})

            train = StandardScaler().fit_transform(df[['A', 'B']])
            target = label_binarize(df['target'], classes=['no', 'yes'])

            clf = DecisionTreeClassifier()
            clf = clf.fit(train, target)

            test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'cat_col': ['cat_a', 'cat_b'], 
                'target': ['no', 'yes']})
            test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
            test_score = clf.score(test_df[['A', 'B']], test_labels)
            assert test_score == 1.0
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(ColumnPropagation(["cat_col"], 2)) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    propagation_output = inspection_results[14][ColumnPropagation(["cat_col"], 2)]
    expected_df = DataFrame([[0, 'cat_a'], [1, 'cat_b']], columns=['array', 'mlinspect_cat_col'])
    pandas.testing.assert_frame_equal(propagation_output.reset_index(drop=True), expected_df.reset_index(drop=True))
