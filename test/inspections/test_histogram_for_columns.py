"""
Tests whether HistogramForColumns works
"""
from inspect import cleandoc

from testfixtures import compare

from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections import HistogramForColumns


def test_histogram_merge():
    """
    Tests whether HistogramForColumns works for joins
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c', 'cat_b'], 'B': [1, 2, 4, 5, 7]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(HistogramForColumns(["A"])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    histogram_output = inspection_results[0][HistogramForColumns(["A"])]
    expected_histogram = {'A': {'cat_a': 2, 'cat_b': 2, 'cat_c': 1}}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[1][HistogramForColumns(["A"])]
    expected_histogram = {'A': {}}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[2][HistogramForColumns(["A"])]
    expected_histogram = {'A': {'cat_a': 2, 'cat_b': 1, 'cat_c': 1}}
    compare(histogram_output, expected_histogram)


def test_histogram_projection():
    """
    Tests whether HistogramForColumns works for projections
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
        .add_required_inspection(HistogramForColumns(["A"])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    histogram_output = inspection_results[0][HistogramForColumns(["A"])]
    expected_histogram = {'A': {'cat_a': 2, 'cat_b': 2, 'cat_c': 1}}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[1][HistogramForColumns(["A"])]
    expected_histogram = {'A': {'cat_a': 2, 'cat_b': 2, 'cat_c': 1}}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[2][HistogramForColumns(["A"])]
    expected_histogram = {'A': {'cat_a': 2, 'cat_b': 2, 'cat_c': 1}}
    compare(histogram_output, expected_histogram)


def test_histogram_score():
    """
    Tests whether HistogramForColumns works for projections
    """
    test_code = cleandoc("""
            import pandas as pd
            from sklearn.preprocessing import label_binarize, StandardScaler
            from sklearn.tree import DecisionTreeClassifier
            import numpy as np

            df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

            train = StandardScaler().fit_transform(df[['A', 'B']])
            target = label_binarize(df['target'], classes=['no', 'yes'])

            clf = DecisionTreeClassifier()
            clf = clf.fit(train, target)

            test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
            test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
            test_score = clf.score(test_df[['A', 'B']], test_labels)
            assert test_score == 1.0
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(HistogramForColumns(["target"])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    histogram_output = inspection_results[14][HistogramForColumns(["target"])]
    expected_histogram = {'target': {'no': 1, 'yes': 1}}
    compare(histogram_output, expected_histogram)
