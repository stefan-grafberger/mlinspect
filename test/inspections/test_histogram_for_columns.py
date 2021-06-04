"""
Tests whether HistogramForColumns works
"""
from inspect import cleandoc

from testfixtures import compare

from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections import HistogramForColumns


def test_histogram_merge():
    """
    Tests whether RowLineage works for joins
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
    Tests whether RowLineage works for joins
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
