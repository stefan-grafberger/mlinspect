"""
Tests whether IntersectionalHistogramForColumns works
"""
from inspect import cleandoc

from testfixtures import compare

from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections import IntersectionalHistogramForColumns


def test_histogram_merge():
    """
    Tests whether IntersectionalHistogramForColumns works for joins
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_b', 'cat_b'], 
                'B': [1, 2, 4, 5, 7],
                'C': [True, False, True, True, True]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'D': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(IntersectionalHistogramForColumns(["A", "C"])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    histogram_output = inspection_results[0][IntersectionalHistogramForColumns(["A", "C"])]
    expected_histogram = {('cat_a', True): 2, ('cat_b', False): 1, ('cat_b', True): 2}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[1][IntersectionalHistogramForColumns(["A", "C"])]
    expected_histogram = {(None, None): 5}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[2][IntersectionalHistogramForColumns(["A", "C"])]
    expected_histogram = {('cat_a', True): 2, ('cat_b', False): 1, ('cat_b', True): 1}
    compare(histogram_output, expected_histogram)


def test_histogram_projection():
    """
    Tests whether IntersectionalHistogramForColumns works for projections
    """
    test_code = cleandoc("""
            import pandas as pd

            pandas_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c', 'cat_b'], 
                'B': [1, 2, 4, 5, 7], 'C': [True, False, True, True, True]})
            pandas_df = pandas_df[['B', 'C']]
            pandas_df = pandas_df[['C']]
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(IntersectionalHistogramForColumns(["A", "C"])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    histogram_output = inspection_results[0][IntersectionalHistogramForColumns(["A", "C"])]
    expected_histogram = {('cat_a', True): 2, ('cat_b', False): 1, ('cat_c', True): 1, ('cat_b', True): 1}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[1][IntersectionalHistogramForColumns(["A", "C"])]
    expected_histogram = {('cat_a', True): 2, ('cat_b', False): 1, ('cat_c', True): 1, ('cat_b', True): 1}
    compare(histogram_output, expected_histogram)

    histogram_output = inspection_results[2][IntersectionalHistogramForColumns(["A", "C"])]
    expected_histogram = {('cat_a', True): 2, ('cat_b', False): 1, ('cat_c', True): 1, ('cat_b', True): 1}
    compare(histogram_output, expected_histogram)
