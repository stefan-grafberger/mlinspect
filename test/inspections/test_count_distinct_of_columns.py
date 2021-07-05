"""
Tests whether CountDistinctOfColumns works
"""
from inspect import cleandoc

from testfixtures import compare

from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections import CountDistinctOfColumns


def test_count_distinct_merge():
    """
    Tests whether CountDistinctOfColumns works for joins
    """
    test_code = cleandoc("""
            import numpy as np
            import pandas as pd

            df_a = pd.DataFrame({'A': ['cat_a', None, 'cat_a', 'cat_c', None], 'B': [1, 2, 4, 5, 7]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, np.nan], 'C': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(CountDistinctOfColumns(['A', 'B'])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    count_distinct_output = inspection_results[0][CountDistinctOfColumns(['A', 'B'])]
    expected_count_distinct = {'A': 3, 'B': 5}
    compare(count_distinct_output, expected_count_distinct)

    count_distinct_output = inspection_results[1][CountDistinctOfColumns(['A', 'B'])]
    expected_count_distinct = {'B': 5}
    compare(count_distinct_output, expected_count_distinct)

    count_distinct_output = inspection_results[2][CountDistinctOfColumns(['A', 'B'])]
    expected_count_distinct = {'A': 2, 'B': 3}
    compare(count_distinct_output, expected_count_distinct)


def test_count_distinct_projection():
    """
    Tests whether CountDistinctOfColumns works for projections
    """
    test_code = cleandoc("""
            import pandas as pd
            import numpy as np

            pandas_df = pd.DataFrame({'A': ['cat_a', 'cat_b', None, 'cat_c', 'cat_b'], 
                'B': [1, None, np.nan, None, 7], 'C': [2, 2, 10, 5, 7]})
            pandas_df = pandas_df[['B', 'C']]
            pandas_df = pandas_df[['C']]
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(CountDistinctOfColumns(['A', 'B'])) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    count_distinct_output = inspection_results[0][CountDistinctOfColumns(['A', 'B'])]
    expected_count_distinct = {'A': 4, 'B': 5}
    compare(count_distinct_output, expected_count_distinct)

    count_distinct_output = inspection_results[1][CountDistinctOfColumns(['A', 'B'])]
    expected_count_distinct = {'B': 5}
    compare(count_distinct_output, expected_count_distinct)

    count_distinct_output = inspection_results[2][CountDistinctOfColumns(['A', 'B'])]
    expected_count_distinct = {}
    compare(count_distinct_output, expected_count_distinct)
