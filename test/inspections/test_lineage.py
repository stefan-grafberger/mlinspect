"""
Tests whether RowLineage works
"""
from inspect import cleandoc

import pandas
import numpy as np
from pandas import DataFrame

from mlinspect import OperatorType
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.inspections import RowLineage
from mlinspect.inspections._lineage import LineageId


def test_row_lineage_merge():
    """
    Tests whether RowLineage works for joins
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(RowLineage(2)) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    lineage_output = inspection_results[0][RowLineage(2)]
    expected_lineage_df = DataFrame([[0, 1, {LineageId(0, 0)}],
                                     [2, 2, {LineageId(0, 1)}]],
                                    columns=['A', 'B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[1][RowLineage(2)]
    expected_lineage_df = DataFrame([[1, 1., {LineageId(1, 0)}],
                                     [2, 5., {LineageId(1, 1)}]],
                                    columns=['B', 'C', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[2][RowLineage(2)]
    expected_lineage_df = DataFrame([[0, 1, 1., {LineageId(0, 0), LineageId(1, 0)}],
                                     [2, 2, 5., {LineageId(0, 1), LineageId(1, 1)}]],
                                    columns=['A', 'B', 'C', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_row_lineage_concat():
    """
    Tests whether RowLineage works for concats
    """
    test_code = cleandoc("""
            import pandas as pd
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer

            df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
            column_transformer = ColumnTransformer(transformers=[
                ('numeric', StandardScaler(), ['A']),
                ('categorical', OneHotEncoder(sparse=False), ['B'])
            ])
            encoded_data = column_transformer.fit_transform(df)
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(RowLineage(2)) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    lineage_output = inspection_results[0][RowLineage(2)]
    expected_lineage_df = DataFrame([[1, 'cat_a', {LineageId(0, 0)}],
                                     [2, 'cat_b', {LineageId(0, 1)}]],
                                    columns=['A', 'B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[1][RowLineage(2)]
    expected_lineage_df = DataFrame([[1, {LineageId(0, 0)}],
                                     [2, {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[2][RowLineage(2)]
    expected_lineage_df = DataFrame([[np.array([-1.0]), {LineageId(0, 0)}],
                                     [np.array([-0.7142857142857143]), {LineageId(0, 1)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[3][RowLineage(2)]
    expected_lineage_df = DataFrame([['cat_a', {LineageId(0, 0)}],
                                     ['cat_b', {LineageId(0, 1)}]],
                                    columns=['B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[4][RowLineage(2)]
    expected_lineage_df = DataFrame([[np.array([1., 0., 0.]), {LineageId(0, 0)}],
                                     [np.array([0., 1., 0.]), {LineageId(0, 1)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[5][RowLineage(2)]
    expected_lineage_df = DataFrame([[np.array([-1.0, 1., 0., 0.]), {LineageId(0, 0)}],
                                     [np.array([-0.7142857142857143, 0., 1., 0.]), {LineageId(0, 1)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_all_rows_for_op_type():
    """
    Tests whether RowLineage works for materialising all data from specific operators
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': [0, 2], 'B': [1, 2]})
            df_b = pd.DataFrame({'B': [1, 2], 'C': [1, 5]})
            df_merged = df_a.merge(df_b, on='B')
            """)
    row_lineage = RowLineage(RowLineage.ALL_ROWS, [OperatorType.DATA_SOURCE])
    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(row_lineage) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    lineage_output = inspection_results[0][row_lineage]
    expected_lineage_df = DataFrame([[0, 1, {LineageId(0, 0)}],
                                     [2, 2, {LineageId(0, 1)}]],
                                    columns=['A', 'B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[1][row_lineage]
    expected_lineage_df = DataFrame([[1, 1, {LineageId(1, 0)}],
                                     [2, 5, {LineageId(1, 1)}]],
                                    columns=['B', 'C', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    lineage_output = inspection_results[2][row_lineage]
    assert lineage_output is None
