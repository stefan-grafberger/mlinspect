"""
Tests whether the monkey patching works for all patched pandas methods
"""
from inspect import cleandoc

import networkx
import pandas
from pandas import DataFrame
from testfixtures import compare, StringComparison

from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, OperatorType, CodeReference
from mlinspect.inspections._lineage import RowLineage, LineageId


def test_read_csv():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works
    """
    test_code = cleandoc("""
        import os
        import pandas as pd
        from mlinspect.utils import get_project_root
        
        train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
        raw_data = pd.read_csv(train_file, na_values='?', index_col=0)
        assert len(raw_data) == 22792
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])

    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]
    expected_node = DagNode(0, "<string-source>", 6, OperatorType.DATA_SOURCE, ('pandas.io.parsers', 'read_csv'),
                            description=StringComparison(r".*\.csv"),
                            columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                     'hours-per-week', 'native-country', 'income-per-year'],
                            optional_code_reference=CodeReference(6, 11, 6, 62),
                            optional_source_code="pd.read_csv(train_file, na_values='?', index_col=0)")
    compare(extracted_node, expected_node)

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[extracted_node]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                                      'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K',
                                      {LineageId(0, 0)}],
                                     [29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                                      'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K',
                                      {LineageId(0, 1)}]],
                                    columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                             'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                                             'income-per-year', 'mlinspect_lineage'])

    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__init__():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame([0, 1, 2], columns=['A'])
        assert len(df) == 3
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(0, "<string-source>", 3, OperatorType.DATA_SOURCE, ('pandas.core.frame', 'DataFrame'),
                            description='', columns=['A'], optional_code_reference=CodeReference(3, 5, 3, 43),
                            optional_source_code="pd.DataFrame([0, 1, 2], columns=['A'])")
    compare(extracted_node, expected_node)

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[extracted_node]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0, {LineageId(0, 0)}],
                                     [1, {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame_dropna():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'dropna') works
    """
    test_code = cleandoc("""
        import pandas as pd
        
        df = pd.DataFrame([0, 2, 4, 5, None], columns=['A'])
        assert len(df) == 5
        df = df.dropna()
        assert len(df) == 4
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])

    expected_dag = networkx.DiGraph()
    expected_missing_op = DagNode(0, "<string-source>", 3, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(3, 5, 3, 52),
                                  optional_source_code="pd.DataFrame([0, 2, 4, 5, None], columns=['A'])")
    expected_select = DagNode(1, "<string-source>", 5, OperatorType.SELECTION, module=('pandas.core.frame', 'dropna'),
                              description='dropna', columns=['A'], optional_code_reference=CodeReference(5, 5, 5, 16),
                              optional_source_code='df.dropna()')
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(2)].to_pandas_df()
    expected_lineage_df = DataFrame([[0., 0],
                                     [2., 1]],
                                    columns=['A', 'mlinspect_row_lineage_2_data_source_0'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__getitem__series():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for series arguments
    """
    test_code = cleandoc("""
            import pandas as pd

            df = pd.DataFrame([0, 2, 4, 8, None], columns=['A'])
            a = df['A']
            pd.testing.assert_series_equal(a, pd.Series([0, 2, 4, 8, None], name='A'))
            """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_missing_op = DagNode(0, "<string-source>", 3, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(3, 5, 3, 52),
                                  optional_source_code="pd.DataFrame([0, 2, 4, 8, None], columns=['A'])")
    expected_project = DagNode(1, "<string-source>", 4, OperatorType.PROJECTION,
                               module=('pandas.core.frame', '__getitem__'), description="to ['A']", columns=['A'],
                               optional_code_reference=CodeReference(4, 4, 4, 11),
                               optional_source_code="df['A']")
    expected_dag.add_edge(expected_missing_op, expected_project)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_project]
    lineage_output = inspection_results_data_source[RowLineage(2)].to_pandas_df()
    expected_lineage_df = DataFrame([[0., 0],
                                     [2., 1]],
                                    columns=['A', 'mlinspect_row_lineage_2_data_source_0'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__getitem__frame():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for df arguments
    """
    test_code = cleandoc("""
                import pandas as pd

                df = pd.DataFrame([[0, None, 2], [1, 2, 3], [4, None, 2], [9, 2, 3], [6, 1, 2], [1, 2, 3]], 
                    columns=['A', 'B', 'C'])
                df_projection = df[['A', 'C']]
                df_expected = pd.DataFrame([[0, 2], [1, 3], [4, 2], [9, 3], [6, 2], [1, 3]], columns=['A', 'C'])
                pd.testing.assert_frame_equal(df_projection, df_expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_missing_op = DagNode(0, "<string-source>", 3, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B', 'C'],
                                  optional_code_reference=CodeReference(3, 5, 4, 28),
                                  optional_source_code="pd.DataFrame([[0, None, 2], [1, 2, 3], [4, None, 2], "
                                                       "[9, 2, 3], [6, 1, 2], [1, 2, 3]], \n"
                                                       "    columns=['A', 'B', 'C'])")
    expected_project = DagNode(1, "<string-source>", 5, OperatorType.PROJECTION,
                               module=('pandas.core.frame', '__getitem__'), description="to ['A', 'C']",
                               columns=['A', 'C'], optional_code_reference=CodeReference(5, 16, 5, 30),
                               optional_source_code="df[['A', 'C']]")
    expected_dag.add_edge(expected_missing_op, expected_project)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_project]
    lineage_output = inspection_results_data_source[RowLineage(2)].to_pandas_df()
    expected_lineage_df = DataFrame([[0, 2, 0],
                                     [1, 3, 1]],
                                    columns=['A', 'C', 'mlinspect_row_lineage_2_data_source_0'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__getitem__selection():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for df arguments
    """
    test_code = cleandoc("""
                import pandas as pd

                df = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 5, 4, 11, None]})
                df_selection = df[df['A'] > 3]
                df_expected = pd.DataFrame({'A': [4, 8, 5], 'B': [4, 11, None]})
                pd.testing.assert_frame_equal(df_selection.reset_index(drop=True), df_expected.reset_index(drop=True))
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0, "<string-source>", 3, OperatorType.DATA_SOURCE,
                                   ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B'],
                                   optional_code_reference=CodeReference(3, 5, 3, 67),
                                   optional_source_code="pd.DataFrame({'A': [0, 2, 4, 8, 5], "
                                                        "'B': [1, 5, 4, 11, None]})")
    expected_projection = DagNode(1, "<string-source>", 4, OperatorType.PROJECTION,
                                  module=('pandas.core.frame', '__getitem__'), description="to ['A']",
                                  columns=['A'], optional_code_reference=CodeReference(4, 18, 4, 25),
                                  optional_source_code="df['A']")
    expected_dag.add_edge(expected_data_source, expected_projection)
    expected_selection = DagNode(2, "<string-source>", 4, OperatorType.SELECTION,
                                 module=('pandas.core.frame', '__getitem__'), description='Select by Series',
                                 columns=['A', 'B'], optional_code_reference=CodeReference(4, 15, 4, 30),
                                 optional_source_code="df[df['A'] > 3]")
    expected_dag.add_edge(expected_data_source, expected_selection)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_selection]
    lineage_output = inspection_results_data_source[RowLineage(2)].to_pandas_df()
    expected_lineage_df = DataFrame([[4, 4., 2],
                                     [8, 11., 3]],
                                    columns=['A', 'B', 'mlinspect_row_lineage_2_data_source_0'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_series__init__():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'Series') works
    """
    test_code = cleandoc("""
        import pandas as pd

        pd_series = pd.Series([0, 2, 4, None], name='A')
        assert len(pd_series) == 4
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(0, "<string-source>", 3, OperatorType.DATA_SOURCE, ('pandas.core.series', 'Series'),
                            description='', columns=['A'], optional_code_reference=CodeReference(3, 12, 3, 48),
                            optional_source_code="pd.Series([0, 2, 4, None], name='A')")
    compare(extracted_node, expected_node)

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_node]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0., {LineageId(0, 0)}],
                                     [2., {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))
