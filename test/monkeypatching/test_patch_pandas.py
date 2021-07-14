"""
Tests whether the monkey patching works for all patched pandas methods
"""
import math
from inspect import cleandoc

import networkx
import pandas
from pandas import DataFrame
from testfixtures import compare, StringComparison

from mlinspect import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.inspections._lineage import RowLineage, LineageId


def test_read_csv():
    """
    Tests whether the monkey patching of ('pandas.io.parsers', 'read_csv') works
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
    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 6),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.io.parsers', 'read_csv')),
                            DagNodeDetails(StringComparison(r".*\.csv"),
                                           ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                            'marital-status',
                                            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                            'hours-per-week', 'native-country', 'income-per-year']),
                            OptionalCodeInfo(CodeReference(6, 11, 6, 62),
                                             "pd.read_csv(train_file, na_values='?', index_col=0)"))
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
    Tests whether the monkey patching of ('pandas.core.frame', 'DataFrame') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame([0, 1, 2], columns=['A'])
        assert len(df) == 3
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                            DagNodeDetails(None, ['A']),
                            OptionalCodeInfo(CodeReference(3, 5, 3, 43), "pd.DataFrame([0, 1, 2], columns=['A'])"))
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
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 52),
                                                    "pd.DataFrame([0, 2, 4, 5, None], columns=['A'])"))
    expected_select = DagNode(1,
                              BasicCodeLocation("<string-source>", 5),
                              OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                              DagNodeDetails('dropna', ['A']),
                              OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'))
    expected_dag.add_edge(expected_data_source, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0., {LineageId(0, 0)}],
                                     [2., {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__getitem__series():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for a single string argument
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
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 52),
                                                    "pd.DataFrame([0, 2, 4, 8, None], columns=['A'])"))
    expected_project = DagNode(1,
                               BasicCodeLocation("<string-source>", 4),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['A']", ['A']),
                               OptionalCodeInfo(CodeReference(4, 4, 4, 11), "df['A']"))
    expected_dag.add_edge(expected_data_source, expected_project)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_project]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0., {LineageId(0, 0)}],
                                     [2., {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__getitem__frame():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for multiple string arguments
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
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'C']),
                                   OptionalCodeInfo(CodeReference(3, 5, 4, 28),
                                                    "pd.DataFrame([[0, None, 2], [1, 2, 3], [4, None, 2], "
                                                    "[9, 2, 3], [6, 1, 2], [1, 2, 3]], \n"
                                                    "    columns=['A', 'B', 'C'])"))
    expected_project = DagNode(1,
                               BasicCodeLocation("<string-source>", 5),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['A', 'C']", ['A', 'C']),
                               OptionalCodeInfo(CodeReference(5, 16, 5, 30), "df[['A', 'C']]"))
    expected_dag.add_edge(expected_data_source, expected_project)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_project]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0, 2, {LineageId(0, 0)}],
                                     [1, 3, {LineageId(0, 1)}]],
                                    columns=['A', 'C', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__getitem__selection():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for filtering
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
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 67),
                                                    "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 5, 4, 11, None]})"))
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 4),
                                  OperatorContext(OperatorType.PROJECTION,
                                                  FunctionInfo('pandas.core.frame', '__getitem__')),
                                  DagNodeDetails("to ['A']", ['A']),
                                  OptionalCodeInfo(CodeReference(4, 18, 4, 25), "df['A']"))
    expected_dag.add_edge(expected_data_source, expected_projection)
    expected_selection = DagNode(2,
                                 BasicCodeLocation("<string-source>", 4),
                                 OperatorContext(OperatorType.SELECTION,
                                                 FunctionInfo('pandas.core.frame', '__getitem__')),
                                 DagNodeDetails("Select by Series: df[df['A'] > 3]", ['A', 'B']),
                                 OptionalCodeInfo(CodeReference(4, 15, 4, 30), "df[df['A'] > 3]"))
    expected_dag.add_edge(expected_data_source, expected_selection)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_selection]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[4, 4., {LineageId(0, 2)}],
                                     [8, 11., {LineageId(0, 3)}]],
                                    columns=['A', 'B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame__setitem__():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__setitem__') works
    """
    test_code = cleandoc("""
                import pandas as pd

                pandas_df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                              'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                              'baz': [1, 2, 3, 4, 5, 6],
                              'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
                pandas_df['baz'] = pandas_df['baz'] + 1
                df_expected = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                              'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                              'baz': [2, 3, 4, 5, 6, 7],
                              'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
                pd.testing.assert_frame_equal(pandas_df, df_expected)
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['foo', 'bar', 'baz', 'zoo']),
                                   OptionalCodeInfo(CodeReference(3, 12, 6, 53),
                                                    "pd.DataFrame({'foo': ['one', 'one', 'one', 'two', "
                                                    "'two', 'two'],\n"
                                                    "              'bar': ['A', 'B', 'C', 'A', 'B', 'C'],\n"
                                                    "              'baz': [1, 2, 3, 4, 5, 6],\n"
                                                    "              'zoo': ['x', 'y', 'z', 'q', 'w', 't']})"))
    expected_project = DagNode(1,
                               BasicCodeLocation("<string-source>", 7),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['baz']", ['baz']),
                               OptionalCodeInfo(CodeReference(7, 19, 7, 35), "pandas_df['baz']"))
    expected_dag.add_edge(expected_data_source, expected_project)
    expected_project_modify = DagNode(2,
                                      BasicCodeLocation("<string-source>", 7),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                      FunctionInfo('pandas.core.frame', '__setitem__')),
                                      DagNodeDetails("modifies ['baz']", ['foo', 'bar', 'baz', 'zoo']),
                                      OptionalCodeInfo(CodeReference(7, 0, 7, 39),
                                                       "pandas_df['baz'] = pandas_df['baz'] + 1"))
    expected_dag.add_edge(expected_data_source, expected_project_modify)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_project_modify]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([['one', 'A', 2, 'x', {LineageId(0, 0)}],
                                     ['one', 'B', 3, 'y', {LineageId(0, 1)}]],
                                    columns=['foo', 'bar', 'baz', 'zoo', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame_replace():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'replace') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame(['Low', 'Medium', 'Low', 'High', None], columns=['A'])
        df_replace = df.replace('Medium', 'Low')
        df_expected = pd.DataFrame(['Low', 'Low', 'Low', 'High', None], columns=['A'])
        pd.testing.assert_frame_equal(df_replace.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 72),
                                                    "pd.DataFrame(['Low', 'Medium', 'Low', 'High', None], "
                                                    "columns=['A'])"))
    expected_modify = DagNode(1,
                              BasicCodeLocation("<string-source>", 4),
                              OperatorContext(OperatorType.PROJECTION_MODIFY,
                                              FunctionInfo('pandas.core.frame', 'replace')),
                              DagNodeDetails("Replace 'Medium' with 'Low'", ['A']),
                              OptionalCodeInfo(CodeReference(4, 13, 4, 40), "df.replace('Medium', 'Low')"))
    expected_dag.add_edge(expected_data_source, expected_modify)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_modify]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([['Low', {LineageId(0, 0)}],
                                     ['Low', {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame_merge_on():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
        df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
        df_merged = df_a.merge(df_b, on='B')
        df_expected = pd.DataFrame({'A': [0, 2, 4, 8], 'B': [1, 2, 4, 5], 'C': [1, 5, 11, None]})
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})"))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['B', 'C']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 69),
                                          "pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})"))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails("on 'B'", ['A', 'B', 'C']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 36), "df_a.merge(df_b, on='B')"))
    expected_dag.add_edge(expected_a, expected_join)
    expected_dag.add_edge(expected_b, expected_join)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_join]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0, 1, 1., {LineageId(0, 0), LineageId(1, 0)}],
                                     [2, 2, 5., {LineageId(0, 1), LineageId(1, 1)}]],
                                    columns=['A', 'B', 'C', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame_merge_left_right_on():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
        df_b = pd.DataFrame({'C': [1, 2, 3, 4, 5], 'D': [1, 5, 4, 11, None]})
        df_merged = df_a.merge(df_b, left_on='B', right_on='C')
        df_expected = pd.DataFrame({'A': [0, 2, 4, 8], 'B': [1, 2, 4, 5],  'C': [1, 2, 4, 5], 'D': [1, 5, 11, None]})
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})"))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['C', 'D']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 69),
                                          "pd.DataFrame({'C': [1, 2, 3, 4, 5], 'D': [1, 5, 4, 11, None]})"))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails("on 'B' == 'C'", ['A', 'B', 'C', 'D']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 55),
                                             "df_a.merge(df_b, left_on='B', right_on='C')"))
    expected_dag.add_edge(expected_a, expected_join)
    expected_dag.add_edge(expected_b, expected_join)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_join]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0, 1, 1, 1., {LineageId(0, 0), LineageId(1, 0)}],
                                     [2, 2, 2, 5., {LineageId(0, 1), LineageId(1, 1)}]],
                                    columns=['A', 'B', 'C', 'D', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame_merge_index():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
        df_b = pd.DataFrame({'C': [1, 2, 3, 4], 'D': [1, 5, 4, 11]})
        df_merged = df_a.merge(right=df_b, left_index=True, right_index=True, how='outer')
        df_expected = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7],  'C': [1., 2., 3., 4., None], 
            'D': [1., 5., 4., 11., None]})
        print(df_merged)
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})"))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['C', 'D']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 60),
                                          "pd.DataFrame({'C': [1, 2, 3, 4], 'D': [1, 5, 4, 11]})"))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails('on left_index == right_index (outer)', ['A', 'B', 'C', 'D']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 82),
                                             "df_a.merge(right=df_b, left_index=True, right_index=True, how='outer')"))
    expected_dag.add_edge(expected_a, expected_join)
    expected_dag.add_edge(expected_b, expected_join)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_join]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0, 1, 1., 1., {LineageId(0, 0), LineageId(1, 0)}],
                                     [2, 2, 2., 5., {LineageId(0, 1), LineageId(1, 1)}]],
                                    columns=['A', 'B', 'C', 'D', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_frame_merge_sorted():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works if the sort option is set to True
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [7, 5, 4, 2, 1]})
        df_b = pd.DataFrame({'B': [1, 4, 3, 2, 5], 'C': [1, 5, 4, 11, None]})
        df_merged = df_a.merge(df_b, on='B', sort=True)
        df_expected = pd.DataFrame({'A': [5, 8, 4, 2], 'B': [1, 2, 4, 5], 'C': [1, 11, 5, None]})
        pd.testing.assert_frame_equal(df_merged, df_expected)
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(5)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [7, 5, 4, 2, 1]})"))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['B', 'C']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 69),
                                          "pd.DataFrame({'B': [1, 4, 3, 2, 5], 'C': [1, 5, 4, 11, None]})"))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails("on 'B'", ['A', 'B', 'C']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 47), "df_a.merge(df_b, on='B', sort=True)"))
    expected_dag.add_edge(expected_a, expected_join)
    expected_dag.add_edge(expected_b, expected_join)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_join]
    lineage_output = inspection_results_data_source[RowLineage(5)]
    expected_lineage_df = DataFrame([[5, 1, 1.,       {LineageId(0, 4), LineageId(1, 0)}],
                                     [8, 2, 11.,      {LineageId(0, 3), LineageId(1, 3)}],
                                     [4, 4, 5.,       {LineageId(0, 2), LineageId(1, 1)}],
                                     [2, 5, math.nan, {LineageId(0, 1), LineageId(1, 4)}]],
                                    columns=['A', 'B', 'C', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_groupby_agg():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'groupby') and ('pandas.core.groupbygeneric', 'agg')
    works.
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame({'group': ['A', 'B', 'A', 'C', 'B'], 'value': [1, 2, 1, 3, 4]})
        df_groupby_agg = df.groupby('group').agg(mean_value=('value', 'mean'))
        
        df_expected = pd.DataFrame({'group': ['A', 'B', 'C'], 'mean_value': [1, 3, 3]})
        pd.testing.assert_frame_equal(df_groupby_agg.reset_index(drop=False), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                            DagNodeDetails(None, ['group', 'value']),
                            OptionalCodeInfo(CodeReference(3, 5, 3, 81),
                                             "pd.DataFrame({'group': ['A', 'B', 'A', 'C', 'B'], "
                                             "'value': [1, 2, 1, 3, 4]})"))
    expected_groupby_agg = DagNode(1,
                                   BasicCodeLocation("<string-source>", 4),
                                   OperatorContext(OperatorType.GROUP_BY_AGG,
                                                   FunctionInfo('pandas.core.groupby.generic', 'agg')),
                                   DagNodeDetails("Groupby 'group', Aggregate: '{'mean_value': ('value', 'mean')}'",
                                                  ['group', 'mean_value']),
                                   OptionalCodeInfo(CodeReference(4, 17, 4, 70),
                                                    "df.groupby('group').agg(mean_value=('value', 'mean'))"))
    expected_dag.add_edge(expected_data, expected_groupby_agg)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_groupby_agg]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([["A", 1, {LineageId(1, 0)}],
                                     ['B', 3, {LineageId(1, 1)}]],
                                    columns=['group', 'mean_value', 'mlinspect_lineage'])
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

    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.series', 'Series')),
                            DagNodeDetails(None, ['A']),
                            OptionalCodeInfo(CodeReference(3, 12, 3, 48), "pd.Series([0, 2, 4, None], name='A')"))
    compare(extracted_node, expected_node)

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_node]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0., {LineageId(0, 0)}],
                                     [2., {LineageId(0, 1)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))
