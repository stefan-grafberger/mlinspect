"""
Tests whether the monkey patching works for all patched sklearn methods
"""
# pylint: disable=too-many-lines
from inspect import cleandoc

import networkx
import numpy
import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect import OperatorType, OperatorContext, FunctionInfo
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.inspections._lineage import RowLineage, LineageId


def test_label_binarize():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize
                import numpy as np

                pd_series = pd.Series(['yes', 'no', 'no', 'yes'], name='A')
                binarized = label_binarize(pd_series, classes=['no', 'yes'])
                expected = np.array([[1], [0], [0], [1]])
                assert np.array_equal(binarized, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 12, 5, 59),
                                                    "pd.Series(['yes', 'no', 'no', 'yes'], name='A')"))
    expected_binarize = DagNode(1,
                                BasicCodeLocation("<string-source>", 6),
                                OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array']),
                                OptionalCodeInfo(CodeReference(6, 12, 6, 60),
                                                 "label_binarize(pd_series, classes=['no', 'yes'])"))
    expected_dag.add_edge(expected_data_source, expected_binarize)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_binarize]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1]), {LineageId(0, 0)}],
                                     [numpy.array([0]), {LineageId(0, 1)}],
                                     [numpy.array([0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_train_test_split():
    """
    Tests whether the monkey patching of ('sklearn.model_selection._split', 'train_test_split') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.model_selection import train_test_split

                pandas_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                train_data, test_data = train_test_split(pandas_df, random_state=0)
                
                expected_train = pd.DataFrame({'A': [5, 2, 1]})
                expected_test = pd.DataFrame({'A': [10]})
                
                pd.testing.assert_frame_equal(train_data.reset_index(drop=True), expected_train.reset_index(drop=True))
                pd.testing.assert_frame_equal(test_data.reset_index(drop=True), expected_test.reset_index(drop=True))
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[4])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_source = DagNode(0,
                              BasicCodeLocation("<string-source>", 4),
                              OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                              DagNodeDetails(None, ['A']),
                              OptionalCodeInfo(CodeReference(4, 12, 4, 46), "pd.DataFrame({'A': [1, 2, 10, 5]})"))
    expected_train = DagNode(1,
                             BasicCodeLocation("<string-source>", 5),
                             OperatorContext(OperatorType.TRAIN_TEST_SPLIT,
                                             FunctionInfo('sklearn.model_selection._split', 'train_test_split')),
                             DagNodeDetails('(Train Data)', ['A']),
                             OptionalCodeInfo(CodeReference(5, 24, 5, 67),
                                              'train_test_split(pandas_df, random_state=0)'))
    expected_dag.add_edge(expected_source, expected_train)
    expected_test = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.TRAIN_TEST_SPLIT,
                                            FunctionInfo('sklearn.model_selection._split', 'train_test_split')),
                            DagNodeDetails('(Test Data)', ['A']),
                            OptionalCodeInfo(CodeReference(5, 24, 5, 67),
                                             'train_test_split(pandas_df, random_state=0)'))
    expected_dag.add_edge(expected_source, expected_test)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[5, {LineageId(0, 3)}],
                                     [2, {LineageId(0, 1)}],
                                     [1, {LineageId(0, 0)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_test]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[10, {LineageId(0, 2)}]], columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_standard_scaler():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._data', 'StandardScaler') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
                import numpy as np

                df = pd.DataFrame({'A': [1, 2, 10, 5]})
                standard_scaler = StandardScaler()
                encoded_data = standard_scaler.fit_transform(df)
                expected = np.array([[-1.], [-0.71428571], [1.57142857], [0.14285714]])
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 39), "pd.DataFrame({'A': [1, 2, 10, 5]})"))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                   DagNodeDetails('Standard Scaler', ['array']),
                                   OptionalCodeInfo(CodeReference(6, 18, 6, 34), 'StandardScaler()'))
    expected_dag.add_edge(expected_data_source, expected_transformer)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_transformer]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_kbins_discretizer():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._discretization', 'KBinsDiscretizer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import KBinsDiscretizer
                import numpy as np

                df = pd.DataFrame({'A': [1, 2, 10, 5]})
                discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
                encoded_data = discretizer.fit_transform(df)
                expected = np.array([[0.], [0.], [2.], [1.]])
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 39), "pd.DataFrame({'A': [1, 2, 10, 5]})"))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._discretization',
                                                                'KBinsDiscretizer')),
                                   DagNodeDetails('K-Bins Discretizer', ['array']),
                                   OptionalCodeInfo(CodeReference(6, 14, 6, 78),
                                                    "KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')"))
    expected_dag.add_edge(expected_data_source, expected_transformer)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_transformer]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([0.]), {LineageId(0, 0)}],
                                     [numpy.array([0.]), {LineageId(0, 1)}],
                                     [numpy.array([2.]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_simple_imputer():
    """
    Tests whether the monkey patching of ('sklearn.impute._base’, 'SimpleImputer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.impute import SimpleImputer
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                imputed_data = imputer.fit_transform(df)
                expected = np.array([['cat_a'], ['cat_a'], ['cat_a'], ['cat_c']])
                assert np.array_equal(imputed_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 61),
                                                    "pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})"))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                                   DagNodeDetails('Simple Imputer', ['A']),
                                   OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                                    "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"))
    expected_dag.add_edge(expected_data_source, expected_transformer)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_transformer]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array(['cat_a']), {LineageId(0, 0)}],
                                     [numpy.array(['cat_a']), {LineageId(0, 1)}],
                                     [numpy.array(['cat_a']), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_one_hot_encoder_not_sparse():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._encoders', 'OneHotEncoder') with dense output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, OneHotEncoder
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                one_hot_encoder = OneHotEncoder(sparse=False)
                encoded_data = one_hot_encoder.fit_transform(df)
                expected = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                print(encoded_data)
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                   DagNodeDetails('One-Hot Encoder', ['array']),
                                   OptionalCodeInfo(CodeReference(6, 18, 6, 45), 'OneHotEncoder(sparse=False)'))
    expected_dag.add_edge(expected_data_source, expected_transformer)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_transformer]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_one_hot_encoder_sparse():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._encoders', 'OneHotEncoder') works for sparse output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, OneHotEncoder
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                one_hot_encoder = OneHotEncoder()
                encoded_data = one_hot_encoder.fit_transform(df)
                expected = csr_matrix([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                assert numpy.allclose(encoded_data.A, expected.A) and isinstance(encoded_data, csr_matrix)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(6, 5, 6, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                   DagNodeDetails('One-Hot Encoder', ['array']),
                                   OptionalCodeInfo(CodeReference(7, 18, 7, 33), 'OneHotEncoder()'))
    expected_dag.add_edge(expected_data_source, expected_transformer)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_transformer]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_column_transformer_one_transformer():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    one transformer
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': [1, 2, 10, 5]})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A', 'B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = numpy.array([[-1.], [-0.71428571], [1.57142857], [0.14285714]])
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, columns=['A', 'B']),
                                   OptionalCodeInfo(CodeReference(7, 5, 7, 59),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], 'B': [1, 2, 10, 5]})"))
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 8),
                                  OperatorContext(OperatorType.PROJECTION,
                                                  FunctionInfo('sklearn.compose._column_transformer',
                                                               'ColumnTransformer')),
                                  DagNodeDetails("to ['A', 'B']", ['A', 'B']),
                                  OptionalCodeInfo(CodeReference(8, 21, 10, 2),
                                                   "ColumnTransformer(transformers=[\n"
                                                   "    ('numeric', StandardScaler(), ['A', 'B'])\n])"))
    expected_dag.add_edge(expected_data_source, expected_projection)
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler', ['array']),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'))
    expected_dag.add_edge(expected_projection, expected_standard_scaler)
    expected_concat = DagNode(3,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array']),
                              OptionalCodeInfo(CodeReference(8, 21, 10, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A', 'B'])\n])"))
    expected_dag.add_edge(expected_standard_scaler, expected_concat)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_projection]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[1, 1, {LineageId(0, 0)}],
                                     [2, 2, {LineageId(0, 1)}],
                                     [10, 10, {LineageId(0, 2)}]],
                                    columns=['A', 'B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_standard_scaler]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.0, -1.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143, -0.7142857142857143]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714, 1.5714285714285714]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_concat]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    # TODO: Lineage concat
    expected_lineage_df = DataFrame([[numpy.array([-1.0, -1.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143, -0.7142857142857143]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714, 1.5714285714285714]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_column_transformer_multiple_transformers_all_dense():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with dense output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=False), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [ 1.57142857, 1., 0., 0.], 
                    [0.14285714, 0., 0., 1.]])
                print(encoded_data)
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B']),
                                   OptionalCodeInfo(CodeReference(7, 5, 7, 82),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], "
                                                    "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_projection_1 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['A']", ['A']),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"))
    expected_dag.add_edge(expected_data_source, expected_projection_1)
    expected_projection_2 = DagNode(3,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['B']", ['B']),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"))
    expected_dag.add_edge(expected_data_source, expected_projection_2)
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler', ['array']),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler)
    expected_one_hot = DagNode(4,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder', ['array']),
                               OptionalCodeInfo(CodeReference(10, 20, 10, 47), 'OneHotEncoder(sparse=False)'))
    expected_dag.add_edge(expected_projection_2, expected_one_hot)
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array']),
                              OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"))
    expected_dag.add_edge(expected_standard_scaler, expected_concat)
    expected_dag.add_edge(expected_one_hot, expected_concat)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_projection_1]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[1, {LineageId(0, 0)}],
                                     [2, {LineageId(0, 1)}],
                                     [10, {LineageId(0, 2)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_projection_2]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([['cat_a', {LineageId(0, 0)}],
                                     ['cat_b', {LineageId(0, 1)}],
                                     ['cat_a', {LineageId(0, 2)}]],
                                    columns=['B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_standard_scaler]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_one_hot]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_concat]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    # TODO: Lineage concat
    expected_lineage_df = DataFrame([[numpy.array([-1.0, 1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143, 0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714, 1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_column_transformer_multiple_transformers_sparse_dense():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with sparse and dense mixed output    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=True), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [ 1.57142857, 1., 0., 0.], 
                    [0.14285714, 0., 0., 1.]])
                print(encoded_data)
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B']),
                                   OptionalCodeInfo(CodeReference(7, 5, 7, 82),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], "
                                                    "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_projection_1 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['A']", ['A']),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"))
    expected_dag.add_edge(expected_data_source, expected_projection_1)
    expected_projection_2 = DagNode(3,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['B']", ['B']),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"))
    expected_dag.add_edge(expected_data_source, expected_projection_2)
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler', ['array']),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler)
    expected_one_hot = DagNode(4,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder', ['array']),
                               OptionalCodeInfo(CodeReference(10, 20, 10, 46), 'OneHotEncoder(sparse=True)'))
    expected_dag.add_edge(expected_projection_2, expected_one_hot)
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array']),
                              OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"))
    expected_dag.add_edge(expected_standard_scaler, expected_concat)
    expected_dag.add_edge(expected_one_hot, expected_concat)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_projection_1]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[1, {LineageId(0, 0)}],
                                     [2, {LineageId(0, 1)}],
                                     [10, {LineageId(0, 2)}]],
                                    columns=['A', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_projection_2]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([['cat_a', {LineageId(0, 0)}],
                                     ['cat_b', {LineageId(0, 1)}],
                                     ['cat_a', {LineageId(0, 2)}]],
                                    columns=['B', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_standard_scaler]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_one_hot]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_concat]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    # TODO: Lineage concat
    expected_lineage_df = DataFrame([[numpy.array([-1.0, 1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143, 0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714, 1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_decision_tree():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes', 'DecisionTreeClassifier') works
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
                
                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'target']),
                                   OptionalCodeInfo(CodeReference(6, 5, 6, 95),
                                                    "pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                    "'target': ['no', 'no', 'yes', 'yes']})"))
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler', ['array']),
                                       OptionalCodeInfo(CodeReference(8, 8, 8, 24), 'StandardScaler()'))
    expected_data_projection = DagNode(1,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B']),
                                       OptionalCodeInfo(CodeReference(8, 39, 8, 53), "df[['A', 'B']]"))
    expected_dag.add_edge(expected_data_source, expected_data_projection)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler)
    expected_label_projection = DagNode(3,
                                        BasicCodeLocation("<string-source>", 9),
                                        OperatorContext(OperatorType.PROJECTION,
                                                        FunctionInfo('pandas.core.frame', '__getitem__')),
                                        DagNodeDetails("to ['target']", ['target']),
                                        OptionalCodeInfo(CodeReference(9, 24, 9, 36), "df['target']"))
    expected_dag.add_edge(expected_data_source, expected_label_projection)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 9),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array']),
                                    OptionalCodeInfo(CodeReference(9, 9, 9, 60),
                                                     "label_binarize(df['target'], classes=['no', 'yes'])"))
    expected_dag.add_edge(expected_label_projection, expected_label_encode)
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails('Train Data', ['array']),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                    DagNodeDetails('Train Labels', ['array']),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'))
    expected_dag.add_edge(expected_label_encode, expected_train_labels)
    expected_decision_tree = DagNode(7,
                                     BasicCodeLocation("<string-source>", 11),
                                     OperatorContext(OperatorType.ESTIMATOR,
                                                     FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                     DagNodeDetails('Decision Tree', []),
                                     OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'))
    expected_dag.add_edge(expected_train_data, expected_decision_tree)
    expected_dag.add_edge(expected_train_labels, expected_decision_tree)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train_data]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.3416407864998738, -1.3416407864998738]), {LineageId(0, 0)}],
                                     [numpy.array([-0.4472135954999579, -0.4472135954999579]), {LineageId(0, 1)}],
                                     [numpy.array([0.4472135954999579, 0.4472135954999579]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train_labels]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([0]), {LineageId(0, 0)}],
                                     [numpy.array([0]), {LineageId(0, 1)}],
                                     [numpy.array([1]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_decision_tree]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[{LineageId(0, 0)}],
                                     [{LineageId(0, 1)}],
                                     [{LineageId(0, 2)}]],
                                    columns=['mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True),
                                      check_column_type=False)


def test_logistic_regression():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._logistic', 'LogisticRegression') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import LogisticRegression
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = LogisticRegression()
                clf = clf.fit(train, target)

                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'target']),
                                   OptionalCodeInfo(CodeReference(6, 5, 6, 95),
                                                    "pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                    "'target': ['no', 'no', 'yes', 'yes']})"))
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler', ['array']),
                                       OptionalCodeInfo(CodeReference(8, 8, 8, 24), 'StandardScaler()'))
    expected_data_projection = DagNode(1,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B']),
                                       OptionalCodeInfo(CodeReference(8, 39, 8, 53), "df[['A', 'B']]"))
    expected_dag.add_edge(expected_data_source, expected_data_projection)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler)
    expected_label_projection = DagNode(3,
                                        BasicCodeLocation("<string-source>", 9),
                                        OperatorContext(OperatorType.PROJECTION,
                                                        FunctionInfo('pandas.core.frame', '__getitem__')),
                                        DagNodeDetails("to ['target']", ['target']),
                                        OptionalCodeInfo(CodeReference(9, 24, 9, 36), "df['target']"))
    expected_dag.add_edge(expected_data_source, expected_label_projection)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 9),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array']),
                                    OptionalCodeInfo(CodeReference(9, 9, 9, 60),
                                                     "label_binarize(df['target'], classes=['no', 'yes'])"))
    expected_dag.add_edge(expected_label_projection, expected_label_encode)
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')),
                                  DagNodeDetails('Train Data', ['array']),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.linear_model._logistic',
                                                                 'LogisticRegression')),
                                    DagNodeDetails('Train Labels', ['array']),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'))
    expected_dag.add_edge(expected_label_encode, expected_train_labels)
    expected_estimator = DagNode(7,
                                 BasicCodeLocation("<string-source>", 11),
                                 OperatorContext(OperatorType.ESTIMATOR,
                                                 FunctionInfo('sklearn.linear_model._logistic',
                                                              'LogisticRegression')),
                                 DagNodeDetails('Logistic Regression', []),
                                 OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'))
    expected_dag.add_edge(expected_train_data, expected_estimator)
    expected_dag.add_edge(expected_train_labels, expected_estimator)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train_data]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.3416407864998738, -1.3416407864998738]), {LineageId(0, 0)}],
                                     [numpy.array([-0.4472135954999579, -0.4472135954999579]), {LineageId(0, 1)}],
                                     [numpy.array([0.4472135954999579, 0.4472135954999579]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train_labels]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([0]), {LineageId(0, 0)}],
                                     [numpy.array([0]), {LineageId(0, 1)}],
                                     [numpy.array([1]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_estimator]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[{LineageId(0, 0)}],
                                     [{LineageId(0, 1)}],
                                     [{LineageId(0, 2)}]],
                                    columns=['mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True),
                                      check_column_type=False)


def test_keras_wrapper():
    """
    Tests whether the monkey patching of ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])
                
                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(9, activation='relu', input_dim=input_dim))
                    clf.add(Dense(9, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                clf = KerasClassifier(build_fn=create_model, epochs=2, batch_size=1, verbose=0, input_dim=2)
                clf.fit(train, target)

                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                assert test_predict.shape == (2,)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 9),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'target']),
                                   OptionalCodeInfo(CodeReference(9, 5, 9, 95),
                                                    "pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                    "'target': ['no', 'no', 'yes', 'yes']})"))
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 11),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler', ['array']),
                                       OptionalCodeInfo(CodeReference(11, 8, 11, 24), 'StandardScaler()'))
    expected_data_projection = DagNode(1,
                                       BasicCodeLocation("<string-source>", 11),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B']),
                                       OptionalCodeInfo(CodeReference(11, 39, 11, 53), "df[['A', 'B']]"))
    expected_dag.add_edge(expected_data_source, expected_data_projection)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler)
    expected_label_projection = DagNode(3,
                                        BasicCodeLocation("<string-source>", 12),
                                        OperatorContext(OperatorType.PROJECTION,
                                                        FunctionInfo('pandas.core.frame', '__getitem__')),
                                        DagNodeDetails("to ['target']", ['target']),
                                        OptionalCodeInfo(CodeReference(12, 51, 12, 65), "df[['target']]"))
    expected_dag.add_edge(expected_data_source, expected_label_projection)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 12),
                                    OperatorContext(OperatorType.TRANSFORMER,
                                                    FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                    DagNodeDetails('One-Hot Encoder', ['array']),
                                    OptionalCodeInfo(CodeReference(12, 9, 12, 36), 'OneHotEncoder(sparse=False)'))
    expected_dag.add_edge(expected_label_projection, expected_label_encode)
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 22),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Train Data', ['array']),
                                  OptionalCodeInfo(CodeReference(22, 6, 22, 92),
                                                   'KerasClassifier(build_fn=create_model, epochs=2, '
                                                   'batch_size=1, verbose=0, input_dim=2)'))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 22),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                                 'KerasClassifier')),
                                    DagNodeDetails('Train Labels', ['array']),
                                    OptionalCodeInfo(CodeReference(22, 6, 22, 92),
                                                     'KerasClassifier(build_fn=create_model, epochs=2, '
                                                     'batch_size=1, verbose=0, input_dim=2)'))
    expected_dag.add_edge(expected_label_encode, expected_train_labels)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 22),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Neural Network', []),
                                  OptionalCodeInfo(CodeReference(22, 6, 22, 92),
                                                   'KerasClassifier(build_fn=create_model, epochs=2, '
                                                   'batch_size=1, verbose=0, input_dim=2)'))
    expected_dag.add_edge(expected_train_data, expected_classifier)
    expected_dag.add_edge(expected_train_labels, expected_classifier)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train_data]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.3416407864998738, -1.3416407864998738]), {LineageId(0, 0)}],
                                     [numpy.array([-0.4472135954999579, -0.4472135954999579]), {LineageId(0, 1)}],
                                     [numpy.array([0.4472135954999579, 0.4472135954999579]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_train_labels]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1., 0.]), {LineageId(0, 0)}],
                                     [numpy.array([1., 0.]), {LineageId(0, 1)}],
                                     [numpy.array([0., 1.]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_classifier]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[{LineageId(0, 0)}],
                                     [{LineageId(0, 1)}],
                                     [{LineageId(0, 2)}]],
                                    columns=['mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True),
                                      check_column_type=False)
