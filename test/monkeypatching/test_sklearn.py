"""
Tests whether the monkey patching works for all patched sklearn methods
"""
from inspect import cleandoc

import networkx
import numpy
import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, OperatorType, CodeReference
from mlinspect.inspections._lineage import RowLineage, LineageId


def test_label_binarize():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 5, OperatorType.DATA_SOURCE,
                                  ('pandas.core.series', 'Series'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(5, 12, 5, 59),
                                  optional_source_code="pd.Series(['yes', 'no', 'no', 'yes'], name='A')")
    expected_select = DagNode(1, "<string-source>", 6, OperatorType.PROJECTION_MODIFY,
                              module=('sklearn.preprocessing._label', 'label_binarize'),
                              description="label_binarize, classes: ['no', 'yes']", columns=['array'],
                              optional_code_reference=CodeReference(6, 12, 6, 60),
                              optional_source_code="label_binarize(pd_series, classes=['no', 'yes'])")
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1]), {LineageId(0, 0)}],
                                     [numpy.array([0]), {LineageId(0, 1)}],
                                     [numpy.array([0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_standard_scaler():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 5, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(5, 5, 5, 39),
                                  optional_source_code="pd.DataFrame({'A': [1, 2, 10, 5]})")
    expected_select = DagNode(1, "<string-source>", 6, OperatorType.TRANSFORMER,
                              module=('sklearn.preprocessing._data', 'StandardScaler'),
                              description='Standard Scaler', columns=['array'],
                              optional_code_reference=CodeReference(6, 18, 6, 34),
                              optional_source_code='StandardScaler()')
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([-1.0]), {LineageId(0, 0)}],
                                     [numpy.array([-0.7142857142857143]), {LineageId(0, 1)}],
                                     [numpy.array([1.5714285714285714]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_kbins_discretizer():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_data_source = DagNode(0, "<string-source>", 5, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(5, 5, 5, 39),
                                  optional_source_code="pd.DataFrame({'A': [1, 2, 10, 5]})")
    expected_discretizer = DagNode(1, "<string-source>", 6, OperatorType.TRANSFORMER,
                              module=('sklearn.preprocessing._discretization', 'KBinsDiscretizer'),
                              description='K-Bins Discretizer', columns=['array'],
                              optional_code_reference=CodeReference(6, 14, 6, 78),
                              optional_source_code="KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')")
    expected_dag.add_edge(expected_data_source, expected_discretizer)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_discretizer]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([0.]), {LineageId(0, 0)}],
                                     [numpy.array([0.]), {LineageId(0, 1)}],
                                     [numpy.array([2.]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_simple_imputer():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 5, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(5, 5, 5, 61),
                                  optional_source_code="pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})")
    expected_select = DagNode(1, "<string-source>", 6, OperatorType.TRANSFORMER,
                              module=('sklearn.impute._base', 'SimpleImputer'),
                              description='Simple Imputer', columns=['array'],
                              optional_code_reference=CodeReference(6, 10, 6, 72),
                              optional_source_code="SimpleImputer(missing_values=np.nan, strategy='most_frequent')")
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array(['cat_a']), {LineageId(0, 0)}],
                                     [numpy.array(['cat_a']), {LineageId(0, 1)}],
                                     [numpy.array(['cat_a']), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_one_hot_encoder_not_sparse():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 5, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(5, 5, 5, 62),
                                  optional_source_code="pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})")
    expected_select = DagNode(1, "<string-source>", 6, OperatorType.TRANSFORMER,
                              module=('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                              description='One-Hot Encoder', columns=['array'],
                              optional_code_reference=CodeReference(6, 18, 6, 45),
                              optional_source_code='OneHotEncoder(sparse=False)')
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_one_hot_encoder_sparse():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 6, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(6, 5, 6, 62),
                                  optional_source_code="pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})")
    expected_select = DagNode(1, "<string-source>", 7, OperatorType.TRANSFORMER,
                              module=('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                              description='One-Hot Encoder', columns=['array'],
                              optional_code_reference=CodeReference(7, 18, 7, 33),
                              optional_source_code='OneHotEncoder()')
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 1.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([1.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))


def test_column_transformer_one_transformer():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 7, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B'],
                                  optional_code_reference=CodeReference(7, 5, 7, 59),
                                  optional_source_code="pd.DataFrame({'A': [1, 2, 10, 5], 'B': [1, 2, 10, 5]})")
    expected_projection = DagNode(1, "<string-source>", 8, OperatorType.PROJECTION,
                                  module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                  description="to ['A', 'B']", columns=['A', 'B'],
                                  optional_code_reference=CodeReference(8, 21, 10, 2),
                                  optional_source_code="ColumnTransformer(transformers=[\n    "
                                                       "('numeric', StandardScaler(), ['A', 'B'])\n])")
    expected_dag.add_edge(expected_missing_op, expected_projection)
    expected_standard_scaler = DagNode(2, "<string-source>", 9, OperatorType.TRANSFORMER,
                                       module=('sklearn.preprocessing._data', 'StandardScaler'),
                                       description='Standard Scaler', columns=['array'],
                                       optional_code_reference=CodeReference(9, 16, 9, 32),
                                       optional_source_code='StandardScaler()')
    expected_dag.add_edge(expected_projection, expected_standard_scaler)
    expected_concat = DagNode(3, "<string-source>", 8, OperatorType.CONCATENATION,
                              module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                              description='', columns=['array'],
                              optional_code_reference=CodeReference(8, 21, 10, 2),
                              optional_source_code="ColumnTransformer(transformers=[\n    "
                                                   "('numeric', StandardScaler(), ['A', 'B'])\n])")
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
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 7, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B'],
                                  optional_code_reference=CodeReference(7, 5, 7, 82),
                                  optional_source_code="pd.DataFrame({'A': [1, 2, 10, 5], "
                                                       "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})")
    expected_projection_1 = DagNode(1, "<string-source>", 8, OperatorType.PROJECTION,
                                    module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                    description="to ['A']", columns=['A'],
                                    optional_code_reference=CodeReference(8, 21, 11, 2),
                                    optional_source_code="ColumnTransformer(transformers=[\n"
                                                         "    ('numeric', StandardScaler(), ['A']),\n"
                                                         "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])")
    expected_dag.add_edge(expected_missing_op, expected_projection_1)
    expected_projection_2 = DagNode(3, "<string-source>", 8, OperatorType.PROJECTION,
                                    module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                    description="to ['B']", columns=['B'],
                                    optional_code_reference=CodeReference(8, 21, 11, 2),
                                    optional_source_code="ColumnTransformer(transformers=[\n"
                                                         "    ('numeric', StandardScaler(), ['A']),\n"
                                                         "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])")
    expected_dag.add_edge(expected_missing_op, expected_projection_2)
    expected_standard_scaler = DagNode(2, "<string-source>", 9, OperatorType.TRANSFORMER,
                                       module=('sklearn.preprocessing._data', 'StandardScaler'),
                                       description='Standard Scaler', columns=['array'],
                                       optional_code_reference=CodeReference(9, 16, 9, 32),
                                       optional_source_code='StandardScaler()')
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler)
    expected_one_hot = DagNode(4, "<string-source>", 10, OperatorType.TRANSFORMER,
                               module=('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                               description='One-Hot Encoder', columns=['array'],
                               optional_code_reference=CodeReference(10, 20, 10, 47),
                               optional_source_code='OneHotEncoder(sparse=False)')
    expected_dag.add_edge(expected_projection_2, expected_one_hot)
    expected_concat = DagNode(5, "<string-source>", 8, OperatorType.CONCATENATION,
                              module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                              description='', columns=['array'],
                              optional_code_reference=CodeReference(8, 21, 11, 2),
                              optional_source_code="ColumnTransformer(transformers=[\n"
                                                   "    ('numeric', StandardScaler(), ['A']),\n"
                                                   "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])")
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
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_missing_op = DagNode(0, "<string-source>", 7, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B'],
                                  optional_code_reference=CodeReference(7, 5, 7, 82),
                                  optional_source_code="pd.DataFrame({'A': [1, 2, 10, 5], "
                                                       "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})")
    expected_projection_1 = DagNode(1, "<string-source>", 8, OperatorType.PROJECTION,
                                    module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                    description="to ['A']", columns=['A'],
                                    optional_code_reference=CodeReference(8, 21, 11, 2),
                                    optional_source_code="ColumnTransformer(transformers=[\n"
                                                         "    ('numeric', StandardScaler(), ['A']),\n"
                                                         "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])")
    expected_dag.add_edge(expected_missing_op, expected_projection_1)
    expected_projection_2 = DagNode(3, "<string-source>", 8, OperatorType.PROJECTION,
                                    module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                                    description="to ['B']", columns=['B'],
                                    optional_code_reference=CodeReference(8, 21, 11, 2),
                                    optional_source_code="ColumnTransformer(transformers=[\n"
                                                         "    ('numeric', StandardScaler(), ['A']),\n"
                                                         "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])")
    expected_dag.add_edge(expected_missing_op, expected_projection_2)
    expected_standard_scaler = DagNode(2, "<string-source>", 9, OperatorType.TRANSFORMER,
                                       module=('sklearn.preprocessing._data', 'StandardScaler'),
                                       description='Standard Scaler', columns=['array'],
                                       optional_code_reference=CodeReference(9, 16, 9, 32),
                                       optional_source_code='StandardScaler()')
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler)
    expected_one_hot = DagNode(4, "<string-source>", 10, OperatorType.TRANSFORMER,
                               module=('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                               description='One-Hot Encoder', columns=['array'],
                               optional_code_reference=CodeReference(10, 20, 10, 46),
                               optional_source_code='OneHotEncoder(sparse=True)')
    expected_dag.add_edge(expected_projection_2, expected_one_hot)
    expected_concat = DagNode(5, "<string-source>", 8, OperatorType.CONCATENATION,
                              module=('sklearn.compose._column_transformer', 'ColumnTransformer'),
                              description='', columns=['array'],
                              optional_code_reference=CodeReference(8, 21, 11, 2),
                              optional_source_code="ColumnTransformer(transformers=[\n"
                                                   "    ('numeric', StandardScaler(), ['A']),\n"
                                                   "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])")
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
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_data_source = DagNode(0, "<string-source>", 6, OperatorType.DATA_SOURCE,
                                   ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B', 'target'],
                                   optional_code_reference=CodeReference(6, 5, 6, 95),
                                   optional_source_code="pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                        "'target': ['no', 'no', 'yes', 'yes']})")
    expected_standard_scaler = DagNode(2, "<string-source>", 8, OperatorType.TRANSFORMER,
                                       module=('sklearn.preprocessing._data', 'StandardScaler'),
                                       description='Standard Scaler', columns=['array'],
                                       optional_code_reference=CodeReference(8, 8, 8, 24),
                                       optional_source_code='StandardScaler()')
    expected_data_projection = DagNode(1, "<string-source>", 8, OperatorType.PROJECTION,
                                       module=('pandas.core.frame', '__getitem__'),
                                       description="to ['A', 'B']", columns=['A', 'B'],
                                       optional_code_reference=CodeReference(8, 39, 8, 53),
                                       optional_source_code="df[['A', 'B']]")
    expected_dag.add_edge(expected_data_source, expected_data_projection)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler)
    expected_label_projection = DagNode(3, "<string-source>", 9, OperatorType.PROJECTION,
                                        module=('pandas.core.frame', '__getitem__'),
                                        description="to ['target']", columns=['target'],
                                        optional_code_reference=CodeReference(9, 24, 9, 36),
                                        optional_source_code="df['target']")
    expected_dag.add_edge(expected_data_source, expected_label_projection)
    expected_label_encode = DagNode(4, "<string-source>", 9, OperatorType.PROJECTION_MODIFY,
                                    module=('sklearn.preprocessing._label', 'label_binarize'),
                                    description="label_binarize, classes: ['no', 'yes']", columns=['array'],
                                    optional_code_reference=CodeReference(9, 9, 9, 60),
                                    optional_source_code="label_binarize(df['target'], classes=['no', 'yes'])")
    expected_dag.add_edge(expected_label_projection, expected_label_encode)
    expected_train_data = DagNode(5, "<string-source>", 11, OperatorType.TRAIN_DATA,
                                  module=('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                  description='Train Data', columns=['array'],
                                  optional_code_reference=CodeReference(11, 6, 11, 30),
                                  optional_source_code='DecisionTreeClassifier()')
    expected_dag.add_edge(expected_standard_scaler, expected_train_data)
    expected_train_labels = DagNode(6, "<string-source>", 11, OperatorType.TRAIN_LABELS,
                                    module=('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                    description='Train Labels', columns=['array'],
                                    optional_code_reference=CodeReference(11, 6, 11, 30),
                                    optional_source_code='DecisionTreeClassifier()')
    expected_dag.add_edge(expected_label_encode, expected_train_labels)
    expected_decision_tree = DagNode(7, "<string-source>", 11, OperatorType.ESTIMATOR,
                                     module=('sklearn.tree._classes', 'DecisionTreeClassifier'),
                                     description='Decision Tree', columns=[],
                                     optional_code_reference=CodeReference(11, 6, 11, 30),
                                     optional_source_code='DecisionTreeClassifier()')
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
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
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
    expected_data_source = DagNode(0, "<string-source>", 6, OperatorType.DATA_SOURCE,
                                   ('pandas.core.frame', 'DataFrame'), description='', columns=['A', 'B', 'target'],
                                   optional_code_reference=CodeReference(6, 5, 6, 95),
                                   optional_source_code="pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                        "'target': ['no', 'no', 'yes', 'yes']})")
    expected_standard_scaler = DagNode(2, "<string-source>", 8, OperatorType.TRANSFORMER,
                                       module=('sklearn.preprocessing._data', 'StandardScaler'),
                                       description='Standard Scaler', columns=['array'],
                                       optional_code_reference=CodeReference(8, 8, 8, 24),
                                       optional_source_code='StandardScaler()')
    expected_data_projection = DagNode(1, "<string-source>", 8, OperatorType.PROJECTION,
                                       module=('pandas.core.frame', '__getitem__'),
                                       description="to ['A', 'B']", columns=['A', 'B'],
                                       optional_code_reference=CodeReference(8, 39, 8, 53),
                                       optional_source_code="df[['A', 'B']]")
    expected_dag.add_edge(expected_data_source, expected_data_projection)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler)
    expected_label_projection = DagNode(3, "<string-source>", 9, OperatorType.PROJECTION,
                                        module=('pandas.core.frame', '__getitem__'),
                                        description="to ['target']", columns=['target'],
                                        optional_code_reference=CodeReference(9, 24, 9, 36),
                                        optional_source_code="df['target']")
    expected_dag.add_edge(expected_data_source, expected_label_projection)
    expected_label_encode = DagNode(4, "<string-source>", 9, OperatorType.PROJECTION_MODIFY,
                                    module=('sklearn.preprocessing._label', 'label_binarize'),
                                    description="label_binarize, classes: ['no', 'yes']", columns=['array'],
                                    optional_code_reference=CodeReference(9, 9, 9, 60),
                                    optional_source_code="label_binarize(df['target'], classes=['no', 'yes'])")
    expected_dag.add_edge(expected_label_projection, expected_label_encode)
    expected_train_data = DagNode(5, "<string-source>", 11, OperatorType.TRAIN_DATA,
                                  module=('sklearn.linear_model._logistic', 'LogisticRegression'),
                                  description='Train Data', columns=['array'],
                                  optional_code_reference=CodeReference(11, 6, 11, 26),
                                  optional_source_code='LogisticRegression()')
    expected_dag.add_edge(expected_standard_scaler, expected_train_data)
    expected_train_labels = DagNode(6, "<string-source>", 11, OperatorType.TRAIN_LABELS,
                                    module=('sklearn.linear_model._logistic', 'LogisticRegression'),
                                    description='Train Labels', columns=['array'],
                                    optional_code_reference=CodeReference(11, 6, 11, 26),
                                    optional_source_code='LogisticRegression()')
    expected_dag.add_edge(expected_label_encode, expected_train_labels)
    expected_decision_tree = DagNode(7, "<string-source>", 11, OperatorType.ESTIMATOR,
                                     module=('sklearn.linear_model._logistic', 'LogisticRegression'),
                                     description='Logistic Regression', columns=[],
                                     optional_code_reference=CodeReference(11, 6, 11, 26),
                                     optional_source_code='LogisticRegression()')
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
