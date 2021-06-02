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


def test_my_word_to_vec_transformer():
    """
    Tests whether the monkey patching of ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                word_to_vec = MyW2VTransformer(min_count=2, size=2, workers=1)
                encoded_data = word_to_vec.fit_transform(df)
                assert encoded_data.shape == (4, 2)
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(3)])

    expected_dag = networkx.DiGraph()
    expected_missing_op = DagNode(0, "<string-source>", 5, OperatorType.DATA_SOURCE,
                                  ('pandas.core.frame', 'DataFrame'), description='', columns=['A'],
                                  optional_code_reference=CodeReference(5, 5, 5, 62),
                                  optional_source_code="pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})")
    expected_select = DagNode(1, "<string-source>", 6, OperatorType.TRANSFORMER,
                              module=('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'),
                              description='Word2Vec', columns=['array'],
                              optional_code_reference=CodeReference(6, 14, 6, 62),
                              optional_source_code='MyW2VTransformer(min_count=2, size=2, workers=1)')
    expected_dag.add_edge(expected_missing_op, expected_select)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_select]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([0.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 0.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([0.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_series_equal(lineage_output["mlinspect_lineage"], expected_lineage_df["mlinspect_lineage"])
    assert expected_lineage_df.iloc[0, 0].shape == (3,)


def test_my_keras_wrapper():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works for df arguments
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from example_pipelines.healthcare.healthcare_utils import MyKerasClassifier, create_model
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])
                
                clf = MyKerasClassifier(build_fn=create_model, epochs=2, batch_size=1, verbose=0)
                clf.fit(train, target)

                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                assert test_predict.shape == (2,)
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
                                        optional_code_reference=CodeReference(9, 51, 9, 65),
                                        optional_source_code="df[['target']]")
    expected_dag.add_edge(expected_data_source, expected_label_projection)
    expected_label_encode = DagNode(4, "<string-source>", 9, OperatorType.TRANSFORMER,
                                    module=('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                                    description='One-Hot Encoder', columns=['array'],
                                    optional_code_reference=CodeReference(9, 9, 9, 36),
                                    optional_source_code='OneHotEncoder(sparse=False)')
    expected_dag.add_edge(expected_label_projection, expected_label_encode)
    expected_train_data = DagNode(5, "<string-source>", 11, OperatorType.TRAIN_DATA,
                                  module=('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'),
                                  description='Train Data', columns=['array'],
                                  optional_code_reference=CodeReference(11, 6, 11, 81),
                                  optional_source_code='MyKerasClassifier(build_fn=create_model, epochs=2, '
                                                       'batch_size=1, verbose=0)')
    expected_dag.add_edge(expected_standard_scaler, expected_train_data)
    expected_train_labels = DagNode(6, "<string-source>", 11, OperatorType.TRAIN_LABELS,
                                    module=('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'),
                                    description='Train Labels', columns=['array'],
                                    optional_code_reference=CodeReference(11, 6, 11, 81),
                                    optional_source_code='MyKerasClassifier(build_fn=create_model, epochs=2, '
                                                         'batch_size=1, verbose=0)')
    expected_dag.add_edge(expected_label_encode, expected_train_labels)
    expected_decision_tree = DagNode(7, "<string-source>", 11, OperatorType.ESTIMATOR,
                                     module=('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer'),
                                     description='Neural Network', columns=[],
                                     optional_code_reference=CodeReference(11, 6, 11, 81),
                                     optional_source_code='MyKerasClassifier(build_fn=create_model, epochs=2, '
                                                          'batch_size=1, verbose=0)')
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
    expected_lineage_df = DataFrame([[numpy.array([1., 0.]), {LineageId(0, 0)}],
                                     [numpy.array([1., 0.]), {LineageId(0, 1)}],
                                     [numpy.array([0., 1.]), {LineageId(0, 2)}]],
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
