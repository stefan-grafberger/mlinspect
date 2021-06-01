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
