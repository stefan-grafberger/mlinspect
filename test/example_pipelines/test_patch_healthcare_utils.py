"""
Tests whether the monkey patching works for all patched sklearn methods
"""
from inspect import cleandoc

import networkx
import numpy
import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect.inspections._lineage import RowLineage, LineageId
from example_pipelines.healthcare import custom_monkeypatching


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
                                                        inspections=[RowLineage(3)],
                                                        custom_monkey_patching=[custom_monkeypatching])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_estimator = DagNode(1,
                              BasicCodeLocation("<string-source>", 6),
                              OperatorContext(OperatorType.TRANSFORMER,
                                              FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                           'MyW2VTransformer')),
                              DagNodeDetails('Word2Vec', ['array']),
                              OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                               'MyW2VTransformer(min_count=2, size=2, workers=1)'))
    expected_dag.add_edge(expected_data_source, expected_estimator)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_estimator]
    lineage_output = inspection_results_data_source[RowLineage(3)]
    expected_lineage_df = DataFrame([[numpy.array([0.0, 0.0, 0.0]), {LineageId(0, 0)}],
                                     [numpy.array([0.0, 0.0, 0.0]), {LineageId(0, 1)}],
                                     [numpy.array([0.0, 0.0, 0.0]), {LineageId(0, 2)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_series_equal(lineage_output["mlinspect_lineage"], expected_lineage_df["mlinspect_lineage"])
    assert expected_lineage_df.iloc[0, 0].shape == (3,)
