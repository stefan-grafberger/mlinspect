"""
Tests whether the monkey patching works for all patched statsmodels methods
"""
from inspect import cleandoc

import networkx
import numpy
import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect import OperatorContext, FunctionInfo, OperatorType
from mlinspect.inspections._lineage import RowLineage, LineageId
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo


def test_statsmodels_add_constant():
    """
    Tests whether the monkey patching of ('statsmodel.api', 'add_constant') works
    """
    test_code = cleandoc("""
        import numpy as np
        import statsmodels.api as sm
        np.random.seed(42)
        test = np.random.random(100)
        test = sm.add_constant(test)
        assert len(test) == 100
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])

    expected_dag = networkx.DiGraph()
    expected_random = DagNode(0,
                              BasicCodeLocation("<string-source>", 4),
                              OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('numpy.random', 'random')),
                              DagNodeDetails('random', ['array']),
                              OptionalCodeInfo(CodeReference(4, 7, 4, 28), "np.random.random(100)"))

    expected_constant = DagNode(1,
                                BasicCodeLocation("<string-source>", 5),
                                OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                FunctionInfo('statsmodel.api', 'add_constant')),
                                DagNodeDetails('Adds const column', ['array']),
                                OptionalCodeInfo(CodeReference(5, 7, 5, 28), "sm.add_constant(test)"))
    expected_dag.add_edge(expected_random, expected_constant)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_random]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[0.5, {LineageId(0, 0)}],
                                     [0.5, {LineageId(0, 1)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True),
                                      atol=1)

    inspection_results_data_source = inspector_result.dag_node_to_inspection_results[expected_constant]
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame([[numpy.array([0.5, 1.]), {LineageId(0, 0)}],
                                     [numpy.array([0.5, 1.]), {LineageId(0, 1)}]],
                                    columns=['array', 'mlinspect_lineage'])
    pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True),
                                      atol=1)
