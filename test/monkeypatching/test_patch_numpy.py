"""
Tests whether the monkey patching works for all patched pandas methods
"""
from inspect import cleandoc

import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect import OperatorContext, FunctionInfo, OperatorType
from mlinspect.inspections._lineage import RowLineage, LineageId
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo


def test_numpy_random():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'DataFrame') works
    """
    test_code = cleandoc("""
        import numpy as np
        test = np.random.random(100)
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[RowLineage(2)])
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    # expected_node = DagNode(0,
    #                         BasicCodeLocation("<string-source>", 3),
    #                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
    #                         DagNodeDetails(None, ['A']),
    #                         OptionalCodeInfo(CodeReference(3, 5, 3, 43), "pd.DataFrame([0, 1, 2], columns=['A'])"))
    # compare(extracted_node, expected_node)
    #
    # inspection_results_data_source = inspector_result.dag_node_to_inspection_results[extracted_node]
    # lineage_output = inspection_results_data_source[RowLineage(2)]
    # expected_lineage_df = DataFrame([[0, {LineageId(0, 0)}],
    #                                  [1, {LineageId(0, 1)}]],
    #                                 columns=['A', 'mlinspect_lineage'])
    # pandas.testing.assert_frame_equal(lineage_output.reset_index(drop=True), expected_lineage_df.reset_index(drop=True))
