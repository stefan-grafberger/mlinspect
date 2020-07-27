"""
Tests whether the DagVertex works
"""
from mlinspect.instrumentation.dag_vertex import DagVertex


def test_dag_repr():
    """
    Tests whether the DagVertex works
    """
    vertex = DagVertex(1, "Data Source", 1, 5, ('pandas.io.parsers', 'read_csv'))
    assert str(vertex) == "DagVertex(node_id=1: name='Data Source', " \
                          "lineno=1, col_offset=5, module=('pandas.io.parsers', 'read_csv'))"
