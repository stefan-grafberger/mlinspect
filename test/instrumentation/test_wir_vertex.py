"""
Tests whether the WirVertex works
"""
from mlinspect.instrumentation.wir_vertex import WirVertex


def test_wir_repr():
    """
    Tests whether the .py version of the inspector works
    """
    vertex = WirVertex(1, "read_csv", "Call", ('pandas.io.parsers', 'read_csv'))
    assert str(vertex) == "(node_id=1: vertex_name='read_csv', op='Call', " \
                          "module=('pandas.io.parsers', 'read_csv'))"
