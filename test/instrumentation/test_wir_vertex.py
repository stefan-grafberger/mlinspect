"""
Tests whether the WirVertex works
"""
from mlinspect.instrumentation.wir_vertex import WirVertex


def test_wir_repr():
    """
    Tests whether the WirVertex works
    """
    vertex = WirVertex(1, "read_csv", "Call", 1, 5, ('pandas.io.parsers', 'read_csv'))
    assert str(vertex) == "WirVertex(node_id=1: name='read_csv', operation='Call', " \
                          "lineno=1, col_offset=5, module=('pandas.io.parsers', 'read_csv'))"
