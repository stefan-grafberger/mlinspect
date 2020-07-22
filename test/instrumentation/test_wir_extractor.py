"""
Tests whether the WIR extraction works
"""
import ast
from mlinspect.instrumentation.wir_extractor import WirExtractor, Vertex


def test_print_stmt():
    """
    Tests whether the WIR Extraction works for a very simple print statement
    """
    extractor = WirExtractor()
    test_ast = ast.parse("print('test')")
    extracted_dag = extractor.extract_wir(test_ast)

    expected_constant = Vertex(0, "test", [], "Constant_None")
    expected_call = Vertex(1, "print", [expected_constant], "Call")
    expected_graph = [expected_constant, expected_call]
    assert extracted_dag == expected_graph
