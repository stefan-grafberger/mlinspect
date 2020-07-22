"""
Tests whether the WIR extraction works
"""
import ast
from inspect import cleandoc
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


def test_print_var_usage():
    """
    Tests whether the WIR Extraction works for a very simple var usage
    """
    test_code = cleandoc("""
        test_var = "test"
        print(test_var)""")
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_constant = Vertex(0, "test", [], "Constant_None")
    expected_assign = Vertex(1, "test_var", [expected_constant], "Assign")
    expected_call = Vertex(2, "print", [expected_assign], "Call")
    expected_graph = [expected_constant, expected_assign, expected_call]
    assert extracted_dag == expected_graph


def test_string_call_attribute_():
    """
    Tests whether the WIR Extraction works for a very simple attribute call
    """
    test_code = cleandoc("""
        "hello ".join("world")
        """)
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_constant_one = Vertex(0, "hello ", [], "Constant_None")
    expected_constant_two = Vertex(1, "world", [], "Constant_None")
    expected_attribute_call = Vertex(2, "join", [expected_constant_one, expected_constant_two], "Call")
    expected_graph = [expected_constant_one, expected_constant_two, expected_attribute_call]
    assert extracted_dag == expected_graph
