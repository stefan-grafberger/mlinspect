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


def test_string_call_attribute():
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


def test_print_expressions():
    """
    Tests whether the WIR Extraction works for an expression with very simple nested calls
    """
    test_code = cleandoc("""
        print("test".isupper())
        """)
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_constant = Vertex(0, "test", [], "Constant_None")
    expected_call_one = Vertex(1, "isupper", [expected_constant], "Call")
    expected_call_two = Vertex(2, "print", [expected_call_one], "Call")
    expected_graph = [expected_constant, expected_call_one, expected_call_two]
    assert extracted_dag == expected_graph


def test_keyword():
    """
    Tests whether the WIR Extraction works for function calls with keyword usage
    """
    test_code = cleandoc("""
        print('comma', 'separated', 'words', sep=', ')
        """)
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_constant_one = Vertex(0, "comma", [], "Constant_None")
    expected_constant_two = Vertex(1, "separated", [], "Constant_None")
    expected_constant_three = Vertex(2, "words", [], "Constant_None")
    expected_constant_four = Vertex(3, ", ", [], "Constant_None")
    expected_keyword = Vertex(4, "sep", [expected_constant_four], "Keyword")
    expected_call = Vertex(5, "print", [expected_constant_one, expected_constant_two, expected_constant_three,
                                        expected_keyword], "Call")
    expected_graph = [expected_constant_one, expected_constant_two, expected_constant_three, expected_constant_four,
                      expected_keyword, expected_call]
    assert extracted_dag == expected_graph
