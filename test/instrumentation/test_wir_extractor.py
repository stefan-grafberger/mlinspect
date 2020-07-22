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


def test_import():
    """
    Tests whether the WIR Extraction works for imports
    """
    test_code = cleandoc("""
        import math 
        
        math.sqrt(4)
        """)
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_import = Vertex(0, "math", [], "Import")
    expected_constant = Vertex(1, "4", [], "Constant_None")
    expected_constant_call = Vertex(2, "sqrt", [expected_import, expected_constant], "Call")
    expected_graph = [expected_import, expected_constant, expected_constant_call]
    assert extracted_dag == expected_graph


def test_import_as():
    """
    Tests whether the WIR Extraction works for imports as
    """
    test_code = cleandoc("""
            import math as test 

            test.sqrt(4)
            """)
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_import = Vertex(0, "math", [], "Import")
    expected_constant = Vertex(1, "4", [], "Constant_None")
    expected_constant_call = Vertex(2, "sqrt", [expected_import, expected_constant], "Call")
    expected_graph = [expected_import, expected_constant, expected_constant_call]
    assert extracted_dag == expected_graph


def test_import_from():
    """
    Tests whether the WIR Extraction works for from imports
    """
    test_code = cleandoc("""
            from math import sqrt 

            sqrt(4)
            """)
    extractor = WirExtractor()
    test_ast = ast.parse(test_code)
    extracted_dag = extractor.extract_wir(test_ast)
    expected_import = Vertex(0, "math", [], "Import")
    expected_constant = Vertex(1, "4", [], "Constant_None")
    expected_constant_call = Vertex(2, "sqrt", [expected_import, expected_constant], "Call")
    expected_graph = [expected_import, expected_constant, expected_constant_call]
    assert extracted_dag == expected_graph

    # TODO: nested import-froms like 'from mlinspect.utils import get_project_root',
    #  List-Constants, actual pipeline code, nested calls, functions with multiple return values
    # Maybe mark caller explicitly?

    # We can also see if we really need runtime detection to check where a function comes from.
    # Makes things a lot easier if we use it for class member variables however. Maybe in a later step.
    # Also, maybe a knowledge base is possible. Then we can have a map from function
    # to operator and mark which parameters contain important operator attributes
    # Also: mark root nodes. then we can only list root nodes in graph. then we can
    # calculate children from the parents. then we can update the
    # print functions to print children.
