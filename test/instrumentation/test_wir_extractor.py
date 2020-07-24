"""
Tests whether the WIR extraction works
"""
import ast
import os
from inspect import cleandoc
from mlinspect.utils import get_project_root
from mlinspect.instrumentation.wir_extractor import WirExtractor, Vertex
import networkx

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_print_stmt():
    """
    Tests whether the WIR Extraction works for a very simple print statement
    """
    test_code = "print('test')"
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant = Vertex(0, "test", "Constant")
    expected_call = Vertex(1, "print", "Call")
    expected_graph.add_edge(expected_constant, expected_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_print_var_usage():
    """
    Tests whether the WIR Extraction works for a very simple var usage
    """
    test_code = cleandoc("""
        test_var = "test"
        print(test_var)""")
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant = Vertex(0, "test", "Constant")
    expected_assign = Vertex(1, "test_var", "Assign")
    expected_graph.add_edge(expected_constant, expected_assign, type="input")

    expected_call = Vertex(2, "print", "Call")
    expected_graph.add_node(expected_call)
    expected_graph.add_edge(expected_assign, expected_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_string_call_attribute():
    """
    Tests whether the WIR Extraction works for a very simple attribute call
    """
    test_code = cleandoc("""
        "hello ".join("world")
        """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant_one = Vertex(0, "hello ", "Constant")
    expected_constant_two = Vertex(1, "world", "Constant")
    expected_attribute_call = Vertex(2, "join", "Call")
    expected_graph.add_edge(expected_constant_one, expected_attribute_call, type="caller")
    expected_graph.add_edge(expected_constant_two, expected_attribute_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_print_expressions():
    """
    Tests whether the WIR Extraction works for an expression with very simple nested calls
    """
    test_code = cleandoc("""
        print("test".isupper())
        """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant = Vertex(0, "test", "Constant")
    expected_call_one = Vertex(1, "isupper", "Call")
    expected_call_two = Vertex(2, "print", "Call")
    expected_graph.add_edge(expected_constant, expected_call_one, type="caller")
    expected_graph.add_edge(expected_call_one, expected_call_two, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_keyword():
    """
    Tests whether the WIR Extraction works for function calls with keyword usage
    """
    test_code = cleandoc("""
        print('comma', 'separated', 'words', sep=', ')
        """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant_one = Vertex(0, "comma", "Constant")
    expected_constant_two = Vertex(1, "separated", "Constant")
    expected_constant_three = Vertex(2, "words", "Constant")
    expected_constant_four = Vertex(3, ", ", "Constant")
    expected_keyword = Vertex(4, "sep", "Keyword")
    expected_call = Vertex(5, "print", "Call")

    expected_graph.add_edge(expected_constant_four, expected_keyword, type="input")
    expected_graph.add_edge(expected_constant_one, expected_call, type="input")
    expected_graph.add_edge(expected_constant_two, expected_call, type="input")
    expected_graph.add_edge(expected_constant_three, expected_call, type="input")
    expected_graph.add_edge(expected_keyword, expected_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_import():
    """
    Tests whether the WIR Extraction works for imports
    """
    test_code = cleandoc("""
        import math 
        
        math.sqrt(4)
        """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_import = Vertex(0, "math", None, [], "Import")
    expected_constant = Vertex(1, "4", None, [], "Constant")
    expected_constant_call = Vertex(2, "sqrt", expected_import, [expected_constant], "Call")
    expected_graph = [expected_import, expected_constant, expected_constant_call]
    assert extracted_wir == expected_graph


def test_import_as():
    """
    Tests whether the WIR Extraction works for imports as
    """
    test_code = cleandoc("""
            import math as test 

            test.sqrt(4)
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_import = Vertex(0, "math", None, [], "Import")
    expected_constant = Vertex(1, "4", None, [], "Constant")
    expected_constant_call = Vertex(2, "sqrt", expected_import, [expected_constant], "Call")
    expected_graph = [expected_import, expected_constant, expected_constant_call]
    assert extracted_wir == expected_graph


def test_import_from():
    """
    Tests whether the WIR Extraction works for from imports
    """
    test_code = cleandoc("""
            from math import sqrt 

            sqrt(4)
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_import = Vertex(0, "math", None, [], "Import")
    expected_constant = Vertex(1, "4", None, [], "Constant")
    expected_constant_call = Vertex(2, "sqrt", expected_import, [expected_constant], "Call")
    expected_graph = [expected_import, expected_constant, expected_constant_call]
    assert extracted_wir == expected_graph


def test_nested_import_from():
    """
    Tests whether the WIR Extraction works for nested from imports
    """
    test_code = cleandoc("""
            from mlinspect.utils import get_project_root

            print(get_project_root())
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_import = Vertex(0, "mlinspect.utils", None, [], "Import")
    expected_call_one = Vertex(1, "get_project_root", expected_import, [], "Call")
    expected_call_two = Vertex(2, "print", None, [expected_call_one], "Call")
    expected_graph = [expected_import, expected_call_one, expected_call_two]
    assert extracted_wir == expected_graph


def test_list_creation():
    """
    Tests whether the WIR Extraction works for lists
    """
    test_code = cleandoc("""
            print(["test1", "test2"])
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_constant_one = Vertex(0, "test1", None, [], "Constant")
    expected_constant_two = Vertex(1, "test2", None, [], "Constant")
    expected_list = Vertex(2, "as_list", None, [expected_constant_one, expected_constant_two], "List")
    expected_call = Vertex(3, "print", None, [expected_list], "Call")
    expected_graph = [expected_constant_one, expected_constant_two, expected_list, expected_call]
    assert extracted_wir == expected_graph


def test_index_subscript():
    """
    Tests whether the WIR Extraction works for lists
    """
    test_code = cleandoc("""
            import pandas as pd
            
            data = pd.read_csv('test_path')
            data['income-per-year']
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_import = Vertex(0, "pandas", None, [], "Import")
    expected_constant_one = Vertex(1, "test_path", None, [], "Constant")
    expected_call = Vertex(2, "read_csv", expected_import, [expected_constant_one], "Call")
    expected_assign = Vertex(3, "data", None, [expected_call], "Assign")
    expected_constant_two = Vertex(4, "income-per-year", None, [], "Constant")
    expected_index_subscript = Vertex(5, "Index-Subscript", expected_assign, [expected_constant_two], "Subscript")
    expected_graph = [expected_import, expected_constant_one, expected_call, expected_assign,
                      expected_constant_two, expected_index_subscript]
    assert extracted_wir == expected_graph


def test_tuples():
    """
    Tests whether the WIR Extraction works for tuples
    """
    test_code = cleandoc("""
            from sklearn import preprocessing

            ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), ['education', 'workclass'])
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_import_from = Vertex(0, "sklearn", None, [], "Import")
    expected_constant_one = Vertex(1, "categorical", None, [], "Constant")
    expected_constant_two = Vertex(2, "ignore", None, [], "Constant")
    expected_keyword = Vertex(3, "handle_unknown", None, [expected_constant_two], "Keyword")
    expected_call = Vertex(4, "OneHotEncoder", expected_import_from, [expected_keyword], "Call")
    expected_constant_three = Vertex(5, "education", None, [], "Constant")
    expected_constant_four = Vertex(6, "workclass", None, [], "Constant")
    expected_list = Vertex(7, "as_list", None, [expected_constant_three, expected_constant_four], "List")
    expected_tuple = Vertex(8, "as_tuple", None, [expected_constant_one, expected_call, expected_list], "Tuple")
    expected_graph = [expected_import_from, expected_constant_one, expected_constant_two, expected_keyword,
                      expected_call, expected_constant_three, expected_constant_four, expected_list, expected_tuple]
    assert extracted_wir == expected_graph


def test_adult_easy_pipeline():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
        extractor = WirExtractor(test_ast)
        extracted_wir = extractor.extract_wir()

        assert len(extracted_wir) == 59

# TODO: Functions with multiple return values, function definitions, control flow, and other edge cases
