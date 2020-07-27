"""
Tests whether the WIR extraction works
"""
import ast
import os
from inspect import cleandoc
import networkx
import pytest
from mlinspect.utils import get_project_root
from mlinspect.instrumentation.wir_extractor import WirExtractor
from mlinspect.instrumentation.wir_vertex import WirVertex

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

    expected_constant = WirVertex(0, "test", "Constant", 1, 6)
    expected_call = WirVertex(1, "print", "Call", 1, 0)
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

    expected_constant = WirVertex(0, "test", "Constant", 1, 11)
    expected_assign = WirVertex(1, "test_var", "Assign", 1, 0)
    expected_graph.add_edge(expected_constant, expected_assign, type="input")

    expected_call = WirVertex(2, "print", "Call", 2, 0)
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

    expected_constant_one = WirVertex(0, "hello ", "Constant", 1, 0)
    expected_constant_two = WirVertex(1, "world", "Constant", 1, 14)
    expected_attribute_call = WirVertex(2, "join", "Call", 1, 0)
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

    expected_constant = WirVertex(0, "test", "Constant", 1, 6)
    expected_call_one = WirVertex(1, "isupper", "Call", 1, 6)
    expected_graph.add_edge(expected_constant, expected_call_one, type="caller")

    expected_call_two = WirVertex(2, "print", "Call", 1, 0)
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

    expected_constant_one = WirVertex(0, "comma", "Constant", 1, 6)
    expected_constant_two = WirVertex(1, "separated", "Constant", 1, 15)
    expected_constant_three = WirVertex(2, "words", "Constant", 1, 28)
    expected_constant_four = WirVertex(3, ", ", "Constant", 1, 41)
    expected_keyword = WirVertex(4, "sep", "Keyword")
    expected_call = WirVertex(5, "print", "Call", 1, 0)

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
    expected_graph = networkx.DiGraph()

    expected_import = WirVertex(0, "math", "Import", 1, 0)
    expected_constant = WirVertex(1, "4", "Constant", 3, 10)
    expected_constant_call = WirVertex(2, "sqrt", "Call", 3, 0)
    expected_graph.add_edge(expected_import, expected_constant_call, type="caller")
    expected_graph.add_edge(expected_constant, expected_constant_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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
    expected_graph = networkx.DiGraph()

    expected_import = WirVertex(0, "math", "Import", 1, 0)
    expected_constant = WirVertex(1, "4", "Constant", 3, 10)
    expected_constant_call = WirVertex(2, "sqrt", "Call", 3, 0)
    expected_graph.add_edge(expected_import, expected_constant_call, type="caller")
    expected_graph.add_edge(expected_constant, expected_constant_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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
    expected_graph = networkx.DiGraph()

    expected_import = WirVertex(0, "math", "Import", 1, 0)
    expected_constant = WirVertex(1, "4", "Constant", 3, 5)
    expected_constant_call = WirVertex(2, "sqrt", "Call", 3, 0)
    expected_graph.add_edge(expected_import, expected_constant_call, type="caller")
    expected_graph.add_edge(expected_constant, expected_constant_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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
    expected_graph = networkx.DiGraph()

    expected_import = WirVertex(0, "mlinspect.utils", "Import", 1, 0)
    expected_call_one = WirVertex(1, "get_project_root", "Call", 3, 6)
    expected_graph.add_edge(expected_import, expected_call_one, type="caller")

    expected_call_two = WirVertex(2, "print", "Call", 3, 0)
    expected_graph.add_edge(expected_call_one, expected_call_two, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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
    expected_graph = networkx.DiGraph()

    expected_constant_one = WirVertex(0, "test1", "Constant", 1, 7)
    expected_constant_two = WirVertex(1, "test2", "Constant", 1, 16)
    expected_list = WirVertex(2, "as_list", "List", 1, 6)
    expected_graph.add_edge(expected_constant_one, expected_list, type="input")
    expected_graph.add_edge(expected_constant_two, expected_list, type="input")

    expected_call = WirVertex(3, "print", "Call", 1, 0)
    expected_graph.add_edge(expected_list, expected_call, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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
    expected_graph = networkx.DiGraph()

    expected_import = WirVertex(0, "pandas", "Import", 1, 0)
    expected_constant_one = WirVertex(1, "test_path", "Constant", 3, 19)
    expected_call = WirVertex(2, "read_csv", "Call", 3, 7)
    expected_graph.add_edge(expected_import, expected_call, type="caller")
    expected_graph.add_edge(expected_constant_one, expected_call, type="input")

    expected_assign = WirVertex(3, "data", "Assign", 3, 0)
    expected_graph.add_edge(expected_call, expected_assign, type="input")

    expected_constant_two = WirVertex(4, "income-per-year", "Constant", 4, 5)
    expected_index_subscript = WirVertex(5, "Index-Subscript", "Subscript", 4, 0)
    expected_graph.add_edge(expected_assign, expected_index_subscript, type="caller")
    expected_graph.add_edge(expected_constant_two, expected_index_subscript, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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
    expected_graph = networkx.DiGraph()

    expected_import_from = WirVertex(0, "sklearn", "Import", 1, 0)
    expected_constant_one = WirVertex(1, "categorical", "Constant", 3, 1)
    expected_constant_two = WirVertex(2, "ignore", "Constant", 3, 59)
    expected_keyword = WirVertex(3, "handle_unknown", "Keyword")
    expected_graph.add_edge(expected_constant_two, expected_keyword, type="input")

    expected_call = WirVertex(4, "OneHotEncoder", "Call", 3, 16)
    expected_graph.add_edge(expected_import_from, expected_call, type="caller")
    expected_graph.add_edge(expected_keyword, expected_call, type="input")

    expected_constant_three = WirVertex(5, "education", "Constant", 3, 71)
    expected_constant_four = WirVertex(6, "workclass", "Constant", 3, 84)
    expected_list = WirVertex(7, "as_list", "List", 3, 70)
    expected_graph.add_edge(expected_constant_three, expected_list, type="input")
    expected_graph.add_edge(expected_constant_four, expected_list, type="input")

    expected_tuple = WirVertex(8, "as_tuple", "Tuple", 3, 0)
    expected_graph.add_edge(expected_constant_one, expected_tuple, type="input")
    expected_graph.add_edge(expected_call, expected_tuple, type="input")
    expected_graph.add_edge(expected_list, expected_tuple, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


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


def test_index_subscript_with_module_information():
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
    module_info = {
        (3, 7): ('pandas.io.parsers', 'read_csv'),
        (4, 0): ('pandas.core.frame', '__getitem__')
    }
    extracted_wir = extractor.extract_wir()
    extractor.add_call_module_info(module_info)
    expected_graph = networkx.DiGraph()

    expected_import = WirVertex(0, "pandas", "Import", 1, 0)
    expected_constant_one = WirVertex(1, "test_path", "Constant", 3, 19)
    expected_call = WirVertex(2, "read_csv", "Call", 3, 7, ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_edge(expected_import, expected_call, type="caller")
    expected_graph.add_edge(expected_constant_one, expected_call, type="input")

    expected_assign = WirVertex(3, "data", "Assign", 3, 0)
    expected_graph.add_edge(expected_call, expected_assign, type="input")

    expected_constant_two = WirVertex(4, "income-per-year", "Constant", 4, 5)
    expected_index_subscript = WirVertex(5, "Index-Subscript", "Subscript", 4, 0, ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_assign, expected_index_subscript, type="caller")
    expected_graph.add_edge(expected_constant_two, expected_index_subscript, type="input")

    assert networkx.to_dict_of_dicts(extracted_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_fails_for_unknown_ast_node_types():
    """
    Tests whether the WIR Extraction fails properly for node types it can not handle yet
    """
    test_code = "print(*['hello ', 'world'])"
    parsed_ast = ast.parse(test_code)
    exec(compile(parsed_ast, filename="<ast>", mode="exec"), {})

    extractor = WirExtractor(parsed_ast)
    with pytest.raises(AssertionError):
        extractor.extract_wir()

# TODO: Functions with multiple return values, function definitions, control flow, and other edge cases
