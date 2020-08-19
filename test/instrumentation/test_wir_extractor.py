"""
Tests whether the WIR extraction works
"""
import ast
import os
from inspect import cleandoc
import networkx
import pytest
from testfixtures import compare

from mlinspect.instrumentation.dag_node import CodeReference
from mlinspect.utils import get_project_root
from mlinspect.instrumentation.wir_extractor import WirExtractor
from mlinspect.instrumentation.wir_node import WirNode

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

    expected_constant = WirNode(0, "test", "Constant", CodeReference(1, 6, 1, 12))
    expected_call = WirNode(1, "print", "Call", CodeReference(1, 0, 1, 13))
    expected_graph.add_edge(expected_constant, expected_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_constant = WirNode(0, "test", "Constant", CodeReference(1, 11, 1, 17))
    expected_assign = WirNode(1, "test_var", "Assign", CodeReference(1, 0, 1, 17))
    expected_graph.add_edge(expected_constant, expected_assign, type="input", arg_index=0)

    expected_call = WirNode(2, "print", "Call", CodeReference(2, 0, 2, 15))
    expected_graph.add_node(expected_call)
    expected_graph.add_edge(expected_assign, expected_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


def test_tuple_assign():
    """
    Tests whether the WIR Extraction works for a very simple var usage
    """
    test_code = cleandoc("""
        x, y = (1, 2)
        print(x)""")
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant_one = WirNode(1, '1', 'Constant', CodeReference(1, 8, 1, 9))
    expected_constant_two = WirNode(2, '2', 'Constant', CodeReference(1, 11, 1, 12))

    expetected_constant_tuple = WirNode(3, 'as_tuple', 'Tuple', CodeReference(1, 7, 1, 13))
    expected_graph.add_edge(expected_constant_one, expetected_constant_tuple, type="input", arg_index=0)
    expected_graph.add_edge(expected_constant_two, expetected_constant_tuple, type="input", arg_index=1)

    expected_var_x = WirNode(4, 'x', 'Assign', CodeReference(1, 0, 1, 1))
    expected_var_y = WirNode(5, 'y', 'Assign', CodeReference(1, 3, 1, 4))
    expected_graph.add_edge(expetected_constant_tuple, expected_var_x, type="input", arg_index=0)
    expected_graph.add_edge(expetected_constant_tuple, expected_var_y, type="input", arg_index=0)

    expected_call = WirNode(6, 'print', 'Call', CodeReference(2, 0, 2, 8))
    expected_graph.add_edge(expected_var_x, expected_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_constant_one = WirNode(0, "hello ", "Constant", CodeReference(1, 0, 1, 8))
    expected_constant_two = WirNode(1, "world", "Constant", CodeReference(1, 14, 1, 21))
    expected_attribute_call = WirNode(2, "join", "Call", CodeReference(1, 0, 1, 22))
    expected_graph.add_edge(expected_constant_one, expected_attribute_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant_two, expected_attribute_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


def test_call_after_call():
    """
    Tests whether the WIR Extraction works for a very simple attribute call
    """
    test_code = cleandoc("""
        "hello ".capitalize().count()
        """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_constant_one = WirNode(0, "hello ", "Constant", CodeReference(1, 0, 1, 8))
    expected_call_one = WirNode(1, "capitalize", "Call", CodeReference(1, 0, 1, 21))
    expected_call_two = WirNode(2, "count", "Call", CodeReference(1, 0, 1, 29))
    expected_graph.add_edge(expected_constant_one, expected_call_one, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_call_one, expected_call_two, type="caller", arg_index=-1)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_constant = WirNode(0, "test", "Constant", CodeReference(1, 6, 1, 12))
    expected_call_one = WirNode(1, "isupper", "Call", CodeReference(1, 6, 1, 22))
    expected_graph.add_edge(expected_constant, expected_call_one, type="caller", arg_index=-1)

    expected_call_two = WirNode(2, "print", "Call", CodeReference(1, 0, 1, 23))
    expected_graph.add_edge(expected_call_one, expected_call_two, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_constant_one = WirNode(0, "comma", "Constant", CodeReference(1, 6, 1, 13))
    expected_constant_two = WirNode(1, "separated", "Constant", CodeReference(1, 15, 1, 26))
    expected_constant_three = WirNode(2, "words", "Constant", CodeReference(1, 28, 1, 35))
    expected_constant_four = WirNode(3, ", ", "Constant", CodeReference(1, 41, 1, 45))
    expected_keyword = WirNode(4, "sep", "Keyword")
    expected_call = WirNode(5, "print", "Call", CodeReference(1, 0, 1, 46))

    expected_graph.add_edge(expected_constant_four, expected_keyword, type="input", arg_index=0)
    expected_graph.add_edge(expected_constant_one, expected_call, type="input", arg_index=0)
    expected_graph.add_edge(expected_constant_two, expected_call, type="input", arg_index=1)
    expected_graph.add_edge(expected_constant_three, expected_call, type="input", arg_index=2)
    expected_graph.add_edge(expected_keyword, expected_call, type="input", arg_index=3)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_import = WirNode(0, "math", "Import", CodeReference(1, 0, 1, 11))
    expected_constant = WirNode(1, "4", "Constant", CodeReference(3, 10, 3, 11))
    expected_constant_call = WirNode(2, "sqrt", "Call", CodeReference(3, 0, 3, 12))
    expected_graph.add_edge(expected_import, expected_constant_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant, expected_constant_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_import = WirNode(0, "math", "Import", CodeReference(1, 0, 1, 19))
    expected_constant = WirNode(1, "4", "Constant", CodeReference(3, 10, 3, 11))
    expected_constant_call = WirNode(2, "sqrt", "Call", CodeReference(3, 0, 3, 12))
    expected_graph.add_edge(expected_import, expected_constant_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant, expected_constant_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_import = WirNode(0, "math", "Import", CodeReference(1, 0, 1, 21))
    expected_constant = WirNode(1, "4", "Constant", CodeReference(3, 5, 3, 6))
    expected_constant_call = WirNode(2, "sqrt", "Call", CodeReference(3, 0, 3, 7))
    expected_graph.add_edge(expected_import, expected_constant_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant, expected_constant_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_import = WirNode(0, "mlinspect.utils", "Import", CodeReference(1, 0, 1, 44))
    expected_call_one = WirNode(1, "get_project_root", "Call", CodeReference(3, 6, 3, 24))
    expected_graph.add_edge(expected_import, expected_call_one, type="caller", arg_index=-1)

    expected_call_two = WirNode(2, "print", "Call", CodeReference(3, 0, 3, 25))
    expected_graph.add_edge(expected_call_one, expected_call_two, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_constant_one = WirNode(0, "test1", "Constant", CodeReference(1, 7, 1, 14))
    expected_constant_two = WirNode(1, "test2", "Constant", CodeReference(1, 16, 1, 23))
    expected_list = WirNode(2, "as_list", "List", CodeReference(1, 6, 1, 24))
    expected_graph.add_edge(expected_constant_one, expected_list, type="input", arg_index=0)
    expected_graph.add_edge(expected_constant_two, expected_list, type="input", arg_index=1)

    expected_call = WirNode(3, "print", "Call", CodeReference(1, 0, 1, 25))
    expected_graph.add_edge(expected_list, expected_call, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_import = WirNode(0, "pandas", "Import", CodeReference(1, 0, 1, 19))
    expected_constant_one = WirNode(1, "test_path", "Constant", CodeReference(3, 19, 3, 30))
    expected_call = WirNode(2, "read_csv", "Call", CodeReference(3, 7, 3, 31))
    expected_graph.add_edge(expected_import, expected_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant_one, expected_call, type="input", arg_index=0)

    expected_assign = WirNode(3, "data", "Assign", CodeReference(3, 0, 3, 31))
    expected_graph.add_edge(expected_call, expected_assign, type="input", arg_index=0)

    expected_constant_two = WirNode(4, "income-per-year", "Constant", CodeReference(4, 5, 4, 22))
    expected_index_subscript = WirNode(5, "Index-Subscript", "Subscript", CodeReference(4, 0, 4, 23))
    expected_graph.add_edge(expected_assign, expected_index_subscript, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant_two, expected_index_subscript, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


def test_index_assign():
    """
    Tests whether the WIR Extraction works for lists
    """
    test_code = cleandoc("""
            import pandas as pd

            data = pd.read_csv('test_path')
            data['label'] = "test"
            """)
    test_ast = ast.parse(test_code)
    extractor = WirExtractor(test_ast)
    extracted_wir = extractor.extract_wir()
    expected_graph = networkx.DiGraph()

    expected_import = WirNode(0, "pandas", "Import", CodeReference(1, 0, 1, 19))
    expected_constant_one = WirNode(1, "test_path", "Constant", CodeReference(3, 19, 3, 30))
    expected_call = WirNode(2, "read_csv", "Call", CodeReference(3, 7, 3, 31))
    expected_graph.add_edge(expected_import, expected_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant_one, expected_call, type="input", arg_index=0)

    expected_assign = WirNode(3, "data", "Assign", CodeReference(3, 0, 3, 31))
    expected_graph.add_edge(expected_call, expected_assign, type="input", arg_index=0)

    expected_constant_two = WirNode(4, "label", "Constant", CodeReference(4, 5, 4, 12))
    expected_graph.add_node(expected_constant_two)

    expected_constant_three = WirNode(5, "test", "Constant", CodeReference(4, 16, 4, 22))
    expected_graph.add_node(expected_constant_three)

    expected_subscript_assign = WirNode(6, 'data.label', 'Subscript-Assign', CodeReference(4, 0, 4, 13))
    expected_graph.add_edge(expected_assign, expected_subscript_assign, type="caller", arg_index=-1)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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

    expected_import_from = WirNode(0, "sklearn", "Import", CodeReference(1, 0, 1, 33))
    expected_constant_one = WirNode(1, "categorical", "Constant", CodeReference(3, 1, 3, 14))
    expected_constant_two = WirNode(2, "ignore", "Constant", CodeReference(3, 59, 3, 67))
    expected_keyword = WirNode(3, "handle_unknown", "Keyword")
    expected_graph.add_edge(expected_constant_two, expected_keyword, type="input", arg_index=0)

    expected_call = WirNode(4, "OneHotEncoder", "Call", CodeReference(3, 16, 3, 68))
    expected_graph.add_edge(expected_import_from, expected_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_keyword, expected_call, type="input", arg_index=0)

    expected_constant_three = WirNode(5, "education", "Constant", CodeReference(3, 71, 3, 82))
    expected_constant_four = WirNode(6, "workclass", "Constant", CodeReference(3, 84, 3, 95))
    expected_list = WirNode(7, "as_list", "List", CodeReference(3, 70, 3, 96))
    expected_graph.add_edge(expected_constant_three, expected_list, type="input", arg_index=0)
    expected_graph.add_edge(expected_constant_four, expected_list, type="input", arg_index=1)

    expected_tuple = WirNode(8, "as_tuple", "Tuple", CodeReference(3, 0, 3, 97))
    expected_graph.add_edge(expected_constant_one, expected_tuple, type="input", arg_index=0)
    expected_graph.add_edge(expected_call, expected_tuple, type="input", arg_index=1)
    expected_graph.add_edge(expected_list, expected_tuple, type="input", arg_index=2)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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
        CodeReference(3, 7, 3, 31): ('pandas.io.parsers', 'read_csv'),
        CodeReference(4, 0, 4, 23): ('pandas.core.frame', '__getitem__')
    }
    extracted_wir = extractor.extract_wir()
    extractor.add_runtime_info(module_info, {})
    expected_graph = networkx.DiGraph()

    expected_import = WirNode(0, "pandas", "Import", CodeReference(1, 0, 1, 19))
    expected_constant_one = WirNode(1, "test_path", "Constant", CodeReference(3, 19, 3, 30))
    expected_call = WirNode(2, "read_csv", "Call", CodeReference(3, 7, 3, 31), ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_edge(expected_import, expected_call, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant_one, expected_call, type="input", arg_index=0)

    expected_assign = WirNode(3, "data", "Assign", CodeReference(3, 0, 3, 31))
    expected_graph.add_edge(expected_call, expected_assign, type="input", arg_index=0)

    expected_constant_two = WirNode(4, "income-per-year", "Constant", CodeReference(4, 5, 4, 22))
    expected_index_subscript = WirNode(5, "Index-Subscript", "Subscript", CodeReference(4, 0, 4, 23),
                                       ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_assign, expected_index_subscript, type="caller", arg_index=-1)
    expected_graph.add_edge(expected_constant_two, expected_index_subscript, type="input", arg_index=0)

    compare(networkx.to_dict_of_dicts(extracted_wir), networkx.to_dict_of_dicts(expected_graph))


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
