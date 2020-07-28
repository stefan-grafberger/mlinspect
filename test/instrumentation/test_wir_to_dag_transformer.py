"""
Tests whether the DAG extraction works
"""
import ast
import os
import networkx

from mlinspect.instrumentation.wir_to_dag_transformer import WirToDagTransformer
from mlinspect.instrumentation.wir_vertex import WirVertex
from mlinspect.utils import get_project_root
from mlinspect.instrumentation.wir_extractor import WirExtractor
from ..utils import get_expected_dag_adult_easy_py

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_remove_all_nodes_but_calls_and_subscripts():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
        extractor = WirExtractor(test_ast)
        extractor.extract_wir()
        extracted_wir_with_module_info = extractor.add_call_module_info(get_module_info())

        cleaned_wir = WirToDagTransformer().remove_all_nodes_but_calls_and_subscripts(extracted_wir_with_module_info)

        assert len(cleaned_wir) == 15

        expected_graph = get_expected_cleaned_wir_adult_easy()

        assert networkx.to_dict_of_dicts(cleaned_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_remove_all_non_operators_and_update_names():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
        extractor = WirExtractor(test_ast)
        extractor.extract_wir()
        extracted_wir_with_module_info = extractor.add_call_module_info(get_module_info())

        cleaned_wir = WirToDagTransformer().remove_all_nodes_but_calls_and_subscripts(extracted_wir_with_module_info)
        dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

        assert len(dag) == 4

        expected_graph = get_expected_dag_adult_easy_py()

        assert networkx.to_dict_of_dicts(cleaned_wir) == networkx.to_dict_of_dicts(expected_graph)


def test_sklearn_wir_preprocessing():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
        extractor = WirExtractor(test_ast)
        extractor.extract_wir()
        extracted_wir_with_module_info = extractor.add_call_module_info(get_module_info())

        preprocessed_wir = WirToDagTransformer().sklearn_wir_preprocessing(extracted_wir_with_module_info)
        cleaned_wir = WirToDagTransformer().remove_all_nodes_but_calls_and_subscripts(preprocessed_wir)
        dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

        # FIXME
       # assert len(dag) == 4

        expected_graph = get_expected_dag_adult_easy_py()

        assert networkx.to_dict_of_dicts(cleaned_wir) == networkx.to_dict_of_dicts(expected_graph)


def get_module_info():
    """
    Get the module info for the adult_easy pipeline
    """
    module_info = {(10, 0): ('builtins', 'print'),
                   (11, 13): ('posixpath', 'join'),
                   (11, 26): ('builtins', 'str'),
                   (11, 30): ('mlinspect.utils', 'get_project_root'),
                   (12, 11): ('pandas.io.parsers', 'read_csv'),
                   (14, 7): ('pandas.core.frame', 'dropna'),
                   (16, 9): ('sklearn.preprocessing._label', 'label_binarize'),
                   (16, 38): ('pandas.core.frame', '__getitem__'),
                   (18, 25): ('sklearn.compose._column_transformer', 'ColumnTransformer'),
                   (19, 20): ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
                   (20, 16): ('sklearn.preprocessing._data', 'StandardScaler'),
                   (24, 18): ('sklearn.pipeline', 'Pipeline'),
                   (26, 19): ('sklearn.tree._classes', 'DecisionTreeClassifier'),
                   (28, 0): ('sklearn.pipeline', 'fit'),
                   (31, 0): ('builtins', 'print')}

    return module_info


def get_expected_cleaned_wir_adult_easy():
    """
    Get the expected cleaned WIR for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    expected_graph = networkx.DiGraph()

    expected_print_one = WirVertex(6, "print", "Call", 10, 0, ('builtins', 'print'))
    expected_graph.add_node(expected_print_one)

    expected_get_project_root = WirVertex(7, "get_project_root", "Call", 11, 30,
                                          ('mlinspect.utils', 'get_project_root'))
    expected_str = WirVertex(8, "str", "Call", 11, 26, ('builtins', 'str'))
    expected_graph.add_edge(expected_get_project_root, expected_str)

    expected_join = WirVertex(12, "join", "Call", 11, 13, ('posixpath', 'join'))
    expected_graph.add_edge(expected_str, expected_join)

    expected_read_csv = WirVertex(18, "read_csv", "Call", 12, 11, ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_edge(expected_join, expected_read_csv)

    expected_dropna = WirVertex(20, "dropna", "Call", 14, 7, ('pandas.core.frame', 'dropna'))
    expected_graph.add_edge(expected_read_csv, expected_dropna)

    expected_fit = WirVertex(56, "fit", "Call", 28, 0, ('sklearn.pipeline', 'fit'))
    expected_index_subscript = WirVertex(23, "Index-Subscript", "Subscript", 16, 38,
                                         ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_dropna, expected_fit)
    expected_graph.add_edge(expected_dropna, expected_index_subscript)

    expected_label_binarize = WirVertex(28, "label_binarize", "Call", 16, 9,
                                        ('sklearn.preprocessing._label', 'label_binarize'))
    expected_graph.add_edge(expected_index_subscript, expected_label_binarize)
    expected_graph.add_edge(expected_label_binarize, expected_fit)

    expected_one_hot_encoder = WirVertex(33, "OneHotEncoder", "Call", 19, 20,
                                         ('sklearn.preprocessing._encoders', 'OneHotEncoder'))
    expected_standard_scaler = WirVertex(39, "StandardScaler", "Call", 20, 16,
                                         ('sklearn.preprocessing._data', 'StandardScaler'))
    expected_column_transformer = WirVertex(46, "ColumnTransformer", "Call", 18, 25,
                                            ('sklearn.compose._column_transformer', 'ColumnTransformer'))
    expected_graph.add_edge(expected_one_hot_encoder, expected_column_transformer)
    expected_graph.add_edge(expected_standard_scaler, expected_column_transformer)

    expected_decision_tree_classifier = WirVertex(51, "DecisionTreeClassifier", "Call", 26, 19,
                                                  ('sklearn.tree._classes', 'DecisionTreeClassifier'))
    expected_pipeline = WirVertex(54, "Pipeline", "Call", 24, 18, ('sklearn.pipeline', 'Pipeline'))
    expected_graph.add_edge(expected_column_transformer, expected_pipeline)
    expected_graph.add_edge(expected_decision_tree_classifier, expected_pipeline)
    expected_graph.add_edge(expected_pipeline, expected_fit)

    expected_print_two = WirVertex(58, "print", "Call", 31, 0, ('builtins', 'print'))
    expected_graph.add_node(expected_print_two)

    return expected_graph
