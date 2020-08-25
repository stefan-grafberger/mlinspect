"""
Tests whether the DAG extraction works
"""
import networkx
from testfixtures import compare

from mlinspect.instrumentation.dag_node import CodeReference
from mlinspect.backends.sklearn_wir_preprocessor import SklearnWirPreprocessor
from mlinspect.instrumentation.wir_to_dag_transformer import WirToDagTransformer
from mlinspect.instrumentation.wir_node import WirNode
from mlinspect.instrumentation.wir_extractor import WirExtractor
from ..utils import get_expected_dag_adult_easy_py, get_module_info, get_adult_easy_py_ast, get_test_wir


def test_remove_all_nodes_but_calls_and_subscripts():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    test_ast = get_adult_easy_py_ast()
    extractor = WirExtractor(test_ast)
    extractor.extract_wir()
    extracted_wir_with_module_info = extractor.add_runtime_info(get_module_info(), {})

    cleaned_wir = WirToDagTransformer().remove_all_nodes_but_calls_and_subscripts(extracted_wir_with_module_info)

    assert len(cleaned_wir) == 15

    expected_graph = get_expected_cleaned_wir_adult_easy()

    compare(networkx.to_dict_of_dicts(cleaned_wir), networkx.to_dict_of_dicts(expected_graph))


def test_remove_all_non_operators_and_update_names():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    preprocessed_wir = SklearnWirPreprocessor().preprocess_wir(get_test_wir())
    cleaned_wir = WirToDagTransformer().remove_all_nodes_but_calls_and_subscripts(preprocessed_wir)
    dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

    assert len(dag) == 17

    expected_graph = get_expected_dag_adult_easy_py()

    assert networkx.to_dict_of_dicts(cleaned_wir) == networkx.to_dict_of_dicts(expected_graph)


def get_expected_cleaned_wir_adult_easy():
    """
    Get the expected cleaned WIR for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    expected_graph = networkx.DiGraph()

    expected_print_one = WirNode(6, "print", "Call", CodeReference(10, 0, 10, 23), ('builtins', 'print'))
    expected_graph.add_node(expected_print_one)

    expected_get_project_root = WirNode(7, "get_project_root", "Call", CodeReference(11, 30, 11, 48),
                                        ('mlinspect.utils', 'get_project_root'))
    expected_str = WirNode(8, "str", "Call", CodeReference(11, 26, 11, 49), ('builtins', 'str'))
    expected_graph.add_edge(expected_get_project_root, expected_str)

    expected_join = WirNode(12, "join", "Call", CodeReference(11, 13, 11, 85), ('posixpath', 'join'))
    expected_graph.add_edge(expected_str, expected_join)

    expected_read_csv = WirNode(18, "read_csv", "Call", CodeReference(12, 11, 12, 62),
                                ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_edge(expected_join, expected_read_csv)

    expected_dropna = WirNode(20, "dropna", "Call", CodeReference(14, 7, 14, 24), ('pandas.core.frame', 'dropna'))
    expected_graph.add_edge(expected_read_csv, expected_dropna)

    expected_fit = WirNode(56, "fit", "Call", CodeReference(28, 0, 28, 33), ('sklearn.pipeline', 'fit'))
    expected_index_subscript = WirNode(23, "Index-Subscript", "Subscript", CodeReference(16, 38, 16, 61),
                                       ('pandas.core.frame', '__getitem__', 'Projection'))
    expected_graph.add_edge(expected_dropna, expected_fit)
    expected_graph.add_edge(expected_dropna, expected_index_subscript)

    expected_label_binarize = WirNode(28, "label_binarize", "Call", CodeReference(16, 9, 16, 89),
                                      ('sklearn.preprocessing._label', 'label_binarize'))
    expected_graph.add_edge(expected_index_subscript, expected_label_binarize)
    expected_graph.add_edge(expected_label_binarize, expected_fit)

    expected_one_hot_encoder = WirNode(33, "OneHotEncoder", "Call", CodeReference(19, 20, 19, 72),
                                       ('sklearn.preprocessing._encoders', 'OneHotEncoder'))
    expected_standard_scaler = WirNode(39, "StandardScaler", "Call", CodeReference(20, 16, 20, 46),
                                       ('sklearn.preprocessing._data', 'StandardScaler'))
    expected_column_transformer = WirNode(46, "ColumnTransformer", "Call", CodeReference(18, 25, 21, 2),
                                          ('sklearn.compose._column_transformer', 'ColumnTransformer'))
    expected_graph.add_edge(expected_one_hot_encoder, expected_column_transformer)
    expected_graph.add_edge(expected_standard_scaler, expected_column_transformer)

    expected_decision_tree_classifier = WirNode(51, "DecisionTreeClassifier", "Call", CodeReference(26, 19, 26, 48),
                                                ('sklearn.tree._classes', 'DecisionTreeClassifier'))
    expected_pipeline = WirNode(54, "Pipeline", "Call", CodeReference(24, 18, 26, 51), ('sklearn.pipeline', 'Pipeline'))
    expected_graph.add_edge(expected_column_transformer, expected_pipeline)
    expected_graph.add_edge(expected_decision_tree_classifier, expected_pipeline)
    expected_graph.add_edge(expected_pipeline, expected_fit)

    expected_print_two = WirNode(58, "print", "Call", CodeReference(31, 0, 31, 26), ('builtins', 'print'))
    expected_graph.add_node(expected_print_two)

    return expected_graph
