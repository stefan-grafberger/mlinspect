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

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_remove_all_nodes_but_calls_and_subscripts():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    with open(FILE_PY) as file:
        test_code = file.read()

        test_ast = ast.parse(test_code)
        extractor = WirExtractor(test_ast)
        extracted_wir = extractor.extract_wir()
        cleaned_wir = WirToDagTransformer().remove_all_nodes_but_calls_and_subscripts(extracted_wir)

        assert len(cleaned_wir) == 15

        expected_graph = networkx.DiGraph()

        expected_print_one = WirVertex(6, "print", "Call", 10, 0)
        expected_graph.add_node(expected_print_one)

        expected_get_project_root = WirVertex(7, "get_project_root", "Call", 11, 30)
        expected_str = WirVertex(8, "str", "Call", 11, 26)
        expected_graph.add_edge(expected_get_project_root, expected_str)

        expected_join = WirVertex(12, "join", "Call", 11, 13)
        expected_graph.add_edge(expected_str, expected_join)

        expected_read_csv = WirVertex(18, "read_csv", "Call", 12, 11)
        expected_graph.add_edge(expected_join, expected_read_csv)

        expected_dropna = WirVertex(20, "dropna", "Call", 14, 7)
        expected_graph.add_edge(expected_read_csv, expected_dropna)

        expected_fit = WirVertex(56, "fit", "Call", 28, 0)
        expected_index_subscript = WirVertex(23, "Index-Subscript", "Subscript", 16, 38)
        expected_graph.add_edge(expected_dropna, expected_fit)
        expected_graph.add_edge(expected_dropna, expected_index_subscript)

        expected_label_binarize = WirVertex(28, "label_binarize", "Call", 16, 9)
        expected_graph.add_edge(expected_index_subscript, expected_label_binarize)
        expected_graph.add_edge(expected_label_binarize, expected_fit)

        expected_one_hot_encoder = WirVertex(33, "OneHotEncoder", "Call", 19, 20)
        expected_standard_scaler = WirVertex(39, "StandardScaler", "Call", 20, 16)
        expected_column_transformer = WirVertex(46, "ColumnTransformer", "Call", 18, 25)
        expected_graph.add_edge(expected_one_hot_encoder, expected_column_transformer)
        expected_graph.add_edge(expected_standard_scaler, expected_column_transformer)

        expected_decision_tree_classifier = WirVertex(51, "DecisionTreeClassifier", "Call", 26, 19)
        expected_pipeline = WirVertex(54, "Pipeline", "Call", 24, 18)
        expected_graph.add_edge(expected_column_transformer, expected_pipeline)
        expected_graph.add_edge(expected_decision_tree_classifier, expected_pipeline)
        expected_graph.add_edge(expected_pipeline, expected_fit)

        expected_print_two = WirVertex(58, "print", "Call", 31, 0)
        expected_graph.add_node(expected_print_two)

        assert networkx.to_dict_of_dicts(cleaned_wir) == networkx.to_dict_of_dicts(expected_graph)
