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
        expected_str = WirVertex(8, "get_project_root", "Call", 11, 26)
        expected_graph.add_edge(expected_get_project_root, expected_str, type="input")

        assert networkx.to_dict_of_dicts(cleaned_wir) == {}  # networkx.to_dict_of_dicts(expected_graph)
