"""
Tests whether the DAG extraction works
"""
import ast
import os
from mlinspect.instrumentation.wir_to_dag_transformer import WirToDagTransformer
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

        assert len(cleaned_wir) == 59
