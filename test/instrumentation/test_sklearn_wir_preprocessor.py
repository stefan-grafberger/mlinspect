"""
Tests whether the Sklearn DAG extraction works
"""
import ast
import os

from mlinspect.instrumentation.wir_extractor import WirExtractor
from mlinspect.instrumentation.sklearn_wir_preprocessor import SklearnWirPreprocessor
from mlinspect.instrumentation.wir_to_dag_transformer import WirToDagTransformer
from mlinspect.utils import get_project_root
from ..utils import get_module_info

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


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

        preprocessed_wir = SklearnWirPreprocessor.sklearn_wir_preprocessing(extracted_wir_with_module_info)
        cleaned_wir = WirToDagTransformer.remove_all_nodes_but_calls_and_subscripts(preprocessed_wir)
        dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

        assert len(dag) == 13

        # expected_dag = get_expected_dag_adult_easy_py()

        # assert networkx.to_dict_of_dicts(preprocessed_wir) == {} #networkx.to_dict_of_dicts(expected_dag)
