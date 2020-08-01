"""
Tests whether the Sklearn DAG extraction works
"""
import os

import networkx

from mlinspect.instrumentation.wir_extractor import WirExtractor
from mlinspect.instrumentation.sklearn_wir_preprocessor import SklearnWirPreprocessor
from mlinspect.instrumentation.wir_to_dag_transformer import WirToDagTransformer
from mlinspect.utils import get_project_root
from ..utils import get_module_info, get_adult_easy_py_ast, get_expected_dag_adult_easy_py, get_call_description_info

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_sklearn_wir_preprocessing():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    test_ast = get_adult_easy_py_ast()
    extractor = WirExtractor(test_ast)
    extractor.extract_wir()
    wir = extractor.add_runtime_info(get_module_info(), get_call_description_info())

    preprocessed_wir = SklearnWirPreprocessor().sklearn_wir_preprocessing(wir)
    cleaned_wir = WirToDagTransformer.remove_all_nodes_but_calls_and_subscripts(preprocessed_wir)
    dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

    assert len(dag) == 17

    expected_dag = get_expected_dag_adult_easy_py()

    assert networkx.to_dict_of_dicts(preprocessed_wir) == networkx.to_dict_of_dicts(expected_dag)
