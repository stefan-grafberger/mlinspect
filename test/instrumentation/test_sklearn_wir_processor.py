"""
Tests whether the Sklearn DAG extraction works
"""
import os

import networkx
from testfixtures import compare

from mlinspect.backends._sklearn_wir_processor import SklearnWirPreprocessor
from mlinspect.instrumentation._wir_to_dag_transformer import WirToDagTransformer
from mlinspect.utils._utils import get_project_root
from ..testing_helper_utils import get_test_wir, get_expected_dag_adult_easy_py_without_columns

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_sklearn_wir_preprocessing():
    """
    Tests whether the WIR Extraction works for the adult_easy pipeline
    """
    preprocessed_wir = SklearnWirPreprocessor().process_wir(get_test_wir())
    cleaned_wir = WirToDagTransformer.remove_all_nodes_but_calls_and_subscripts(preprocessed_wir)
    dag = WirToDagTransformer.remove_all_non_operators_and_update_names(cleaned_wir)

    assert len(dag) == 17

    expected_dag = get_expected_dag_adult_easy_py_without_columns()

    compare(networkx.to_dict_of_dicts(preprocessed_wir), networkx.to_dict_of_dicts(expected_dag))
