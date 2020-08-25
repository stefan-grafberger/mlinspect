"""
Tests whether the PipelineExecutor works
"""
from test.utils import get_pandas_read_csv_and_dropna_code, run_random_annotation_testing_analyzer, \
    run_row_index_annotation_testing_analyzer, run_multiple_test_analyzers


def test_pandas_backend_random_annotation_propagation():
    """
    Tests whether the pandas backend works
    """
    code = get_pandas_read_csv_and_dropna_code()
    random_annotation_analyzer_result = run_random_annotation_testing_analyzer(code)
    assert len(random_annotation_analyzer_result) == 2


def test_pandas_backend_row_index_annotation_propagation():
    """
    Tests whether the pandas backend works
    """
    code = get_pandas_read_csv_and_dropna_code()
    lineage_result = run_row_index_annotation_testing_analyzer(code)
    assert len(lineage_result) == 2


def test_pandas_backend_annotation_propagation_multiple_analyzers():
    """
    Tests whether the pandas backend works
    """
    code = get_pandas_read_csv_and_dropna_code()

    analyzer_results, analyzers = run_multiple_test_analyzers(code)

    for analyzer in analyzers:
        result = analyzer_results[analyzer]
        assert len(result) == 2
