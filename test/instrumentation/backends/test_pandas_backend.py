"""
Tests whether the PipelineExecutor works
"""
from test.instrumentation.backends.annotation_testing_analyzer import AnnotationTestingAnalyzer
from test.utils import get_pandas_read_csv_and_dropna_code
from mlinspect.pipeline_inspector import PipelineInspector


def test_pandas_backend_annotation_propagation():
    """
    Tests whether the capturing of module information works
    """
    code = get_pandas_read_csv_and_dropna_code()

    inspection_result = PipelineInspector \
        .on_pipeline_from_string(code) \
        .add_analyzer(AnnotationTestingAnalyzer(10)) \
        .execute()

    analyzer_results = inspection_result.analyzer_to_annotations
    assert AnnotationTestingAnalyzer(10) in analyzer_results
    result = analyzer_results[AnnotationTestingAnalyzer(10)]

    assert len(result) == 2
