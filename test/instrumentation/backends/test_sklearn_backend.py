"""
Tests whether the PipelineExecutor works
"""
import os

from test.instrumentation.backends.annotation_testing_analyzer import AnnotationTestingAnalyzer
from mlinspect.utils import get_project_root
from mlinspect.instrumentation.analyzers.materialize_first_rows_analyzer import MaterializeFirstRowsAnalyzer
from mlinspect.pipeline_inspector import PipelineInspector

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_sklearn_backend_annotation_propagation():
    """
    Tests whether the capturing of module information works
    """
    with open(FILE_PY) as file:
        code = file.read()

        inspection_result = PipelineInspector \
            .on_pipeline_from_string(code) \
            .add_analyzer(AnnotationTestingAnalyzer(10)) \
            .execute()

        analyzer_results = inspection_result.analyzer_to_annotations
        assert AnnotationTestingAnalyzer(10) in analyzer_results
        result = analyzer_results[AnnotationTestingAnalyzer(10)]

        assert len(result) == 12


def test_sklearn_backend_annotation_propagation_multiple_analyzers():
    """
    Tests whether the capturing of module information works
    """
    with open(FILE_PY) as file:
        code = file.read()

        analyzers = [AnnotationTestingAnalyzer(2), MaterializeFirstRowsAnalyzer(5), AnnotationTestingAnalyzer(10)]

        inspection_result = PipelineInspector \
            .on_pipeline_from_string(code) \
            .add_analyzers(analyzers) \
            .execute()
        analyzer_results = inspection_result.analyzer_to_annotations

        for analyzer in analyzers:
            assert analyzer in analyzer_results
            result = analyzer_results[analyzer]

            assert len(result) == 12
