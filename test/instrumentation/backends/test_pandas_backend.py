"""
Tests whether the PipelineExecutor works
"""
from inspect import cleandoc

from test.instrumentation.backends.annotation_testing_analyzer import AnnotationTestingAnalyzer
from mlinspect.pipeline_inspector import PipelineInspector


def test_pandas_backend_annotation_propagation():
    """
    Tests whether the capturing of module information works
    """
    code = cleandoc("""
            import os
            import pandas as pd
            from mlinspect.utils import get_project_root
            
            train_file = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")
            raw_data = pd.read_csv(train_file)
            data = raw_data.dropna()
            """)

    inspection_result = PipelineInspector \
        .on_pipeline_from_string(code) \
        .add_analyzer(AnnotationTestingAnalyzer(10)) \
        .execute()

    analyzer_results = inspection_result.analyzer_to_annotations
    assert AnnotationTestingAnalyzer(10) in analyzer_results
    result = analyzer_results[AnnotationTestingAnalyzer(10)]

    assert len(result) == 2
