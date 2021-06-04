"""
Tests whether MissingEmbeddings works
"""
from inspect import cleandoc

from testfixtures import compare

from demo.feature_overview.missing_embeddings import MissingEmbeddings, MissingEmbeddingsInfo
from example_pipelines.healthcare import custom_monkeypatching
from mlinspect._pipeline_inspector import PipelineInspector


def test_missing_embeddings():
    """
    Tests whether MissingEmbeddings works for joins
    """
    test_code = cleandoc("""
            import pandas as pd
            from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer

            df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
            word_to_vec = MyW2VTransformer(min_count=2, size=2, workers=1)
            encoded_data = word_to_vec.fit_transform(df)
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(MissingEmbeddings(10)) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .execute()
    inspection_results = list(inspector_result.dag_node_to_inspection_results.values())

    missing_embeddings_output = inspection_results[0][MissingEmbeddings(10)]
    expected_result = None
    compare(missing_embeddings_output, expected_result)

    missing_embeddings_output = inspection_results[1][MissingEmbeddings(10)]
    expected_result = MissingEmbeddingsInfo(2, ['cat_b', 'cat_c'])
    compare(missing_embeddings_output, expected_result)
