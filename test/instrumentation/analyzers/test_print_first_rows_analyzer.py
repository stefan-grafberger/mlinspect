"""
Tests whether the DagVertex works
"""
import os

from mlinspect.instrumentation.analyzer_input import AnalyzerInputRow
from mlinspect.instrumentation.analyzers.print_first_rows_analyzer import PrintFirstRowsAnalyzer
from mlinspect.instrumentation.dag_vertex import DagVertex
from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.utils import get_project_root

FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")


def test_print_first_rows_analyzer():
    """
    Tests whether the DagVertex works
    """
    inspection_result = PipelineInspector \
        .on_pipeline_from_py_file(FILE_PY) \
        .add_analyzer(PrintFirstRowsAnalyzer(2)) \
        .execute()
    analyzer_results = inspection_result.analyzer_to_annotations
    assert PrintFirstRowsAnalyzer(2) in analyzer_results
    result = analyzer_results[PrintFirstRowsAnalyzer(2)]

    assert result == get_expected_result()


def get_expected_result():
    """
    Get the expected PrintFirstRowsAnalyzer(2) result for the adult_easy example
    """
    expected_result = {
        DagVertex(node_id=18, operator_name='Data Source', module=('pandas.io.parsers', 'read_csv'), lineno=12,
                  col_offset=11, description='adult_train.csv'): [
                      AnalyzerInputRow(
                          values=[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                                  'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                          fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                  'hours-per-week', 'native-country', 'income-per-year']),
                      AnalyzerInputRow(
                          values=[29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                                  'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K'],
                          fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                  'marital-status', 'occupation', 'relationship', 'race',
                                  'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                                  'native-country', 'income-per-year'])],
        DagVertex(node_id=20, operator_name='Selection', module=('pandas.core.frame', 'dropna'), lineno=14,
                  col_offset=7, description='dropna'): [
                      AnalyzerInputRow(
                          values=[46, 'Private', 128645, 'Some-college', 10, 'Divorced', 'Prof-specialty',
                                  'Not-in-family', 'White', 'Female', 0, 0, 40, 'United-States', '<=50K'],
                          fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                  'capital-gain', 'capital-loss', 'hours-per-week',
                                  'native-country', 'income-per-year']),
                      AnalyzerInputRow(
                          values=[29, 'Local-gov', 115585, 'Some-college', 10, 'Never-married', 'Handlers-cleaners',
                                  'Not-in-family', 'White', 'Male', 0, 0, 50, 'United-States', '<=50K'],
                          fields=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                  'capital-gain', 'capital-loss', 'hours-per-week',
                                  'native-country', 'income-per-year'])]}
    return expected_result
