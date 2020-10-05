"""
Tests whether the healthcare demo works
"""
import ast
import os

from importnb import Notebook
import matplotlib

from demo.healthcare.missing_embeddings_inspection import MissingEmbeddingInspection
from mlinspect.checks.check import Check
from mlinspect.inspections.lineage_inspection import LineageInspection
from mlinspect.inspections.materialize_first_rows_inspection import MaterializeFirstRowsInspection
from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.utils import get_project_root

ADULT_EASY_FILE_PY = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.py")
FILE_NB = os.path.join(str(get_project_root()), "test", "pipelines", "adult_easy.ipynb")


PIPELINE_FILE_PY = os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare.py")
DEMO_NB_FILE = os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare_demo.ipynb")


def test_py_pipeline_runs():
    """
    Tests whether the pipeline works without instrumentation
    """
    with open(PIPELINE_FILE_PY) as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    check = Check()\
        .no_bias_introduced_for(["age_group", "race"])\
        .no_illegal_features()

    PipelineInspector\
        .on_pipeline_from_py_file(PIPELINE_FILE_PY) \
        .add_check(check) \
        .add_required_inspection(MissingEmbeddingInspection(20)) \
        .add_required_inspection(LineageInspection(5)) \
        .add_required_inspection(MaterializeFirstRowsInspection(5)) \
        .execute()


def test_demo_nb():
    """
    Tests whether the demo notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(DEMO_NB_FILE)
