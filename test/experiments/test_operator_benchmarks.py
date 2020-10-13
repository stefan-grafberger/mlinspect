"""
Tests whether the healthcare demo works
"""
import os

import matplotlib
from importnb import Notebook

from experiments.benchmark_utils import do_op_instrumentation_benchmarks, OperatorBenchmarkType, \
    do_op_inspections_benchmarks, do_adult_easy_benchmarks, do_healthcare_benchmarks, do_adult_normal_benchmarks, \
    do_compas_benchmarks
from mlinspect.utils import get_project_root

EXPERIMENT_NB_FILE = os.path.join(str(get_project_root()), "experiments", "operator_benchmarks.ipynb")


def test_instrumentation_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    for op_type in OperatorBenchmarkType:
        benchmark_results = do_op_instrumentation_benchmarks(100, op_type)

        assert benchmark_results["no mlinspect"]
        assert benchmark_results["no inspection"]
        assert benchmark_results["one inspection"]
        assert benchmark_results["two inspections"]
        assert benchmark_results["three inspections"]


def test_inspection_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    for op_type in OperatorBenchmarkType:
        benchmark_results = do_op_inspections_benchmarks(100, op_type)

        assert benchmark_results["empty inspection"]
        assert benchmark_results["MaterializeFirstRowsInspection(10)"]
        assert benchmark_results["LineageInspection(10)"]
        assert benchmark_results["HistogramInspection(['group_col'])"]


def test_adult_easy_pipeline_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_adult_easy_benchmarks(1)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_adult_normal_pipeline_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_adult_normal_benchmarks(1)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_compas_pipeline_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    #benchmark_results = do_compas_benchmarks(1)
    pass
    #assert benchmark_results["no mlinspect"]
    #assert benchmark_results["no inspection"]
    #assert benchmark_results["one inspection"]
    #assert benchmark_results["two inspections"]
    #assert benchmark_results["three inspections"]


def test_healthcare_pipeline_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_healthcare_benchmarks(1)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_experiment_nb():
    """
    Tests whether the experiment notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(EXPERIMENT_NB_FILE)
