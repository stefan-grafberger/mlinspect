"""
Tests whether the healthcare demo works
"""
import os

import matplotlib
from importnb import Notebook

from experiments.benchmark_utils import do_projection_benchmarks, do_selection_benchmarks, do_join_benchmarks, \
    do_one_hot_encoder_benchmarks, do_standard_scaler_benchmarks, do_decision_tree_benchmarks
from mlinspect.utils import get_project_root

EXPERIMENT_NB_FILE = os.path.join(str(get_project_root()), "experiments", "operator_benchmarks.ipynb")


def test_projection_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_projection_benchmarks(data_frame_rows=100)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_selection_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_selection_benchmarks(data_frame_rows=100)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_decision_tree_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_decision_tree_benchmarks(data_frame_rows=100)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_join_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_join_benchmarks(data_frame_rows=100)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_one_hot_encoder_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_one_hot_encoder_benchmarks(data_frame_rows=100)

    assert benchmark_results["no mlinspect"]
    assert benchmark_results["no inspection"]
    assert benchmark_results["one inspection"]
    assert benchmark_results["two inspections"]
    assert benchmark_results["three inspections"]


def test_standard_scaler_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_results = do_standard_scaler_benchmarks(data_frame_rows=100)

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
