"""
Tests whether the healthcare demo works
"""
import os

import matplotlib
from importnb import Notebook

from experiments.benchmark_utils import do_projection_benchmark
from mlinspect.utils import get_project_root

EXPERIMENT_NB_FILE = os.path.join(str(get_project_root()), "experiments", "operator_benchmarks.ipynb")


def test_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    benchmark_result_no_mlinspect, benchmark_result_one_inspection = do_projection_benchmark(data_frame_rows=10)

    assert benchmark_result_no_mlinspect
    assert benchmark_result_one_inspection


def test_experiment_nb():
    """
    Tests whether the experiment notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(EXPERIMENT_NB_FILE)
