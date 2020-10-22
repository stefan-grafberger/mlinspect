"""
Tests whether the performance benchmark notebook works
"""
import os

import matplotlib
from importnb import Notebook

from mlinspect.utils import get_project_root

EXPERIMENT_NB_FILE = os.path.join(str(get_project_root()), "experiments", "performance", "performance_benchmarks.ipynb")


def test_experiment_nb():
    """
    Tests whether the experiment notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(EXPERIMENT_NB_FILE)
