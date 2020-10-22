"""
Tests whether the healthcare demo works
"""
import os

from importnb import Notebook
import matplotlib

from mlinspect.utils import get_project_root


DEMO_NB_FILE = os.path.join(str(get_project_root()), "demo", "feature_overview", "feature_overview.ipynb")


def test_demo_nb():
    """
    Tests whether the demo notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(DEMO_NB_FILE)
