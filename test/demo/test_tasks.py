"""
Tests whether the healthcare demo works
"""
import os

from importnb import Notebook
import matplotlib

from mlinspect.utils import get_project_root


ADULT_EASY_TASK_NB = os.path.join(str(get_project_root()), "demo", "adult_easy", "adult_easy_task.ipynb")
COMPAS_TASK_NB = os.path.join(str(get_project_root()), "demo", "compas", "compas_task.ipynb")
HEALTHCARE_TASK_NB = os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare_task.ipynb")


def test_adult_easy_task_nb():
    """
    Tests whether this task notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(ADULT_EASY_TASK_NB)


def test_compas_task_nb():
    """
    Tests whether this task notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(COMPAS_TASK_NB)


def test_healthcare_task_nb():
    """
    Tests whether this task notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(HEALTHCARE_TASK_NB)
