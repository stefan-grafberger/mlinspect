"""
Tests whether the healthcare demo works
"""
import os

from importnb import Notebook
import matplotlib

from mlinspect.utils import get_project_root


ADULT_SIMPLE_TASK_NB = os.path.join(str(get_project_root()), "experiments", "user_interviews",
                                    "example-task-with-solution.ipynb")
COMPAS_TASK_NB = os.path.join(str(get_project_root()), "experiments", "user_interviews", "task-1-solution.ipynb")
HEALTHCARE_TASK_NB = os.path.join(str(get_project_root()), "experiments", "user_interviews", "task-2-solution.ipynb")


def test_adult_simple_task_nb():
    """
    Tests whether this task notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    Notebook.load(ADULT_SIMPLE_TASK_NB)


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
