"""
Some useful utils for the project
"""
import os

from mlinspect.utils import get_project_root

ADULT_EASY_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_easy", "adult_easy.py")
ADULT_EASY_IPYNB = os.path.join(str(get_project_root()), "example_pipelines", "adult_easy", "adult_easy.ipynb")

ADULT_NORMAL_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_normal", "adult_normal.py")

COMPAS_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")

HEALTHCARE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")
