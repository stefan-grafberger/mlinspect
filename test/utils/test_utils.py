"""
Tests whether the utils work
"""
from pathlib import Path

from mlinspect.utils._utils import get_project_root


def test_get_project_root():
    """
    Tests whether get_project_root works
    """
    assert get_project_root() == Path(__file__).parent.parent.parent
