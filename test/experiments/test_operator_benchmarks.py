"""
Tests whether the healthcare demo works
"""
import os
from inspect import cleandoc

import matplotlib

from mlinspect.inspections.materialize_first_rows_inspection import MaterializeFirstRowsInspection
from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.utils import get_project_root

EXPERIMENT_NB_FILE = os.path.join(str(get_project_root()), "experiment", "operator_benchmarks.ipynb")


def test_instrumented_y_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    test_code = cleandoc("""
    import pandas as pd
    import numpy as np
    from numpy.random import randint

    array = randint(0,100,size=({}, 4))
    df = pd.DataFrame(array, columns=['A', 'B', 'C', 'D'])
    test = df[['A']]
    """.format(100))

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_required_inspection(MaterializeFirstRowsInspection(1))\
        .execute()

    assert inspector_result


def test_experiment_nb():
    """
    Tests whether the experiment notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    # Notebook.load(EXPERIMENT_NB_FILE)
