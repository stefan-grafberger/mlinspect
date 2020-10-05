"""
Tests whether the healthcare demo works
"""
import os
from inspect import cleandoc
import timeit
from test.test_utils import get_test_df_creation_str, get_test_projection_str
import matplotlib

from mlinspect.utils import get_project_root

EXPERIMENT_NB_FILE = os.path.join(str(get_project_root()), "experiment", "operator_benchmarks.ipynb")


def test_benchmark_mechanism():
    """
    Tests whether the pipeline works with instrumentation
    """
    data_frame_rows = 10000

    df_creation_str = get_test_df_creation_str(data_frame_rows)
    df_projection = get_test_projection_str()
    print(df_creation_str)
    print(df_projection)

    setup = cleandoc("""
    from mlinspect.inspections.materialize_first_rows_inspection import MaterializeFirstRowsInspection
    from mlinspect.instrumentation.pipeline_executor import singleton
    from test.test_utils import get_test_df_creation_str, get_test_projection_str

    test_code_setup = get_test_df_creation_str({})
    inspector_result = singleton.run(None, None, test_code_setup, [], [])
    test_code_benchmark = get_test_projection_str()
    """.format(data_frame_rows))
    benchmark = cleandoc("""
    inspector_result_two = singleton.run(None, None, test_code_benchmark, [MaterializeFirstRowsInspection(1)], [], 
                                         False)
    """)

    benchmark_result_no_mlinspect = timeit.repeat(stmt=df_projection, setup=df_creation_str, repeat=20, number=1)
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=20, number=1)

    assert benchmark_result_no_mlinspect
    assert benchmark_result_one_inspection


def test_experiment_nb():
    """
    Tests whether the experiment notebook works
    """
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    # Notebook.load(EXPERIMENT_NB_FILE)
