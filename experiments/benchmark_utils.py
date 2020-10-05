"""
Tests whether the utils work
"""
import timeit
from inspect import cleandoc


def get_test_df_creation_str(data_frame_rows):
    """
    Get a complete code str that creates a DF with random value
    """
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint

        array = randint(0,100,size=({}, 4))
        df = pd.DataFrame(array, columns=['A', 'B', 'C', 'D'])
        """.format(data_frame_rows))
    return test_code


def get_test_projection_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        test = df[['A']]
        """)
    return test_code


def get_benchmark_code(inspections_str):
    """
    Get the benchmark str for timeit
    """
    benchmark = cleandoc("""
    inspector_result_two = singleton.run(None, None, test_code_benchmark, {}, [], 
                                         False)
    """.format(inspections_str))
    return benchmark


def get_df_setup_code(data_frame_rows, benchmark_str):
    """
    Get the setup str for timeit
    """
    setup = cleandoc("""
    from mlinspect.inspections.materialize_first_rows_inspection import MaterializeFirstRowsInspection
    from mlinspect.instrumentation.pipeline_executor import singleton
    from experiments.benchmark_utils import get_test_df_creation_str, get_test_projection_str

    test_code_setup = get_test_df_creation_str({})
    inspector_result = singleton.run(None, None, test_code_setup, [], [])
    test_code_benchmark = {}
    """.format(data_frame_rows, benchmark_str))
    return setup


def do_projection_benchmark(data_frame_rows):
    """
    Do the projection benchmark
    """
    # no mlinspect
    df_creation_str = get_test_df_creation_str(data_frame_rows)
    df_projection = get_test_projection_str()
    benchmark_result_no_mlinspect = timeit.repeat(stmt=df_projection, setup=df_creation_str, repeat=20, number=1)

    # one inspection
    setup = get_df_setup_code(data_frame_rows, "get_test_projection_str()")
    benchmark = get_benchmark_code("[MaterializeFirstRowsInspection(1)]")
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=20, number=1)

    return benchmark_result_no_mlinspect, benchmark_result_one_inspection
