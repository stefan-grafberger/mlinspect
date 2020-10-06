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


def get_test_selection_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        test = df[df['A'] > 50]
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


def get_df_setup_code(data_frame_rows, benchmark_str, inspections):
    """
    Get the setup str for timeit
    """
    setup = cleandoc("""
    from experiments.empty_inspection import EmptyInspection
    from mlinspect.instrumentation.pipeline_executor import singleton
    from experiments.benchmark_utils import get_test_df_creation_str, get_test_projection_str, get_test_selection_str

    test_code_setup = get_test_df_creation_str({})
    inspector_result = singleton.run(None, None, test_code_setup, {}, [])
    test_code_benchmark = {}
    """.format(data_frame_rows, inspections, benchmark_str))
    return setup


def do_projection_benchmarks(data_frame_rows, repeats=5):
    """
    Do the projection benchmarks
    """
    # no mlinspect
    benchmark_results = {}

    df_creation_str = get_test_df_creation_str(data_frame_rows)
    df_projection = get_test_projection_str()
    benchmark_result = timeit.repeat(stmt=df_projection, setup=df_creation_str, repeat=repeats, number=1)
    benchmark_results["no mlinspect"] = benchmark_result

    # no inspection
    benchmark_result = benchmark_projection_with_inspections(data_frame_rows, "[]", repeats)
    benchmark_results["no inspection"] = benchmark_result

    # one inspection
    benchmark_result = benchmark_projection_with_inspections(data_frame_rows, "[EmptyInspection(0)]", repeats)
    benchmark_results["one inspection"] = benchmark_result

    # two inspections
    benchmark_result = benchmark_projection_with_inspections(data_frame_rows,
                                                             "[EmptyInspection(0), EmptyInspection(1)]", repeats)
    benchmark_results["two inspections"] = benchmark_result

    # three inspections
    benchmark_result = benchmark_projection_with_inspections(data_frame_rows, "[EmptyInspection(0), " +
                                                             "EmptyInspection(1), EmptyInspection(2)]", repeats)
    benchmark_results["three inspections"] = benchmark_result

    return benchmark_results


def do_selection_benchmarks(data_frame_rows, repeats=5):
    """
    Do the projection benchmarks
    """
    # no mlinspect
    benchmark_results = {}

    df_creation_str = get_test_df_creation_str(data_frame_rows)
    df_selection = get_test_selection_str()
    benchmark_result = timeit.repeat(stmt=df_selection, setup=df_creation_str, repeat=repeats, number=1)
    benchmark_results["no mlinspect"] = benchmark_result

    # no inspection
    benchmark_result = benchmark_selection_with_inspections(data_frame_rows, "[]", repeats)
    benchmark_results["no inspection"] = benchmark_result

    # one inspection
    benchmark_result = benchmark_selection_with_inspections(data_frame_rows, "[EmptyInspection(0)]", repeats)
    benchmark_results["one inspection"] = benchmark_result

    # two inspections
    benchmark_result = benchmark_selection_with_inspections(data_frame_rows,
                                                            "[EmptyInspection(0), EmptyInspection(1)]", repeats)
    benchmark_results["two inspections"] = benchmark_result

    # three inspections
    benchmark_result = benchmark_selection_with_inspections(data_frame_rows, "[EmptyInspection(0), " +
                                                            "EmptyInspection(1), EmptyInspection(2)]", repeats)
    benchmark_results["three inspections"] = benchmark_result

    return benchmark_results


def benchmark_projection_with_inspections(data_frame_rows, inspections, repeats):
    """
    Execute one single projection benchmark
    """
    setup = get_df_setup_code(data_frame_rows, "get_test_projection_str()", inspections)
    benchmark = get_benchmark_code(inspections)
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=repeats, number=1)
    return benchmark_result_one_inspection


def benchmark_selection_with_inspections(data_frame_rows, inspections, repeats):
    """
    Execute one single projection benchmark
    """
    setup = get_df_setup_code(data_frame_rows, "get_test_selection_str()", inspections)
    benchmark = get_benchmark_code(inspections)
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=repeats, number=1)
    return benchmark_result_one_inspection
