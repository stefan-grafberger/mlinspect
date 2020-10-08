"""
Functions to benchmark mlinspect
"""
import timeit
from inspect import cleandoc


def do_projection_benchmarks(data_frame_rows, repeats=5):
    """
    Do the projection benchmarks
    """
    benchmark_setup = get_single_df_creation_str(data_frame_rows)
    benchmark_exec = get_test_projection_str()
    benchmark_setup_func_str = "get_single_df_creation_str({})".format(data_frame_rows)
    benchmark_exec_func_str = "get_test_projection_str()"

    benchmark_results = exec_benchmarks(benchmark_exec, benchmark_exec_func_str, benchmark_setup,
                                        benchmark_setup_func_str, repeats)

    return benchmark_results


def do_selection_benchmarks(data_frame_rows, repeats=5):
    """
    Do the selection benchmarks
    """
    benchmark_setup = get_single_df_creation_str(data_frame_rows)
    benchmark_exec = get_test_selection_str()
    benchmark_setup_func_str = "get_single_df_creation_str({})".format(data_frame_rows)
    benchmark_exec_func_str = "get_test_selection_str()"

    benchmark_results = exec_benchmarks(benchmark_exec, benchmark_exec_func_str, benchmark_setup,
                                        benchmark_setup_func_str, repeats)

    return benchmark_results


def exec_benchmarks(benchmark_exec, benchmark_exec_func_str, benchmark_setup, benchmark_setup_func_str, repeats):
    """
    Benchmark some code without mlinspect and with mlinspect with varying numbers of inspections
    """
    benchmark_results = {
        "no mlinspect": timeit.repeat(stmt=benchmark_exec, setup=benchmark_setup, repeat=repeats, number=1),
        "no inspection": benchmark_code_str_with_inspections(benchmark_exec_func_str, benchmark_setup_func_str, "[]",
                                                             repeats),
        "one inspection": benchmark_code_str_with_inspections(benchmark_exec_func_str, benchmark_setup_func_str,
                                                              "[EmptyInspection(0)]", repeats),
        "two inspections": benchmark_code_str_with_inspections(benchmark_exec_func_str, benchmark_setup_func_str,
                                                               "[EmptyInspection(0), EmptyInspection(1)]", repeats),
        "three inspections": benchmark_code_str_with_inspections(benchmark_exec_func_str, benchmark_setup_func_str,
                                                                 "[EmptyInspection(0), " +
                                                                 "EmptyInspection(1), EmptyInspection(2)]", repeats)}

    return benchmark_results


def benchmark_code_str_with_inspections(benchmark_str, setup_str, inspections_str, repeats):
    """
    Execute one single benchmark
    """
    setup = prepare_benchmark_exec(benchmark_str, setup_str, inspections_str)
    benchmark = trigger_benchmark_exec(inspections_str)
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=repeats, number=1)
    return benchmark_result_one_inspection


def prepare_benchmark_exec(benchmark_str, setup_str, inspections):
    """
    Get the setup str for timeit
    """
    setup = cleandoc("""
    from experiments.empty_inspection import EmptyInspection
    from mlinspect.instrumentation.pipeline_executor import singleton
    from experiments.benchmark_utils import get_single_df_creation_str, get_test_projection_str, get_test_selection_str

    test_code_setup = {}
    inspector_result = singleton.run(None, None, test_code_setup, {}, [])
    test_code_benchmark = {}
    """.format(setup_str, inspections, benchmark_str))
    return setup


def trigger_benchmark_exec(inspections_str):
    """
    Get the benchmark str for timeit
    """
    benchmark = cleandoc("""
    inspector_result_two = singleton.run(None, None, test_code_benchmark, {}, [], 
                                         False)
    """.format(inspections_str))
    return benchmark


def get_single_df_creation_str(data_frame_rows):
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
