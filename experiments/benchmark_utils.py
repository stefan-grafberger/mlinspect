"""
Functions to benchmark mlinspect
"""
import timeit
from inspect import cleandoc


def do_one_hot_encoder_benchmarks(data_frame_rows, repeats=5):
    """
    Do the projection benchmarks
    """
    benchmark_setup = get_np_array_str(data_frame_rows)
    benchmark_exec = get_test_one_hot_encoder_str()
    benchmark_setup_func_str = "get_np_array_str({})".format(data_frame_rows)
    benchmark_exec_func_str = "get_test_one_hot_encoder_str()"

    benchmark_results = exec_benchmarks(benchmark_exec, benchmark_exec_func_str, benchmark_setup,
                                        benchmark_setup_func_str, repeats)

    return benchmark_results


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


def do_join_benchmarks(data_frame_rows, repeats=5):
    """
    Do the selection benchmarks
    """
    benchmark_setup = get_multiple_dfs_creation_str(data_frame_rows)
    benchmark_exec = get_test_join_str()
    benchmark_setup_func_str = "get_multiple_dfs_creation_str({})".format(data_frame_rows)
    benchmark_exec_func_str = "get_test_join_str()"

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
    from experiments.benchmark_utils import get_single_df_creation_str, get_multiple_dfs_creation_str, \
        get_test_projection_str, get_test_selection_str, get_test_join_str, get_np_array_str, \
        get_test_one_hot_encoder_str

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


def get_multiple_dfs_creation_str(data_frame_rows):
    """
    Get a complete code str that creates a DF with random value
    """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

    # mlinspect does not support some ast nodes yet like *, /, and {}, so we need to avoid them
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint, shuffle

        id_a = np.arange({sizes_before_join})
        shuffle(id_a)
        a = randint(0,100,size=({sizes_before_join}))
        b = randint(0,100,size=({sizes_before_join}))
        
        id_b = np.arange({start_with_offset}, {end_with_offset})
        shuffle(id_b)
        c = randint(0,100,size=({sizes_before_join})) 
        d = randint(0,100,size=({sizes_before_join}))
        
        df_a = pd.DataFrame(zip(id_a, a, b), columns=['id', 'A', 'B'])
        df_b = pd.DataFrame(zip(id_b, c, d), columns=['id', 'C', 'D'])
        """.format(sizes_before_join=sizes_before_join, start_with_offset=start_with_offset,
                   end_with_offset=end_with_offset))
    return test_code


def get_test_join_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        test = df_a.merge(df_b, on='id')
        """)
    return test_code


def get_np_array_str(data_frame_rows):
    """
    Get a complete code str that creates a np array with random values
    """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random

        categories = ['cat_a', 'cat_b', 'cat_c']
        series = pd.Series(random.choices(categories, k={}))
        df = pd.DataFrame(series, columns=["cat"])
        """.format(data_frame_rows))
    return test_code


def get_test_one_hot_encoder_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        one_hot_encoder = OneHotEncoder()
        encoded_data = one_hot_encoder.fit_transform(df)
        """)
    return test_code
