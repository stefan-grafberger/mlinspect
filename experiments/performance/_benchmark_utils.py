"""
Functions to benchmark mlinspect
"""
import timeit
from dataclasses import dataclass
from enum import Enum
from inspect import cleandoc

from example_pipelines import HEALTHCARE_PY, ADULT_SIMPLE_PY, ADULT_COMPLEX_PY, COMPAS_PY


class OperatorBenchmarkType(Enum):
    """
    The different operators we benchmark
    """
    PROJECTION = "projection"
    SELECTION = "selection"
    JOIN = "join"
    ONE_HOT_ENCODER = "one_hot_encoder"
    STANDARD_SCALER = "standard_scaler"
    DECISION_TREE = "decision_tree"


class PipelineBenchmarkType(Enum):
    """
    The different operators we benchmark
    """
    HEALTHCARE = "healthcare (1000 rows)"
    COMPAS = "compas (train: 5050 rows, test: 2166)"
    ADULT_SIMPLE = "adult_simple (22793 rows)"
    ADULT_COMPLEX = "adult_complex (train: 22793 rows, test: 9770 rows)"


@dataclass(frozen=True, eq=True)
class CodeToBenchmark:
    """
    Simple data class to pass the code strings around.
    """
    benchmark_setup: str
    benchmark_exec: str
    benchmark_setup_func_str: str
    benchmark_exec_func_str: str


def do_op_instrumentation_benchmarks(data_frame_rows, operator_type: OperatorBenchmarkType, repeats=1):
    """
    Do the projection benchmarks
    """
    code_to_benchmark = get_code_for_op_benchmark(data_frame_rows, operator_type)
    benchmark_results = exec_benchmarks_empty_inspection(code_to_benchmark, repeats)
    return benchmark_results


def do_op_inspections_benchmarks(data_frame_rows, operator_type: OperatorBenchmarkType, repeats=1):
    """
    Do the projection benchmarks
    """
    code_to_benchmark = get_code_for_op_benchmark(data_frame_rows, operator_type)
    benchmark_results = exec_benchmarks_nonempty_inspection(code_to_benchmark, repeats)
    return benchmark_results


def do_full_pipeline_benchmarks(pipeline: PipelineBenchmarkType, repeats=1):
    """
    Do the projection benchmarks
    """
    code_to_benchmark = get_code_for_pipeline_benchmark(pipeline)
    benchmark_results = exec_pipeline_benchmarks_empty_inspection(code_to_benchmark, repeats)
    return benchmark_results


def get_code_for_op_benchmark(data_frame_rows, operator_type):
    """
    Get the code to benchmark for the operator benchmark type
    """
    if operator_type == OperatorBenchmarkType.PROJECTION:
        benchmark_setup = get_single_df_creation_str(data_frame_rows)
        benchmark_exec = get_test_projection_str()
        benchmark_setup_func_str = "get_single_df_creation_str({})".format(data_frame_rows)
        benchmark_exec_func_str = "get_test_projection_str()"
    elif operator_type == OperatorBenchmarkType.SELECTION:
        benchmark_setup = get_single_df_creation_str(data_frame_rows)
        benchmark_exec = get_test_selection_str()
        benchmark_setup_func_str = "get_single_df_creation_str({})".format(data_frame_rows)
        benchmark_exec_func_str = "get_test_selection_str()"
    elif operator_type == OperatorBenchmarkType.JOIN:
        benchmark_setup = get_multiple_dfs_creation_str(data_frame_rows)
        benchmark_exec = get_test_join_str()
        benchmark_setup_func_str = "get_multiple_dfs_creation_str({})".format(data_frame_rows)
        benchmark_exec_func_str = "get_test_join_str()"
    elif operator_type == OperatorBenchmarkType.ONE_HOT_ENCODER:
        benchmark_setup = get_np_cat_array_str(data_frame_rows)
        benchmark_exec = get_test_one_hot_encoder_str()
        benchmark_setup_func_str = "get_np_cat_array_str({})".format(data_frame_rows)
        benchmark_exec_func_str = "get_test_one_hot_encoder_str()"
    elif operator_type == OperatorBenchmarkType.STANDARD_SCALER:
        benchmark_setup = get_np_num_array_str(data_frame_rows)
        benchmark_exec = get_test_standard_scaler_str()
        benchmark_setup_func_str = "get_np_num_array_str({})".format(data_frame_rows)
        benchmark_exec_func_str = "get_test_standard_scaler_str()"
    elif operator_type == OperatorBenchmarkType.DECISION_TREE:
        benchmark_setup = get_estimator_train_data_str(data_frame_rows)
        benchmark_exec = get_decision_tree_str()
        benchmark_setup_func_str = "get_estimator_train_data_str({})".format(data_frame_rows)
        benchmark_exec_func_str = "get_decision_tree_str()"
    else:
        assert False
    code_to_benchmark = CodeToBenchmark(benchmark_setup, benchmark_exec, benchmark_setup_func_str,
                                        benchmark_exec_func_str)
    return code_to_benchmark


def get_code_for_pipeline_benchmark(pipeline_type):
    """
    Get the code to benchmark for the operator benchmark type
    """
    if pipeline_type == PipelineBenchmarkType.HEALTHCARE:
        benchmark_exec = get_healthcare_py_str()
        benchmark_setup_func_str = "get_healthcare_py_str()"
    elif pipeline_type == PipelineBenchmarkType.COMPAS:
        benchmark_exec = get_compas_py_str()
        benchmark_setup_func_str = "get_compas_py_str()"
    elif pipeline_type == PipelineBenchmarkType.ADULT_SIMPLE:
        benchmark_exec = get_adult_simple_py_str()
        benchmark_setup_func_str = "get_adult_simple_py_str()"
    elif pipeline_type == PipelineBenchmarkType.ADULT_COMPLEX:
        benchmark_exec = get_adult_complex_py_str()
        benchmark_setup_func_str = "get_adult_complex_py_str()"
    else:
        assert False
    code_to_benchmark = CodeToBenchmark("pass", benchmark_exec, benchmark_setup_func_str, "")
    return code_to_benchmark


def exec_benchmarks_empty_inspection(code_to_benchmark, repeats):
    """
    Benchmark some code without mlinspect and with mlinspect with varying numbers of inspections
    """
    benchmark_results = {
        "no mlinspect": timeit.repeat(stmt=code_to_benchmark.benchmark_exec, setup=code_to_benchmark.benchmark_setup,
                                      repeat=repeats, number=1),
        "no inspection": benchmark_code_str_with_inspections(code_to_benchmark.benchmark_exec_func_str,
                                                             code_to_benchmark.benchmark_setup_func_str, "[]",
                                                             repeats),
        "one inspection": benchmark_code_str_with_inspections(code_to_benchmark.benchmark_exec_func_str,
                                                              code_to_benchmark.benchmark_setup_func_str,
                                                              "[EmptyInspection(0)]", repeats),
        "two inspections": benchmark_code_str_with_inspections(code_to_benchmark.benchmark_exec_func_str,
                                                               code_to_benchmark.benchmark_setup_func_str,
                                                               "[EmptyInspection(0), EmptyInspection(1)]", repeats),
        "three inspections": benchmark_code_str_with_inspections(code_to_benchmark.benchmark_exec_func_str,
                                                                 code_to_benchmark.benchmark_setup_func_str,
                                                                 "[EmptyInspection(0), " +
                                                                 "EmptyInspection(1), EmptyInspection(2)]", repeats)}

    return benchmark_results


def exec_benchmarks_nonempty_inspection(code_to_benchmark, repeats):
    """
    Benchmark some code without mlinspect and with mlinspect with varying numbers of inspections
    """
    benchmark_results = {
        "empty inspection": benchmark_code_str_with_inspections(code_to_benchmark.benchmark_exec_func_str,
                                                                code_to_benchmark.benchmark_setup_func_str,
                                                                "[EmptyInspection(0)]", repeats),
        "MaterializeFirstOutputRows(10)": benchmark_code_str_with_inspections(
            code_to_benchmark.benchmark_exec_func_str,
            code_to_benchmark.benchmark_setup_func_str,
            "[MaterializeFirstOutputRows(10)]", repeats),
        "RowLineage(10)": benchmark_code_str_with_inspections(code_to_benchmark.benchmark_exec_func_str,
                                                              code_to_benchmark.benchmark_setup_func_str,
                                                              "[RowLineage(10)]", repeats),
        "HistogramForColumns(['group_col_1'])": benchmark_code_str_with_inspections(
            code_to_benchmark.benchmark_exec_func_str,
            code_to_benchmark.benchmark_setup_func_str,
            "[HistogramForColumns(['group_col'])]", repeats),
        "HistogramForColumns(['group_col_1', 'group_col_2', 'group_col_3'])": benchmark_code_str_with_inspections(
            code_to_benchmark.benchmark_exec_func_str,
            code_to_benchmark.benchmark_setup_func_str,
            "[HistogramForColumns(['group_col_1', 'group_col_2', 'group_col_3'])]", repeats)
    }

    return benchmark_results


def exec_pipeline_benchmarks_empty_inspection(code_to_benchmark, repeats):
    """
    Benchmark some code without mlinspect and with mlinspect with varying numbers of inspections
    """
    benchmark_results = {
        "no mlinspect": timeit.repeat(stmt=code_to_benchmark.benchmark_exec, setup=code_to_benchmark.benchmark_setup,
                                      repeat=repeats, number=1),
        "no inspection": benchmark_pipeline_code_str_with_inspections(code_to_benchmark.benchmark_setup_func_str, "[]",
                                                                      repeats),
        "one inspection": benchmark_pipeline_code_str_with_inspections(code_to_benchmark.benchmark_setup_func_str,
                                                                       "[EmptyInspection(0)]", repeats),
        "two inspections": benchmark_pipeline_code_str_with_inspections(code_to_benchmark.benchmark_setup_func_str,
                                                                        "[EmptyInspection(0), EmptyInspection(1)]",
                                                                        repeats),
        "three inspections": benchmark_pipeline_code_str_with_inspections(code_to_benchmark.benchmark_setup_func_str,
                                                                          "[EmptyInspection(0), " +
                                                                          "EmptyInspection(1), EmptyInspection(2)]",
                                                                          repeats)}

    return benchmark_results


def benchmark_code_str_with_inspections(benchmark_str, setup_str, inspections_str, repeats):
    """
    Execute one single benchmark
    """
    setup = prepare_benchmark_exec(benchmark_str, setup_str, inspections_str)
    benchmark = trigger_benchmark_exec(inspections_str)
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=repeats, number=1)
    return benchmark_result_one_inspection


def benchmark_pipeline_code_str_with_inspections(setup_str, inspections_str, repeats):
    """
    Execute one single benchmark
    """
    setup = prepare_pipeline_benchmark_exec(setup_str)
    benchmark = trigger_pipeline_benchmark_exec(inspections_str)
    benchmark_result_one_inspection = timeit.repeat(stmt=benchmark, setup=setup, repeat=repeats, number=1)
    return benchmark_result_one_inspection


def prepare_benchmark_exec(benchmark_str, setup_str, inspections):
    """
    Get the setup str for timeit
    """
    setup = cleandoc("""
    from experiments.performance._empty_inspection import EmptyInspection
    from mlinspect.instrumentation._pipeline_executor import singleton
    from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows
    from experiments.performance._benchmark_utils import get_single_df_creation_str, get_multiple_dfs_creation_str, \
        get_test_projection_str, get_test_selection_str, get_test_join_str, get_np_cat_array_str, \
        get_test_one_hot_encoder_str, get_np_num_array_str, get_test_standard_scaler_str, \
        get_estimator_train_data_str, get_decision_tree_str

    test_code_setup = {}
    inspector_result = singleton.run(python_code=test_code_setup, inspections={})
    test_code_benchmark = {}
    """.format(setup_str, inspections, benchmark_str))
    return setup


def trigger_benchmark_exec(inspections_str):
    """
    Get the benchmark str for timeit
    """
    benchmark = cleandoc("""
    inspector_result_two = singleton.run(python_code=test_code_benchmark, inspections={}, reset_state=False)
    """.format(inspections_str))
    return benchmark


def prepare_pipeline_benchmark_exec(test_code):
    """
    Get the benchmark str for timeit
    """
    benchmark = cleandoc("""
    from experiments.performance._benchmark_utils import get_adult_simple_py_str, get_adult_complex_py_str, \
        get_healthcare_py_str, get_compas_py_str
    
    code = {}
    """.format(test_code))
    return benchmark


def trigger_pipeline_benchmark_exec(inspections_str):
    """
    Get the benchmark str for timeit
    """
    benchmark = cleandoc("""
    from experiments.performance._empty_inspection import EmptyInspection
    from mlinspect import PipelineInspector
    
    PipelineInspector\
            .on_pipeline_from_string(code)\
            .add_required_inspections({}) \
            .execute()
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
        import random

        a = randint(0,100,size=({data_frame_rows}))
        b = randint(0,100,size=({data_frame_rows}))
        c = randint(0,100,size=({data_frame_rows}))
        d = randint(0,100,size=({data_frame_rows}))
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_2 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_3 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(a, b, c, d, group_col_1, group_col_2, group_col_3), columns=['A', 'B', 'C', 'D', 
            'group_col_1', 'group_col_2', 'group_col_3'])
        """.format(data_frame_rows=data_frame_rows))
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
        import random

        id_a = np.arange({sizes_before_join})
        shuffle(id_a)
        a = randint(0,100,size=({sizes_before_join}))
        b = randint(0,100,size=({sizes_before_join}))
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={sizes_before_join}))
        group_col_2 = pd.Series(random.choices(categories, k={sizes_before_join}))
        group_col_3 = pd.Series(random.choices(categories, k={sizes_before_join}))
        
        id_b = np.arange({start_with_offset}, {end_with_offset})
        shuffle(id_b)
        c = randint(0,100,size=({sizes_before_join})) 
        d = randint(0,100,size=({sizes_before_join}))
        
        df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3), columns=['id', 'A', 'B', 
            'group_col_1', 'group_col_2', 'group_col_3'])
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


def get_np_cat_array_str(data_frame_rows):
    """
    Get a complete code str that creates a np array with random values
    """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random

        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(group_col_1), columns=["group_col_1"])
        """.format(data_frame_rows=data_frame_rows))
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


def get_np_num_array_str(data_frame_rows):
    """
    Get a complete code str that creates a np array with random values
    """
    test_code = cleandoc("""
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        from numpy.random import randint

        series = randint(0,100,size=({})) 
        df = pd.DataFrame(series, columns=["num"])
        """.format(data_frame_rows))
    return test_code


def get_test_standard_scaler_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        standard_scaler = StandardScaler()
        encoded_data = standard_scaler.fit_transform(df)
        """)
    return test_code


def get_estimator_train_data_str(data_frame_rows):
    """
    Get a complete code str that creates a np array with random values
    """
    test_code = cleandoc("""
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        from numpy.random import randint
        from sklearn import tree, datasets
        
        data, target = datasets.load_digits(return_X_y=True, as_frame=True)
        
        data = data.sample(n={data_frame_rows}, replace=True, random_state=2)
        target = target.sample(n={data_frame_rows}, replace=True, random_state=2)

        data_df = pd.DataFrame(data)
        target_df = pd.DataFrame(target)
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_decision_tree_str():
    """
    Get a pandas projection code str
    """
    test_code = cleandoc("""
        classifier = tree.DecisionTreeClassifier()
        encoded_data = classifier.fit(data_df, target_df)
        """)
    return test_code


def get_adult_simple_py_str():
    """
    Get the code str for the adult_easy pipeline
    """
    with open(ADULT_SIMPLE_PY) as file:
        test_code = file.read()
    return test_code


def get_adult_complex_py_str():
    """
    Get the code str for the adult_easy pipeline
    """
    with open(ADULT_COMPLEX_PY) as file:
        test_code = file.read()
    return test_code


def get_compas_py_str():
    """
    Get the code str for the adult_easy pipeline
    """
    with open(COMPAS_PY) as file:
        test_code = file.read()
    return test_code


def get_healthcare_py_str():
    """
    Get the code str for the adult_easy pipeline
    """
    with open(HEALTHCARE_PY) as file:
        test_code = file.read()
    return test_code
