"""
Tests whether the benchmark utils work
"""
import timeit

from experiments.performance._lineage_benchmark_utils import get_multiple_dfs_creation_str_orig, \
    get_pandas_join_orig, \
    get_pandas_dropna_orig, \
    get_dropna_df_creation_str_orig, \
    get_one_hot_df_creation_str_orig, get_sklearn_one_hot_orig, \
    get_pandas_join_mlinspect_lineage, \
    get_multiple_dfs_creation_str_mlinspect_lineage, \
    get_pandas_dropna_mlinspect_lineage, \
    get_dropna_df_creation_str_mlinspect_lineage, \
    get_one_hot_df_creation_str_mlinspect_lineage, \
    get_sklearn_one_hot_mlinspect_lineage, \
    do_op_lineage_benchmarks

DATA_FRAME_ROWS = 1000
REPEATS = 2


def test_join_orig():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_orig(), setup=get_multiple_dfs_creation_str_orig(DATA_FRAME_ROWS),
                           repeat=REPEATS, number=1)
    print(result)


def test_join_mlinspect_old_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_mlinspect_lineage(),
                           setup=get_multiple_dfs_creation_str_mlinspect_lineage(DATA_FRAME_ROWS, False),
                           repeat=REPEATS, number=1)
    print(result)


def test_join_mlinspect_new_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_mlinspect_lineage(),
                           setup=get_multiple_dfs_creation_str_mlinspect_lineage(DATA_FRAME_ROWS, True),
                           repeat=REPEATS, number=1)
    print(result)


def test_dropna_orig():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_orig(), setup=get_dropna_df_creation_str_orig(DATA_FRAME_ROWS),
                           repeat=REPEATS, number=1)
    print(result)


def test_dropna_mlinspect_old_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_mlinspect_lineage(),
                           setup=get_dropna_df_creation_str_mlinspect_lineage(DATA_FRAME_ROWS, False),
                           repeat=REPEATS, number=1)
    print(result)


def test_dropna_mlinspect_new_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_mlinspect_lineage(),
                           setup=get_dropna_df_creation_str_mlinspect_lineage(DATA_FRAME_ROWS, True),
                           repeat=REPEATS, number=1)
    print(result)


def test_one_hot_orig():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_orig(), setup=get_one_hot_df_creation_str_orig(DATA_FRAME_ROWS),
                           repeat=REPEATS, number=1)
    print(result)


def test_one_hot_mlinspect_old_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_lineage(),
                           setup=get_one_hot_df_creation_str_mlinspect_lineage(DATA_FRAME_ROWS, False),
                           repeat=REPEATS, number=1)
    print(result)


def test_one_hot_mlinspect_new_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_lineage(),
                           setup=get_one_hot_df_creation_str_mlinspect_lineage(DATA_FRAME_ROWS, True),
                           repeat=REPEATS, number=1)
    print(result)


def test_do_op_lineage_benchmarks():
    """ Performance experiments """
    benchmark_results = do_op_lineage_benchmarks(DATA_FRAME_ROWS)
    assert benchmark_results["dropna orig"]
    assert benchmark_results["dropna old"]
    assert benchmark_results["dropna new"]

    assert benchmark_results["join orig"]
    assert benchmark_results["join old"]
    assert benchmark_results["join new"]

    assert benchmark_results["one-hot orig"]
    assert benchmark_results["one-hot old"]
    assert benchmark_results["one-hot new"]
