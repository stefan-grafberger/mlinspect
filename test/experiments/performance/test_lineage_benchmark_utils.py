"""
Tests whether the benchmark utils work
"""
import timeit

from experiments.performance._lineage_benchmark_utils import get_multiple_dfs_creation_str_orig, \
    get_pandas_join_orig, get_pandas_join_mlinspect_old_histogram, \
    get_multiple_dfs_creation_str_mlinspect_old_histogram, \
    get_pandas_join_mlinspect_new_histogram, get_multiple_dfs_creation_str_mlinspect_new_histogram, \
    get_pandas_dropna_orig, \
    get_dropna_df_creation_str_orig, get_pandas_dropna_mlinspect_old_histogram, \
    get_dropna_df_creation_str_mlinspect_old_histogram, \
    get_pandas_dropna_mlinspect_new_histogram, get_dropna_df_creation_str_mlinspect_new_histogram, \
    get_one_hot_df_creation_str_orig, get_sklearn_one_hot_orig, get_sklearn_one_hot_mlinspect_old_histogram, \
    get_one_hot_df_creation_str_mlinspect_old_histogram, get_sklearn_one_hot_mlinspect_new_histogram, \
    get_one_hot_df_creation_str_mlinspect_new_histogram, get_pandas_join_mlinspect_old_lineage, \
    get_multiple_dfs_creation_str_mlinspect_old_lineage, get_pandas_join_mlinspect_new_lineage, \
    get_multiple_dfs_creation_str_mlinspect_new_lineage, get_pandas_dropna_mlinspect_old_lineage, \
    get_dropna_df_creation_str_mlinspect_old_lineage, get_pandas_dropna_mlinspect_new_lineage, \
    get_dropna_df_creation_str_mlinspect_new_lineage, get_one_hot_df_creation_str_mlinspect_old_lineage, \
    get_sklearn_one_hot_mlinspect_old_lineage, get_sklearn_one_hot_mlinspect_new_lineage, \
    get_one_hot_df_creation_str_mlinspect_new_lineage, do_op_histogram_benchmarks, do_op_lineage_benchmarks

data_frame_rows = 1000
repeats = 2


def test_join_orig():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_orig(), setup=get_multiple_dfs_creation_str_orig(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_join_mlinspect_old_histogram_histogram():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_mlinspect_old_histogram(),
                           setup=get_multiple_dfs_creation_str_mlinspect_old_histogram(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_join_mlinspect_old_histogram_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_mlinspect_old_lineage(),
                           setup=get_multiple_dfs_creation_str_mlinspect_old_lineage(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_join_mlinspect_new_histogram_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_mlinspect_new_lineage(),
                           setup=get_multiple_dfs_creation_str_mlinspect_new_lineage(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_join_mlinspect_new_histogram():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_join_mlinspect_new_histogram(),
                           setup=get_multiple_dfs_creation_str_mlinspect_new_histogram(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_dropna_orig():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_orig(), setup=get_dropna_df_creation_str_orig(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_dropna_mlinspect_old_histogram():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_mlinspect_old_histogram(),
                           setup=get_dropna_df_creation_str_mlinspect_old_histogram(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_dropna_mlinspect_old_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_mlinspect_old_lineage(),
                           setup=get_dropna_df_creation_str_mlinspect_old_lineage(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_dropna_mlinspect_new_histogram():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_mlinspect_new_histogram(),
                           setup=get_dropna_df_creation_str_mlinspect_new_histogram(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_dropna_mlinspect_new_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_pandas_dropna_mlinspect_new_lineage(),
                           setup=get_dropna_df_creation_str_mlinspect_new_lineage(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_one_hot_orig():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_orig(), setup=get_one_hot_df_creation_str_orig(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_one_hot_mlinspect_old_histogram():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_old_histogram(),
                           setup=get_one_hot_df_creation_str_mlinspect_old_histogram(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_one_hot_mlinspect_old_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_old_lineage(),
                           setup=get_one_hot_df_creation_str_mlinspect_old_lineage(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_one_hot_mlinspect_new_histogram():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_new_histogram(),
                           setup=get_one_hot_df_creation_str_mlinspect_new_histogram(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_one_hot_mlinspect_new_lineage():
    """ Performance experiments """
    result = timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_new_lineage(),
                           setup=get_one_hot_df_creation_str_mlinspect_new_lineage(data_frame_rows),
                           repeat=repeats, number=1)
    print(result)


def test_do_op_histogram_benchmarks():
    """ Performance experiments """
    benchmark_results = do_op_histogram_benchmarks(data_frame_rows)
    assert benchmark_results["dropna orig"]
    assert benchmark_results["dropna old"]
    assert benchmark_results["dropna new"]

    assert benchmark_results["join orig"]
    assert benchmark_results["join old"]
    assert benchmark_results["join new"]

    assert benchmark_results["one-hot orig"]
    assert benchmark_results["one-hot old"]
    assert benchmark_results["one-hot new"]


def test_do_op_lineage_benchmarks():
    """ Performance experiments """
    benchmark_results = do_op_lineage_benchmarks(data_frame_rows)
    assert benchmark_results["dropna orig"]
    assert benchmark_results["dropna old"]
    assert benchmark_results["dropna new"]

    assert benchmark_results["join orig"]
    assert benchmark_results["join old"]
    assert benchmark_results["join new"]

    assert benchmark_results["one-hot orig"]
    assert benchmark_results["one-hot old"]
    assert benchmark_results["one-hot new"]
