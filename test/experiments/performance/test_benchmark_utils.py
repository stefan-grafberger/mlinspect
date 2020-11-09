"""
Tests whether the benchmark utils work
"""

from experiments.performance._benchmark_utils import do_op_instrumentation_benchmarks, OperatorBenchmarkType, \
    do_op_inspections_benchmarks, do_full_pipeline_benchmarks, PipelineBenchmarkType


def test_instrumentation_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    for op_type in OperatorBenchmarkType:
        benchmark_results = do_op_instrumentation_benchmarks(100, op_type)

        assert benchmark_results["no mlinspect"]
        assert benchmark_results["no inspection"]
        assert benchmark_results["one inspection"]
        assert benchmark_results["two inspections"]
        assert benchmark_results["three inspections"]


def test_inspection_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    for op_type in OperatorBenchmarkType:
        benchmark_results = do_op_inspections_benchmarks(100, op_type)

        assert benchmark_results["empty inspection"]
        assert benchmark_results["MaterializeFirstOutputRows(10)"]
        assert benchmark_results["RowLineage(10)"]
        assert benchmark_results["HistogramForColumns(['group_col_1'])"]
        assert benchmark_results["HistogramForColumns(['group_col_1', 'group_col_2', 'group_col_3'])"]


def test_full_pipeline_benchmarks():
    """
    Tests whether the pipeline works with instrumentation
    """
    for pipeline in PipelineBenchmarkType:
        benchmark_results = do_full_pipeline_benchmarks(pipeline)

        assert benchmark_results["no mlinspect"]
        assert benchmark_results["no inspection"]
        assert benchmark_results["one inspection"]
        assert benchmark_results["two inspections"]
        assert benchmark_results["three inspections"]
