"""
Functions to benchmark mlinspect
"""
import timeit
from inspect import cleandoc


def do_op_histogram_benchmarks(data_frame_rows, repeats=1):
    """ Performance experiments """
    benchmark_results = {
        "dropna orig": timeit.repeat(stmt=get_pandas_dropna_orig(),
                                     setup=get_dropna_df_creation_str_orig(data_frame_rows),
                                     repeat=repeats, number=1),
        "dropna old": timeit.repeat(stmt=get_pandas_dropna_mlinspect_old_histogram(),
                                    setup=get_dropna_df_creation_str_mlinspect_old_histogram(data_frame_rows),
                                    repeat=repeats, number=1),
        "dropna new": timeit.repeat(stmt=get_pandas_dropna_mlinspect_new_histogram(),
                                    setup=get_dropna_df_creation_str_mlinspect_new_histogram(data_frame_rows),
                                    repeat=repeats, number=1),
        "join orig": timeit.repeat(stmt=get_pandas_join_orig(),
                                   setup=get_multiple_dfs_creation_str_orig(data_frame_rows),
                                   repeat=repeats, number=1),
        "join old": timeit.repeat(stmt=get_pandas_join_mlinspect_old_histogram(),
                                  setup=get_multiple_dfs_creation_str_mlinspect_old_histogram(data_frame_rows),
                                  repeat=repeats, number=1),
        "join new": timeit.repeat(stmt=get_pandas_join_mlinspect_new_histogram(),
                                  setup=get_multiple_dfs_creation_str_mlinspect_new_histogram(data_frame_rows),
                                  repeat=repeats, number=1),
        "one-hot orig": timeit.repeat(stmt=get_sklearn_one_hot_orig(),
                                      setup=get_one_hot_df_creation_str_orig(data_frame_rows),
                                      repeat=repeats, number=1),
        "one-hot old": timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_old_histogram(),
                                     setup=get_one_hot_df_creation_str_mlinspect_old_histogram(data_frame_rows),
                                     repeat=repeats, number=1),
        "one-hot new": timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_new_histogram(),
                                     setup=get_one_hot_df_creation_str_mlinspect_new_histogram(data_frame_rows),
                                     repeat=repeats, number=1)
    }

    return benchmark_results


def do_op_lineage_benchmarks(data_frame_rows, repeats=1):
    """ Performance experiments """
    benchmark_results = {
        "dropna orig": timeit.repeat(stmt=get_pandas_dropna_orig(),
                                     setup=get_dropna_df_creation_str_orig(data_frame_rows),
                                     repeat=repeats, number=1),
        "dropna old": timeit.repeat(stmt=get_pandas_dropna_mlinspect_old_lineage(),
                                    setup=get_dropna_df_creation_str_mlinspect_old_lineage(data_frame_rows),
                                    repeat=repeats, number=1),
        "dropna new": timeit.repeat(stmt=get_pandas_dropna_mlinspect_new_lineage(),
                                    setup=get_dropna_df_creation_str_mlinspect_new_lineage(data_frame_rows),
                                    repeat=repeats, number=1),
        "join orig": timeit.repeat(stmt=get_pandas_join_orig(),
                                   setup=get_multiple_dfs_creation_str_orig(data_frame_rows),
                                   repeat=repeats, number=1),
        "join old": timeit.repeat(stmt=get_pandas_join_mlinspect_old_lineage(),
                                  setup=get_multiple_dfs_creation_str_mlinspect_old_lineage(data_frame_rows),
                                  repeat=repeats, number=1),
        "join new": timeit.repeat(stmt=get_pandas_join_mlinspect_new_lineage(),
                                  setup=get_multiple_dfs_creation_str_mlinspect_new_lineage(data_frame_rows),
                                  repeat=repeats, number=1),
        "one-hot orig": timeit.repeat(stmt=get_sklearn_one_hot_orig(),
                                      setup=get_one_hot_df_creation_str_orig(data_frame_rows),
                                      repeat=repeats, number=1),
        "one-hot old": timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_old_lineage(),
                                     setup=get_one_hot_df_creation_str_mlinspect_old_lineage(data_frame_rows),
                                     repeat=repeats, number=1),
        "one-hot new": timeit.repeat(stmt=get_sklearn_one_hot_mlinspect_new_lineage(),
                                     setup=get_one_hot_df_creation_str_mlinspect_new_lineage(data_frame_rows),
                                     repeat=repeats, number=1)
    }

    return benchmark_results


def get_multiple_dfs_creation_str_orig(data_frame_rows):
    """ Performance experiments """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

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


def get_pandas_join_orig():
    """ Performance experiments """
    test_code = cleandoc("""
    orig_merged_df = df_a.merge(df_b, on='id')
    """)
    return test_code


def get_multiple_dfs_creation_str_mlinspect_old_histogram(data_frame_rows):
    """ Performance experiments """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

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
        
        from mlinspect.backends._pandas_backend import PandasBackend
        from mlinspect.backends._sklearn_backend import SklearnBackend
        from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext
        from mlinspect.inspections import HistogramForColumns
        from mlinspect.instrumentation._pipeline_executor import singleton
        
        singleton.inspections = [HistogramForColumns(["group_col_1"])]
        
        function_info = FunctionInfo('...', 'get_multiple_dfs')
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df_a
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df_a)
        annotated_df_a = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df_b
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df_b)
        annotated_df_b = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        """.format(sizes_before_join=sizes_before_join, start_with_offset=start_with_offset,
                   end_with_offset=end_with_offset))
    return test_code


def get_multiple_dfs_creation_str_mlinspect_old_lineage(data_frame_rows):
    """ Performance experiments """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

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
        from mlinspect.backends._pandas_backend import PandasBackend
        from mlinspect.backends._sklearn_backend import SklearnBackend
        from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext
        from mlinspect.inspections import RowLineage
        from mlinspect.instrumentation._pipeline_executor import singleton
        
        singleton.inspections = [RowLineage(10)]
        
        function_info = FunctionInfo('...', 'get_multiple_dfs')
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df_a
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df_a)
        annotated_df_a = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df_b
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df_b)
        annotated_df_b = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        """.format(sizes_before_join=sizes_before_join, start_with_offset=start_with_offset,
                   end_with_offset=end_with_offset))
    return test_code


def get_multiple_dfs_creation_str_mlinspect_new_lineage(data_frame_rows):
    """ Performance experiments """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

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
        
        lineage_id_list_a = ["LineageId(0, " + str(row_id) + ")" for row_id in range(len(df_a))]
        lineage_ids_a = pd.DataFrame({{"RowLineage(10)": pd.Series(lineage_id_list_a, dtype="object")}})
        lineage_id_list_b = ["LineageId(1, " + str(row_id) + ")" for row_id in range(len(df_b))]
        lineage_ids_b = pd.DataFrame({{"RowLineage(10)": pd.Series(lineage_id_list_b, dtype="object")}})
        """.format(sizes_before_join=sizes_before_join, start_with_offset=start_with_offset,
                   end_with_offset=end_with_offset))
    return test_code


def get_pandas_join_mlinspect_old_histogram():
    """ Performance experiments """
    test_code = cleandoc("""
    function_info = FunctionInfo('pandas.core.frame', 'merge')
    operator_context = OperatorContext(OperatorType.JOIN, function_info)
    
    input_infos = PandasBackend.before_call(operator_context, [annotated_df_a, annotated_df_b])
    merged_df = input_infos[0].result_data.merge(input_infos[1].result_data, on='id')
    backend_result = PandasBackend.after_call(operator_context,
                                              input_infos,
                                              merged_df)
    annotated_merged_df = backend_result.annotated_dfobject
    # print(backend_result.annotated_dfobject.result_data)
    # print(backend_result.annotated_dfobject.result_annotation)
    # print(backend_result.dag_node_annotation)
    """)
    return test_code


def get_pandas_join_mlinspect_old_lineage():
    """ Performance experiments """
    test_code = cleandoc("""
    function_info = FunctionInfo('pandas.core.frame', 'merge')
    operator_context = OperatorContext(OperatorType.JOIN, function_info)
    
    input_infos = PandasBackend.before_call(operator_context, [annotated_df_a, annotated_df_b])
    merged_df = input_infos[0].result_data.merge(input_infos[1].result_data, on='id')
    backend_result = PandasBackend.after_call(operator_context,
                                              input_infos,
                                              merged_df)
    annotated_merged_df = backend_result.annotated_dfobject
    # print(backend_result.annotated_dfobject.result_data)
    # print(backend_result.annotated_dfobject.result_annotation)
    # print(backend_result.dag_node_annotation)
    """)
    return test_code


def get_pandas_join_mlinspect_new_lineage():
    """ Performance experiments """
    test_code = cleandoc("""
    df_a['lineage_left'] = lineage_ids_a
    df_b['lineage_right'] = lineage_ids_b
    
    join_result = df_a.merge(df_b, on='id')
    
    join_result['annotations'] = join_result["lineage_left"] + ", " + join_result["lineage_right"]
    join_result.drop('lineage_left', inplace=True, axis=1)
    join_result.drop('lineage_right', inplace=True, axis=1)
    dag_annotation = join_result.head(10)
    annotation = join_result.pop('annotations')
    
    # print(join_result)
    # print(annotation)
    # print(dag_annotation)
    """)
    return test_code


def get_multiple_dfs_creation_str_mlinspect_new_histogram(data_frame_rows):
    """ Performance experiments """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

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
        df_a_annotation = df_a['group_col_1']
        df_a_annotation.name = "mlinspect_annotation"
        """.format(sizes_before_join=sizes_before_join, start_with_offset=start_with_offset,
                   end_with_offset=end_with_offset))
    return test_code


def get_pandas_join_mlinspect_new_histogram():
    """ Performance experiments """
    test_code = cleandoc("""
    df_a_with_annot = pd.concat([df_a, df_a_annotation], axis=1)
    merged_df = df_a_with_annot.merge(df_b, on='id')
    
    histogram = merged_df.groupby(['group_col_1']).size()
    new_annotation = merged_df['mlinspect_annotation']
    merged_df = merged_df.drop(['mlinspect_annotation'], axis=1)
    
    # print(merged_df)
    # print(new_annotation)
    # print(histogram)
    """)
    return test_code


def get_dropna_df_creation_str_orig(data_frame_rows):
    """ Performance experiments """
    data_frame_rows_after_dropna = data_frame_rows
    data_frame_rows_before_dropna = int(data_frame_rows_after_dropna * (4 / 3))  # 3 categories + None, None dropped
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint, shuffle
        import random
        id_a = np.arange({data_frame_rows})
        shuffle(id_a)
        a = randint(0,100,size=({data_frame_rows}))
        b = randint(0,100,size=({data_frame_rows}))
        categories = ['cat_a', 'cat_b', 'cat_c', None]
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_2 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_3 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3), columns=['id', 'A', 'B', 
            'group_col_1', 'group_col_2', 'group_col_3'])
        """.format(data_frame_rows=data_frame_rows_before_dropna))
    return test_code


def get_pandas_dropna_orig():
    """ Performance experiments """
    test_code = cleandoc("""
    orig_df_a_dropna = df_a.dropna()
    """)
    return test_code


def get_dropna_df_creation_str_mlinspect_old_histogram(data_frame_rows):
    """ Performance experiments """
    data_frame_rows_after_dropna = data_frame_rows
    data_frame_rows_before_dropna = int(data_frame_rows_after_dropna * (4 / 3))  # 3 categories + None, None dropped
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint, shuffle
        import random
        id_a = np.arange({data_frame_rows})
        shuffle(id_a)
        a = randint(0,100,size=({data_frame_rows}))
        b = randint(0,100,size=({data_frame_rows}))
        categories = ['cat_a', 'cat_b', 'cat_c', None]
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_2 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_3 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3), columns=['id', 'A', 'B', 
            'group_col_1', 'group_col_2', 'group_col_3'])
        
        from mlinspect.backends._pandas_backend import PandasBackend
        from mlinspect.backends._sklearn_backend import SklearnBackend
        from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext
        from mlinspect.inspections import HistogramForColumns
        from mlinspect.instrumentation._pipeline_executor import singleton
        
        singleton.inspections = [HistogramForColumns(["group_col_1"])]
        
        function_info = FunctionInfo('...', 'get_multiple_dfs')
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df_a
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df_a)
        annotated_df_a = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        """.format(data_frame_rows=data_frame_rows_before_dropna))
    return test_code


def get_dropna_df_creation_str_mlinspect_old_lineage(data_frame_rows):
    """ Performance experiments """
    data_frame_rows_after_dropna = data_frame_rows
    data_frame_rows_before_dropna = int(data_frame_rows_after_dropna * (4 / 3))  # 3 categories + None, None dropped
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint, shuffle
        import random
        id_a = np.arange({data_frame_rows})
        shuffle(id_a)
        a = randint(0,100,size=({data_frame_rows}))
        b = randint(0,100,size=({data_frame_rows}))
        categories = ['cat_a', 'cat_b', 'cat_c', None]
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_2 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_3 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3), columns=['id', 'A', 'B', 
            'group_col_1', 'group_col_2', 'group_col_3'])
        from mlinspect.backends._pandas_backend import PandasBackend
        from mlinspect.backends._sklearn_backend import SklearnBackend
        from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext
        from mlinspect.inspections import RowLineage
        from mlinspect.instrumentation._pipeline_executor import singleton
        singleton.inspections = [RowLineage(10)]
        function_info = FunctionInfo('...', 'get_multiple_dfs')
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        input_infos = PandasBackend.before_call(operator_context, [])
        # df_a
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df_a)
        annotated_df_a = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        """.format(data_frame_rows=data_frame_rows_before_dropna))
    return test_code


def get_pandas_dropna_mlinspect_old_histogram():
    """ Performance experiments """
    test_code = cleandoc("""
    function_info = FunctionInfo('pandas.core.frame', 'dropna')
    operator_context = OperatorContext(OperatorType.SELECTION, function_info)
    
    input_infos_dropna = PandasBackend.before_call(operator_context, [annotated_df_a])
    mlinspect_df_a_dropna = input_infos_dropna[0].result_data.dropna()
    backend_result_dropna = PandasBackend.after_call(operator_context,
                                                     input_infos_dropna,
                                                     mlinspect_df_a_dropna)
    annotated_dropna_df = backend_result_dropna.annotated_dfobject
    # print(backend_result_dropna.annotated_dfobject.result_data)
    # print(backend_result_dropna.annotated_dfobject.result_annotation)
    # print(backend_result_dropna.dag_node_annotation)
    """)
    return test_code


def get_pandas_dropna_mlinspect_old_lineage():
    """ Performance experiments """
    test_code = cleandoc("""
    function_info = FunctionInfo('pandas.core.frame', 'dropna')
    operator_context = OperatorContext(OperatorType.SELECTION, function_info)
    
    input_infos_dropna = PandasBackend.before_call(operator_context, [annotated_df_a])
    mlinspect_df_a_dropna = input_infos_dropna[0].result_data.dropna()
    backend_result_dropna = PandasBackend.after_call(operator_context,
                                                     input_infos_dropna,
                                                     mlinspect_df_a_dropna)
    annotated_dropna_df = backend_result_dropna.annotated_dfobject
    
    # print(backend_result_dropna.annotated_dfobject.result_data)
    # print(backend_result_dropna.annotated_dfobject.result_annotation)
    # print(backend_result_dropna.dag_node_annotation)
    """)
    return test_code


def get_dropna_df_creation_str_mlinspect_new_histogram(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint, shuffle
        import random
        id_a = np.arange({data_frame_rows})
        shuffle(id_a)
        a = randint(0,100,size=({data_frame_rows}))
        b = randint(0,100,size=({data_frame_rows}))
        categories = ['cat_a', 'cat_b', 'cat_c', None]
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_2 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_3 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3), columns=['id', 'A', 'B', 
            'group_col_1', 'group_col_2', 'group_col_3'])
        df_a_annotation = df_a['group_col_1']
        df_a_annotation.name = "mlinspect_annotation"
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_dropna_df_creation_str_mlinspect_new_lineage(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        import pandas as pd
        import numpy as np
        from numpy.random import randint, shuffle
        import random
        id_a = np.arange({data_frame_rows})
        shuffle(id_a)
        a = randint(0,100,size=({data_frame_rows}))
        b = randint(0,100,size=({data_frame_rows}))
        categories = ['cat_a', 'cat_b', 'cat_c', None]
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_2 = pd.Series(random.choices(categories, k={data_frame_rows}))
        group_col_3 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3), columns=['id', 'A', 'B', 
            'group_col_1', 'group_col_2', 'group_col_3'])
        lineage_id_list_a = ["LineageId(0, " + str(row_id) + ")" for row_id in range(len(df_a))]
        lineage_ids_a = pd.DataFrame({{"RowLineage(10)": pd.Series(lineage_id_list_a, dtype="object")}})
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_pandas_dropna_mlinspect_new_histogram():
    """ Performance experiments """
    test_code = cleandoc("""
    df_a_with_annot = pd.concat([df_a, df_a_annotation], axis=1)
    new_dropna_df = df_a_with_annot.dropna(subset=["id", "A", "B", "group_col_1", "group_col_2", "group_col_3"])
    new_dropna_histogram = new_dropna_df.groupby(['group_col_1']).size()
    new_dropna__annotation = new_dropna_df['mlinspect_annotation']
    new_dropna_df = new_dropna_df.drop(['mlinspect_annotation'], axis=1)
    
    # print(new_dropna_df)
    # print(new_dropna__annotation)
    # print(new_dropna_histogram)
    """)
    return test_code


def get_pandas_dropna_mlinspect_new_lineage():
    """ Performance experiments """
    test_code = cleandoc("""
    df_a["annotations"] = lineage_ids_a
    new_dropna_df = df_a.dropna(subset=["id", "A", "B", "group_col_1", "group_col_2", "group_col_3"])
    dag_annotation = new_dropna_df.head(10)
    annotation = new_dropna_df.pop('annotations')
    
    # print(new_dropna_df)
    # print(annotation)
    # print(dag_annotation)
    """)
    return test_code


def get_one_hot_df_creation_str_orig(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(group_col_1), columns=["group_col_1"])
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_sklearn_one_hot_orig():
    """ Performance experiments """
    test_code = cleandoc("""
    one_hot_df = OneHotEncoder(sparse=False).fit_transform(df)
    # print(one_hot_df)
    """)
    return test_code


def get_one_hot_df_creation_str_mlinspect_old_histogram(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(group_col_1), columns=["group_col_1"])
        
        from mlinspect.backends._pandas_backend import PandasBackend
        from mlinspect.backends._sklearn_backend import SklearnBackend
        from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext
        from mlinspect.inspections import HistogramForColumns
        from mlinspect.instrumentation._pipeline_executor import singleton
        
        singleton.inspections = [HistogramForColumns(["group_col_1"])]
        
        function_info = FunctionInfo('...', 'get_multiple_dfs')
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df)
        annotated_df_a = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_one_hot_df_creation_str_mlinspect_old_lineage(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random
    
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(group_col_1), columns=["group_col_1"])
        
        from mlinspect.backends._pandas_backend import PandasBackend
        from mlinspect.backends._sklearn_backend import SklearnBackend
        from mlinspect import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext
        from mlinspect.inspections import RowLineage
        from mlinspect.instrumentation._pipeline_executor import singleton
        
        singleton.inspections = [RowLineage(10)]
        
        function_info = FunctionInfo('...', 'get_multiple_dfs')
        operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
        
        input_infos = PandasBackend.before_call(operator_context, [])
        # df
        backend_result = PandasBackend.after_call(operator_context,
                                                  input_infos,
                                                  df)
        annotated_df_a = backend_result.annotated_dfobject
        # print(backend_result.annotated_dfobject.result_data)
        # print(backend_result.annotated_dfobject.result_annotation)
        # print(backend_result.dag_node_annotation)
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_sklearn_one_hot_mlinspect_old_histogram():
    """ Performance experiments """
    test_code = cleandoc("""
    operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
    input_infos = SklearnBackend.before_call(operator_context, [annotated_df_a])
    mlinspect_old_one_hot_df = OneHotEncoder(sparse=False).fit_transform(input_infos[0].result_data)
    backend_result_one_hot = SklearnBackend.after_call(operator_context,
                                              input_infos,
                                              mlinspect_old_one_hot_df)
    
    mlinspect_old_annotated_one_hot_df = backend_result_one_hot.annotated_dfobject
    
    # print(backend_result_one_hot.annotated_dfobject.result_data)
    # print(backend_result_one_hot.annotated_dfobject.result_annotation)
    # print(backend_result_one_hot.dag_node_annotation)
    """)
    return test_code


def get_sklearn_one_hot_mlinspect_old_lineage():
    """ Performance experiments """
    test_code = cleandoc("""
    operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
    input_infos = SklearnBackend.before_call(operator_context, [annotated_df_a])
    mlinspect_old_one_hot_df = OneHotEncoder(sparse=False).fit_transform(input_infos[0].result_data)
    backend_result_one_hot = SklearnBackend.after_call(operator_context,
                                              input_infos,
                                              mlinspect_old_one_hot_df)
    mlinspect_old_annotated_one_hot_df = backend_result_one_hot.annotated_dfobject
    # print(backend_result_one_hot.annotated_dfobject.result_data)
    # print(backend_result_one_hot.annotated_dfobject.result_annotation)
    # print(backend_result_one_hot.dag_node_annotation)
    """)
    return test_code


def get_one_hot_df_creation_str_mlinspect_new_histogram(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(group_col_1), columns=["group_col_1"])
        annotation = df[["group_col_1"]]
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_one_hot_df_creation_str_mlinspect_new_lineage(data_frame_rows):
    """ Performance experiments """
    test_code = cleandoc("""
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import random
        categories = ['cat_a', 'cat_b', 'cat_c']
        group_col_1 = pd.Series(random.choices(categories, k={data_frame_rows}))
        df = pd.DataFrame(zip(group_col_1), columns=["group_col_1"])
        lineage_id_list_one_hot = ["LineageId(0, " + str(row_id) + ")" 
                                   for row_id in range(len(df))]
        lineage_ids_one_hot = pd.Series(lineage_id_list_one_hot, dtype="object")
        """.format(data_frame_rows=data_frame_rows))
    return test_code


def get_sklearn_one_hot_mlinspect_new_histogram():
    """ Performance experiments """
    test_code = cleandoc("""
    result_data = OneHotEncoder(sparse=False).fit_transform(df)
    new_annotation = annotation
    dag_node_annotation = annotation.groupby("group_col_1").size()
    # print(result_data)
    # print(new_annotation)
    # print(dag_node_annotation)
    """)
    return test_code


def get_sklearn_one_hot_mlinspect_new_lineage():
    """ Performance experiments """
    test_code = cleandoc("""
    mlinspect_new_one_hot_df = OneHotEncoder(sparse=False).fit_transform(df)
    dag_annotation = pd.DataFrame({"array": list(mlinspect_new_one_hot_df[:10]), 
                                   "mlinspect_lineage": lineage_ids_one_hot.head(10)})
    annotation = lineage_ids_one_hot
    
    # print(mlinspect_new_one_hot_df)
    # print(annotation)
    # print(dag_annotation)
    """)
    return test_code
