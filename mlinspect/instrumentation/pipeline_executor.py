"""
Instrument and executes the pipeline
"""
import ast
import copy
from typing import List

import nbformat
from nbconvert import PythonExporter

from .analyzers.analyzer import Analyzer
from .backends.pandas_backend import PandasBackend
from .backends.sklearn_backend import SklearnBackend
from .call_capture_transformer import CallCaptureTransformer
from .inspection_result import InspectionResult
from .wir_extractor import WirExtractor
from .wir_to_dag_transformer import WirToDagTransformer


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    script_scope = {}

    backend_map = {
        PandasBackend.prefix: PandasBackend(),
        SklearnBackend.prefix: SklearnBackend()
    }

    def __init__(self):
        self.ast_call_node_id_to_module = {}

    def run(self, notebook_path: str or None, python_path: str or None, python_code: str or None,
            analyzers: List[Analyzer]) -> InspectionResult:
        """
        Instrument and execute the pipeline
        """
        for backend in self.backend_map.values():
            backend.analyzers = analyzers
        # pylint: disable=no-self-use
        PipelineExecutor.script_scope = {}

        source_code = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(source_code)
        original_parsed_ast = copy.deepcopy(parsed_ast)  # Some ast functions modify in-place

        parsed_modified_ast = self.instrument_pipeline(parsed_ast)

        exec(compile(parsed_modified_ast, filename="<ast>", mode="exec"), PipelineExecutor.script_scope)

        wir_extractor = WirExtractor(original_parsed_ast)
        wir_extractor.extract_wir()

        ast_call_node_id_to_description = {}
        for backend in self.backend_map.values():
            ast_call_node_id_to_description = {**ast_call_node_id_to_description, **backend.call_description_map}

        wir = wir_extractor.add_runtime_info(self.ast_call_node_id_to_module, ast_call_node_id_to_description)

        dag = WirToDagTransformer.extract_dag(wir)

        analyzer_to_call_to_annotation = self.build_analyzer_result_map()

        return InspectionResult(dag, analyzer_to_call_to_annotation)

    def build_analyzer_result_map(self):
        """
        Get the analyzer DAG annotations from the backend and build a map with it for convenient usage
        """
        ast_call_node_id_to_annotation = {}
        for backend in self.backend_map.values():
            ast_call_node_id_to_annotation = {**ast_call_node_id_to_annotation, **backend.call_analyzer_output_map}
        analyzer_to_call_to_annotation = {}
        for call_id, analyzer_output_map in ast_call_node_id_to_annotation.items():
            for analyzer, annotation in analyzer_output_map.items():
                call_id_to_annotation = analyzer_to_call_to_annotation.get(analyzer, {})
                call_id_to_annotation[call_id] = annotation
                analyzer_to_call_to_annotation[analyzer] = call_id_to_annotation
        return analyzer_to_call_to_annotation

    @staticmethod
    def instrument_pipeline(parsed_ast):
        """
        Instrument the pipeline AST to instrument function calls
        """
        call_capture_transformer = CallCaptureTransformer()
        parsed_modified_ast = call_capture_transformer.visit(parsed_ast)
        parsed_modified_ast = ast.fix_missing_locations(parsed_modified_ast)
        func_import_node = ast.ImportFrom(module='mlinspect.instrumentation.pipeline_executor',
                                          names=[ast.alias(name='before_call_used_value', asname=None),
                                                 ast.alias(name='before_call_used_args', asname=None),
                                                 ast.alias(name='before_call_used_kwargs', asname=None),
                                                 ast.alias(name='after_call_used', asname=None)],
                                          level=0)
        parsed_modified_ast.body.insert(2, func_import_node)
        inspect_import_node = ast.Import(names=[ast.alias(name='inspect', asname=None)])
        parsed_modified_ast.body.insert(3, inspect_import_node)
        parsed_modified_ast = ast.fix_missing_locations(parsed_modified_ast)
        return parsed_modified_ast

    @staticmethod
    def load_source_code(notebook_path, python_path, python_code):
        """
        Load the pipeline source code from the specified source
        """
        source_code = ""
        sources = [notebook_path, python_path, python_code]
        assert sum(source is not None for source in sources) == 1
        if python_path is not None:
            with open(python_path) as file:
                source_code = file.read()
        elif notebook_path is not None:
            with open(notebook_path) as file:
                notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
                exporter = PythonExporter()
                source_code, _ = exporter.from_notebook_node(notebook)
        elif python_code is not None:
            source_code = python_code
        return source_code

    def before_call_used_value(self, subscript, call_code, value_code, value_value, ast_lineno, ast_col_offset):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        if not subscript:
            function = str(call_code.split("(", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, str(function.split(".")[-1]))
        else:
            function = str(call_code.split("[", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, "__getitem__")

        # TODO: Extract this into backend
        if function_info[0] == 'mlinspect.instrumentation.backends.pandas_backend_frame_wrapper':
            function_info = ('pandas.core.frame', function_info[1])

        function_prefix = function_info[0].split(".", 1)[0]
        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            backend.before_call_used_value(function_info, subscript, call_code, value_code,
                                           value_value, ast_lineno, ast_col_offset)

        return value_value

    def before_call_used_args(self, subscript, call_code, args_code, ast_lineno, ast_col_offset, args_values):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        if not subscript:
            function = str(call_code.split("(", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, str(function.split(".")[-1]))
        else:
            function = str(call_code.split("[", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, "__getitem__")

        # TODO: Extract this into backend
        if function_info[0] == 'mlinspect.instrumentation.backends.pandas_backend_frame_wrapper':
            function_info = ('pandas.core.frame', function_info[1])

        function_prefix = function_info[0].split(".", 1)[0]
        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            backend.before_call_used_args(function_info, subscript, call_code, args_code, ast_lineno,
                                          ast_col_offset, args_values)

        return args_values

    def before_call_used_kwargs(self, subscript, call_code, kwargs_code, ast_lineno, ast_col_offset, kwargs_values):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        assert not subscript  # we currently only consider __getitem__ subscripts, these do not take kwargs
        function = str(call_code.split("(", 1)[0])
        module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

        function_info = (module_info.__name__, str(function.split(".")[-1]))

        # TODO: Extract this into backend
        if function_info[0] == 'mlinspect.instrumentation.backends.pandas_backend_frame_wrapper':
            function_info = ('pandas.core.frame', function_info[1])

        function_prefix = function_info[0].split(".", 1)[0]
        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            backend.before_call_used_kwargs(function_info, subscript, call_code, kwargs_code,
                                            ast_lineno, ast_col_offset, kwargs_values)

        return kwargs_values

    def after_call_used(self, subscript, call_code, return_value, ast_lineno, ast_col_offset):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        if not subscript:
            function = str(call_code.split("(", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, str(function.split(".")[-1]))
        else:
            function = str(call_code.split("[", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, "__getitem__")

        # TODO: Extract this into backend
        if function_info[0] == 'mlinspect.instrumentation.backends.pandas_backend_frame_wrapper':
            function_info = ('pandas.core.frame', function_info[1])
        self.ast_call_node_id_to_module[(ast_lineno, ast_col_offset)] = function_info

        function_prefix = function_info[0].split(".", 1)[0]
        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            return backend.after_call_used(function_info, subscript, call_code, return_value,
                                           ast_lineno, ast_col_offset)

        return return_value


# How we instrument the calls

# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
singleton = PipelineExecutor()


def before_call_used_value(subscript, call_code, value_code, value_value, ast_lineno, ast_col_offset):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.before_call_used_value(subscript, call_code, value_code, value_value, ast_lineno, ast_col_offset)


def before_call_used_args(subscript, call_code, args_code, ast_lineno, ast_col_offset, args_values):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.before_call_used_args(subscript, call_code, args_code, ast_lineno, ast_col_offset, args_values)


def before_call_used_kwargs(subscript, call_code, kwargs_code, ast_lineno, ast_col_offset, **kwarg_values):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.before_call_used_kwargs(subscript, call_code, kwargs_code, ast_lineno, ast_col_offset,
                                             kwarg_values)


def after_call_used(subscript, call_code, return_value, ast_lineno, ast_col_offset):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.after_call_used(subscript, call_code, return_value, ast_lineno, ast_col_offset)
