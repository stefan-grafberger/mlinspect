"""
Instrument and executes the pipeline
"""
import ast
import copy
from typing import List

import nbformat
from nbconvert import PythonExporter

from ..inspections.inspection import Inspection
from ..backends.all_backends import get_all_backends
from .call_capture_transformer import CallCaptureTransformer
from .dag_node import CodeReference, DagNodeIdentifier
from .inspection_result import InspectionResult
from .wir_extractor import WirExtractor
from .wir_to_dag_transformer import WirToDagTransformer


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    REPLACEMENT_TYPE_MAP = dict(replacement for backend in get_all_backends() for replacement in
                                backend.replacement_type_map.items())

    script_scope = {}
    backend_map = {}
    code_reference_to_module = {}

    def run(self, notebook_path: str or None, python_path: str or None, python_code: str or None,
            inspections: List[Inspection]) -> InspectionResult:
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use
        self.initialize_static_variables(inspections)

        source_code = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(source_code)
        original_parsed_ast = copy.deepcopy(parsed_ast)  # Some ast functions modify in-place

        parsed_modified_ast = self.instrument_pipeline(parsed_ast)

        exec(compile(parsed_modified_ast, filename="<ast>", mode="exec"), PipelineExecutor.script_scope)

        wir_extractor = WirExtractor(original_parsed_ast)
        wir_extractor.extract_wir()

        code_reference_to_description = {}
        for backend in self.backend_map.values():
            code_reference_to_description = {**code_reference_to_description, **backend.code_reference_to_description}
        wir = wir_extractor.add_runtime_info(self.code_reference_to_module, code_reference_to_description)

        for backend in self.backend_map.values():
            wir = backend.preprocess_wir(wir)

        dag = WirToDagTransformer.extract_dag(wir)

        for backend in self.backend_map.values():
            dag = backend.postprocess_dag(dag)

        inspection_to_call_to_annotation = self.build_inspection_result_map(dag)

        return InspectionResult(dag, inspection_to_call_to_annotation)

    def initialize_static_variables(self, inspections):
        """
        Because variables that the user code has to update are static, we need to set them here
        """
        PipelineExecutor.backend_map = dict((backend.prefix, backend) for backend in get_all_backends())
        for backend in self.backend_map.values():
            backend.inspections = inspections
        PipelineExecutor.script_scope = {}
        PipelineExecutor.code_reference_to_module = {}

    def build_inspection_result_map(self, dag):
        """
        Get the inspection DAG annotations from the backend and build a map with it for convenient usage
        """
        dag_node_identifiers_to_dag_nodes = {}
        for node in dag.nodes:
            dag_node_identifier = DagNodeIdentifier(node.operator_type, node.code_reference, node.description)
            dag_node_identifiers_to_dag_nodes[dag_node_identifier] = node

        dag_node_identifier_to_inspection_output = {}
        for backend in self.backend_map.values():
            dag_node_identifier_to_inspection_output = {**dag_node_identifier_to_inspection_output,
                                                        **backend.dag_node_identifier_to_inspection_output}
        inspection_to_dag_node_to_annotation = {}
        for dag_node_identifier, inspection_output_map in dag_node_identifier_to_inspection_output.items():
            for inspection, annotation in inspection_output_map.items():
                dag_node_to_annotation = inspection_to_dag_node_to_annotation.get(inspection, {})
                dag_node = dag_node_identifiers_to_dag_nodes[dag_node_identifier]
                dag_node_to_annotation[dag_node] = annotation
                inspection_to_dag_node_to_annotation[inspection] = dag_node_to_annotation
        return inspection_to_dag_node_to_annotation

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

    def before_call_used_value(self, subscript, call_code, value_code, value_value, code_reference):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript, value_value)
        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            backend.before_call_used_value(function_info, subscript, call_code, value_code,
                                           value_value, code_reference)

        return value_value

    def before_call_used_args(self, subscript, call_code, args_code, code_reference, store, args_values):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript, store=store)

        if store:
            self.code_reference_to_module[code_reference] = function_info

        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            backend.before_call_used_args(function_info, subscript, call_code, args_code, code_reference, store,
                                          args_values)

        return args_values

    def before_call_used_kwargs(self, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        assert not subscript  # we currently only consider __getitem__ subscripts, these do not take kwargs
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript)
        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            backend.before_call_used_kwargs(function_info, subscript, call_code, kwargs_code,
                                            code_reference, kwargs_values)

        return kwargs_values

    def after_call_used(self, subscript, call_code, return_value, code_reference):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript, return_value)

        # FIXME: To properly handle all edge cases with chained method calls, we need to add end_col_offset
        #  and end_line_no to code_reference
        self.code_reference_to_module[code_reference] = function_info

        if function_prefix in self.backend_map:
            backend = self.backend_map[function_prefix]
            return backend.after_call_used(function_info, subscript, call_code, return_value, code_reference)

        return return_value

    @staticmethod
    def get_function_info_and_prefix(call_code, subscript, value=None, store=False):
        """
        Get the function info and find out which backend to call
        """
        if not subscript:
            function_string = PipelineExecutor.split_on_bracket(call_code)
            module_info = eval("inspect.getmodule(" + function_string + ")", PipelineExecutor.script_scope)
            function_name = PipelineExecutor.split_on_dot(function_string)
            function_info = (module_info.__name__, function_name)
        else:
            function_string = str(call_code.split("[", 1)[0])
            module_info = eval("inspect.getmodule(" + function_string + ")", PipelineExecutor.script_scope)

            if not store:
                function_info = (module_info.__name__, "__getitem__")
            else:
                function_info = (module_info.__name__, "__setitem__")

        if function_info[0] in PipelineExecutor.REPLACEMENT_TYPE_MAP:
            new_type = PipelineExecutor.REPLACEMENT_TYPE_MAP[function_info[0]]
            function_info = (new_type, function_info[1])

        # FIXME: move this into sklearn backend
        if value is not None and \
                function_info[0] == 'mlinspect.backends.sklearn_backend_transformer_wrapper' and \
                function_info[1] != "score":
            function_info = (value.module_name, str(function_string.split(".")[-1]))

        function_prefix = function_info[0].split(".", 1)[0]

        return function_info, function_prefix

    @staticmethod
    def split_on_bracket(call_code):
        """
        Extract the code to get the function call path from a string like "(some_value(())).some_function(xy())"
        """
        counter = 0
        found_index = None

        for index, char in enumerate(call_code):
            if counter == 0 and char == "(":
                found_index = index
                counter += 1
            elif counter != 0 and char == "(":
                counter += 1
            elif counter != 0 and char == ")":
                counter -= 1

        return call_code[:found_index]

    @staticmethod
    def split_on_dot(call_code):
        """
        Extract the function name from a string like "(some_value(())).some_function(xy())"
        """
        counter = 0
        dot_found = False
        last_dot_index = 0
        for index, char in enumerate(call_code):
            if index != 0 and char == "(" and not dot_found:
                counter += 1
            elif char == ")" and not dot_found:
                counter -= 1
            elif char == "." and counter == 0:
                last_dot_index = index + 1

        return call_code[last_dot_index:]


# How we instrument the calls

# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
singleton = PipelineExecutor()


def before_call_used_value(subscript, call_code, value_code, value_value, ast_lineno, ast_col_offset,
                           ast_end_lineno, ast_end_col_offset):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.before_call_used_value(subscript, call_code, value_code, value_value,
                                            CodeReference(ast_lineno, ast_col_offset, ast_end_lineno,
                                                          ast_end_col_offset))


def before_call_used_args(subscript, call_code, args_code, ast_lineno, ast_col_offset,
                          ast_end_lineno, ast_end_col_offset, store, args_values):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.before_call_used_args(subscript, call_code, args_code,
                                           CodeReference(ast_lineno, ast_col_offset, ast_end_lineno,
                                                         ast_end_col_offset),
                                           store,
                                           args_values)


def before_call_used_kwargs(subscript, call_code, kwargs_code, ast_lineno, ast_col_offset,
                            ast_end_lineno, ast_end_col_offset, **kwarg_values):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.before_call_used_kwargs(subscript, call_code, kwargs_code,
                                             CodeReference(ast_lineno, ast_col_offset, ast_end_lineno,
                                                           ast_end_col_offset),
                                             kwarg_values)


def after_call_used(subscript, call_code, return_value, ast_lineno, ast_col_offset, ast_end_lineno, ast_end_col_offset):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.after_call_used(subscript, call_code, return_value,
                                     CodeReference(ast_lineno, ast_col_offset, ast_end_lineno, ast_end_col_offset))
