"""
Instrument and executes the pipeline
"""
import ast
import copy
from collections import OrderedDict
from typing import Iterable

import nbformat
from nbconvert import PythonExporter

from ..checks._check import Check
from ..inspections._inspection import Inspection
from ..inspections._inspection_result import InspectionResult
from ..backends._all_backends import get_all_backends
from ._call_capture_transformer import CallCaptureTransformer
from ._dag_node import CodeReference, DagNodeIdentifier
from ._wir_extractor import WirExtractor
from ._wir_to_dag_transformer import WirToDagTransformer
from .._inspector_result import InspectorResult


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """

    script_scope = {}
    backends = []

    def run(self, notebook_path: str or None, python_path: str or None, python_code: str or None,
            inspections: Iterable[Inspection], checks: Iterable[Check], reset_state=True) -> InspectorResult:
        """
        Instrument and execute the pipeline and evaluate all checks
        """
        # pylint: disable=too-many-arguments
        check_inspections = set()
        for check in checks:
            check_inspections.update(check.required_inspections)
        all_inspections = list(set(inspections).union(check_inspections))

        if reset_state:
            # reset_state=False should only be used internally for performance experiments etc!
            # It does not ensure the same inspections are still used as args etc.
            self.initialize_static_variables(all_inspections)

        inspection_result = self.run_inspections(notebook_path, python_code, python_path)
        check_to_results = OrderedDict((check, check.evaluate(inspection_result)) for check in checks)
        return InspectorResult(inspection_result.dag, inspection_result.inspection_to_annotations, check_to_results)

    def run_inspections(self, notebook_path, python_code, python_path) -> InspectionResult:
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use, too-many-locals
        source_code = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(source_code)
        original_parsed_ast = copy.deepcopy(parsed_ast)  # Some ast functions modify in-place
        parsed_modified_ast = self.instrument_pipeline(parsed_ast)
        exec(compile(parsed_modified_ast, filename="<ast>", mode="exec"), PipelineExecutor.script_scope)
        wir_extractor = WirExtractor(original_parsed_ast)
        wir_extractor.extract_wir()
        code_reference_to_description = {}
        code_reference_to_module = {}
        for backend in self.backends:
            code_reference_to_description = {**code_reference_to_description, **backend.code_reference_to_description}
            code_reference_to_module = {**code_reference_to_module, **backend.code_reference_to_module}
        wir = wir_extractor.add_runtime_info(code_reference_to_module, code_reference_to_description)
        for backend in self.backends:
            wir = backend.process_wir(wir)
        dag = WirToDagTransformer.extract_dag(wir)
        for backend in self.backends:
            dag = backend.process_dag(dag)
        inspection_to_call_to_annotation = self.build_inspection_result_map(dag)
        return InspectionResult(dag, inspection_to_call_to_annotation)

    def initialize_static_variables(self, inspections):
        """
        Because variables that the user code has to update are static, we need to set them here
        """
        PipelineExecutor.backends = get_all_backends()
        for backend in self.backends:
            backend.inspections = inspections
        PipelineExecutor.script_scope = {}

    def build_inspection_result_map(self, dag):
        """
        Get the inspection DAG annotations from the backend and build a map with it for convenient usage
        """
        dag_node_identifiers_to_dag_nodes = {}
        for node in dag.nodes:
            dag_node_identifier = DagNodeIdentifier(node.operator_type, node.code_reference, node.description)
            dag_node_identifiers_to_dag_nodes[dag_node_identifier] = node

        dag_node_identifier_to_inspection_output = {}
        dag_node_identifier_to_columns = {}
        for backend in self.backends:
            dag_node_identifier_to_inspection_output = {**dag_node_identifier_to_inspection_output,
                                                        **backend.dag_node_identifier_to_inspection_output}
            dag_node_identifier_to_columns = {**dag_node_identifier_to_columns,
                                              **backend.dag_node_identifier_to_columns}
        for dag_node_identifier, inspection_output_map in dag_node_identifier_to_columns.items():
            if dag_node_identifier in dag_node_identifiers_to_dag_nodes:  # Always true if reset_state
                dag_node = dag_node_identifiers_to_dag_nodes[dag_node_identifier]
                dag_node_columns = dag_node_identifier_to_columns[dag_node_identifier]
                dag_node.columns = dag_node_columns

        inspection_to_dag_node_to_annotation = OrderedDict()
        for dag_node_identifier, inspection_output_map in dag_node_identifier_to_inspection_output.items():
            for inspection, annotation in inspection_output_map.items():
                if dag_node_identifier in dag_node_identifiers_to_dag_nodes:  # Always true if reset_state
                    dag_node_to_annotation = inspection_to_dag_node_to_annotation.get(inspection, OrderedDict())
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
        func_import_node = ast.ImportFrom(module='mlinspect.instrumentation._pipeline_executor',
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
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript)
        for backend in self.backends:
            if backend.is_responsible_for_call(function_info, function_prefix, value_value):
                backend.before_call_used_value(function_info, subscript, call_code, value_code,
                                               value_value, code_reference)
                return value_value

        return value_value

    def before_call_used_args(self, subscript, call_code, args_code, code_reference, store, args_values):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript, store=store)

        for backend in self.backends:
            if backend.is_responsible_for_call(function_info, function_prefix):
                backend.before_call_used_args(function_info, subscript, call_code, args_code, code_reference, store,
                                              args_values)
                return args_values

        return args_values

    def before_call_used_kwargs(self, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        assert not subscript  # we currently only consider __getitem__ subscripts, these do not take kwargs
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript)
        for backend in self.backends:
            if backend.is_responsible_for_call(function_info, function_prefix):
                backend.before_call_used_kwargs(function_info, subscript, call_code, kwargs_code,
                                                code_reference, kwargs_values)
                return kwargs_values

        return kwargs_values

    def after_call_used(self, subscript, call_code, return_value, code_reference):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        function_info, function_prefix = self.get_function_info_and_prefix(call_code, subscript)

        for backend in self.backends:
            if backend.is_responsible_for_call(function_info, function_prefix, return_value):
                return backend.after_call_used(function_info, subscript, call_code, return_value, code_reference)

        return return_value

    @staticmethod
    def get_function_info_and_prefix(call_code, subscript, store=False):
        """
        Get the function info and find out which backend to call
        """
        if not subscript:
            function_string = PipelineExecutor.split_on_bracket(call_code)
            module_info = eval("inspect.getmodule(" + function_string + ")", PipelineExecutor.script_scope)
            function_name = PipelineExecutor.split_on_dot(function_string)
            if module_info is not None:
                function_info = (module_info.__name__, function_name)
            else:
                module_info = "builtin_function_or_method"  # TODO: Detect module/object path properly
                function_info = (module_info, function_name)
        else:
            function_string = str(call_code.split("[", 1)[0])
            module_info = eval("inspect.getmodule(" + function_string + ")", PipelineExecutor.script_scope)

            if not store:
                function_info = (module_info.__name__, "__getitem__")
            else:
                function_info = (module_info.__name__, "__setitem__")

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
