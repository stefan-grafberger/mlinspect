"""
Instrument and executes the pipeline
"""
import ast
from typing import Iterable, List

import gorilla
import nbformat
import networkx
from astmonkey.transformers import ParentChildNodeTransformer
from nbconvert import PythonExporter

from .. import monkeypatching
from ._call_capture_transformer import CallCaptureTransformer
from .._inspector_result import InspectorResult
from ..checks._check import Check
from ..inspections import InspectionResult
from ..inspections._inspection import Inspection


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    # pylint: disable=too-many-instance-attributes

    source_code_path = None
    source_code = None
    script_scope = {}
    lineno_next_call_or_subscript = -1
    col_offset_next_call_or_subscript = -1
    end_lineno_next_call_or_subscript = -1
    end_col_offset_next_call_or_subscript = -1
    next_op_id = 0
    next_missing_op_id = -1
    track_code_references = True
    op_id_to_dag_node = dict()
    inspection_results = InspectionResult(networkx.DiGraph(), dict())
    inspections = []
    custom_monkey_patching = []

    def run(self, *,
            notebook_path: str or None = None,
            python_path: str or None = None,
            python_code: str or None = None,
            inspections: Iterable[Inspection] or None = None,
            checks: Iterable[Check] or None = None,
            reset_state: bool = True,
            track_code_references: bool = True,
            custom_monkey_patching: List[any] = None
            ) -> InspectorResult:
        """
        Instrument and execute the pipeline and evaluate all checks
        """
        # pylint: disable=too-many-arguments
        if reset_state:
            # reset_state=False should only be used internally for performance experiments etc!
            # It does not ensure the same inspections are still used as args etc.
            self.reset()

        if inspections is None:
            inspections = []
        if checks is None:
            checks = []
        if custom_monkey_patching is None:
            custom_monkey_patching = []

        check_inspections = set()
        for check in checks:
            check_inspections.update(check.required_inspections)
        all_inspections = list(set(inspections).union(check_inspections))
        self.inspections = all_inspections
        self.track_code_references = track_code_references
        self.custom_monkey_patching = custom_monkey_patching

        self.run_inspections(notebook_path, python_code, python_path)
        check_to_results = dict((check, check.evaluate(self.inspection_results)) for check in checks)
        return InspectorResult(self.inspection_results.dag, self.inspection_results.dag_node_to_inspection_results,
                               check_to_results)

    def run_inspections(self, notebook_path, python_code, python_path):
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use, too-many-locals
        self.source_code, self.source_code_path = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(self.source_code)
        parsed_modified_ast = self.instrument_pipeline(parsed_ast, self.track_code_references)
        exec(compile(parsed_modified_ast, filename=self.source_code_path, mode="exec"), PipelineExecutor.script_scope)

    def get_next_op_id(self):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        current_op_id = self.next_op_id
        self.next_op_id += 1
        return current_op_id

    def get_next_missing_op_id(self):
        """
        Each unknown operator in the DAG gets a consecutive unique negative id
        """
        current_missing_op_id = self.next_missing_op_id
        self.next_missing_op_id -= 1
        return current_missing_op_id

    def reset(self):
        """
        Reset all attributes in the singleton object. This can be used when there are multiple repeated calls to mlinspect
        """
        self.source_code_path = None
        self.source_code = None
        self.script_scope = {}
        self.lineno_next_call_or_subscript = -1
        self.col_offset_next_call_or_subscript = -1
        self.end_lineno_next_call_or_subscript = -1
        self.end_col_offset_next_call_or_subscript = -1
        self.next_op_id = 0
        self.next_missing_op_id = -1
        self.track_code_references = True
        self.op_id_to_dag_node = dict()
        self.inspection_results = InspectionResult(networkx.DiGraph(), dict())
        self.inspections = []
        self.custom_monkey_patching = []

    @staticmethod
    def instrument_pipeline(parsed_ast, track_code_references):
        """
        Instrument the pipeline AST to instrument function calls
        """
        # insert set_code_reference calls
        if track_code_references:
            # Needed to get the parent assign node for subscript assigns.
            #  Without this, "pandas_df['baz'] = baz + 1" would only be "pandas_df['baz']"
            parent_child_transformer = ParentChildNodeTransformer()
            parsed_ast = parent_child_transformer.visit(parsed_ast)
            call_capture_transformer = CallCaptureTransformer()
            parsed_ast = call_capture_transformer.visit(parsed_ast)
            parsed_ast = ast.fix_missing_locations(parsed_ast)

        # from mlinspect2._pipeline_executor import set_code_reference, monkey_patch
        func_import_node = ast.ImportFrom(module='mlinspect.instrumentation._pipeline_executor',
                                          names=[ast.alias(name='set_code_reference_call', asname=None),
                                                 ast.alias(name='set_code_reference_subscript', asname=None),
                                                 ast.alias(name='monkey_patch', asname=None),
                                                 ast.alias(name='undo_monkey_patch', asname=None)],
                                          level=0)
        parsed_ast.body.insert(0, func_import_node)

        # monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.insert(1, inspect_import_node)
        # undo_monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='undo_monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.append(inspect_import_node)

        parsed_ast = ast.fix_missing_locations(parsed_ast)

        return parsed_ast

    @staticmethod
    def load_source_code(notebook_path, python_path, python_code):
        """
        Load the pipeline source code from the specified source
        """
        sources = [notebook_path, python_path, python_code]
        assert sum(source is not None for source in sources) == 1
        if python_path is not None:
            with open(python_path) as file:
                source_code = file.read()
            source_code_path = python_path
        elif notebook_path is not None:
            with open(notebook_path) as file:
                notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
                exporter = PythonExporter()
                source_code, _ = exporter.from_notebook_node(notebook)
            source_code_path = notebook_path
        elif python_code is not None:
            source_code = python_code
            source_code_path = "<string-source>"
        else:
            assert False
        return source_code, source_code_path


# How we instrument the calls

# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
singleton = PipelineExecutor()


def set_code_reference_call(lineno, col_offset, end_lineno, end_col_offset, **kwargs):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return kwargs


def set_code_reference_subscript(lineno, col_offset, end_lineno, end_col_offset, arg):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return arg


def monkey_patch():
    """
    Function that does the actual monkey patching
    """
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.apply(patch)


def undo_monkey_patch():
    """
    Function that does the actual monkey patching
    """
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.revert(patch)


def get_monkey_patching_patch_sources():
    """
    Get monkey patches provided by mlinspect and custom patches provided by the user
    """
    patch_sources = [monkeypatching]
    patch_sources.extend(singleton.custom_monkey_patching)
    return patch_sources
