"""
Instrument and executes the pipeline
"""
import ast
import copy
import astunparse
import astpretty  # pylint: disable=unused-import
import nbformat
from nbconvert import PythonExporter
from .call_capture_transformer import CallCaptureTransformer
from .wir_extractor import WirExtractor


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    script_scope = {}

    def __init__(self):
        self.ast_call_node_id_to_module = {}

    def run(self, notebook_path: str or None, python_path: str or None):
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use
        PipelineExecutor.script_scope = {}

        source_code = ""
        assert (notebook_path is None or python_path is None)
        if python_path is not None:
            with open(python_path) as file:
                source_code = file.read()
        if notebook_path is not None:
            with open(notebook_path) as file:
                notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
                exporter = PythonExporter()
                source_code, _ = exporter.from_notebook_node(notebook)

        parsed_ast = ast.parse(source_code)
        parsed_ast_before_modifications = copy.deepcopy(parsed_ast)

        call_capture_transformer = CallCaptureTransformer()
        parsed_modified_ast = call_capture_transformer.visit(parsed_ast)
        parsed_modified_ast = ast.fix_missing_locations(parsed_modified_ast)

        func_import_node = ast.ImportFrom(module='mlinspect.instrumentation.pipeline_executor',
                                          names=[ast.alias(name='instrumented_call_used',
                                                           asname=None)],
                                          level=0)
        parsed_modified_ast.body.insert(2, func_import_node)
        inspect_import_node = ast.Import(names=[ast.alias(name='inspect', asname=None)])
        parsed_modified_ast.body.insert(3, inspect_import_node)
        parsed_modified_ast = ast.fix_missing_locations(parsed_modified_ast)

        exec(compile(parsed_modified_ast, filename="<ast>", mode="exec"), PipelineExecutor.script_scope)

        initial_wir = self.extract_wir(call_capture_transformer, parsed_ast_before_modifications)
        print(initial_wir)

        return "test"

    def extract_wir(self, call_capture_transformer, parsed_ast):
        """
        Gets the WIR using the WirExtractor
        """
        ast_calls_to_module = {}
        id_to_call_ast = call_capture_transformer.get_id_to_call_ast()
        for ast_call_id, module in self.ast_call_node_id_to_module.items():
            ast_node = id_to_call_ast[ast_call_id]
            ast_calls_to_module[ast_node] = module
        initial_wir = WirExtractor(parsed_ast).extract_wir(ast_calls_to_module)
        return initial_wir

    def instrumented_call_used(self, arg_values, args_code, node, code, ast_node_id):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        print(code)

        function = code.split("(", 1)[0]
        module = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)
        self.ast_call_node_id_to_module[ast_node_id] = module

        print(len(arg_values))
        for arg_code in args_code:
            print(arg_code)
        print(node)

        return node

    @staticmethod
    def output_parsed_ast(parsed_ast):
        """
        Output the unparsed Dag, print the DAG and generate an image of it
        """
        astunparse.unparse(parsed_ast)  # TODO: Remove this
        astpretty.pprint(parsed_ast)  # TODO: Remove this


# How we instrument the calls

# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
pipeline_executor = PipelineExecutor()


def instrumented_call_used(arg_values, args_code, node, code, ast_node_id):
    """
    Method that gets injected into the pipeline code
    """
    return pipeline_executor.instrumented_call_used(arg_values, args_code, node, code, ast_node_id)
