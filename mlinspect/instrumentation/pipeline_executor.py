"""
Instrument and executes the pipeline
"""
import ast

import astunparse
import astpretty  # pylint: disable=unused-import
import nbformat
from nbconvert import PythonExporter
from .call_capture_transformer import CallCaptureTransformer
from .wir_extractor import WirExtractor


def instrumented_call_used(arg_values, args_code, node, code):
    """
    Method that gets injected into the pipeline code
    """
    return PipelineExecutor.instrumented_call_used(arg_values, args_code, node, code)


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """

    # This is a bit ugly currently: we avoid to have to pass the class instance to the instrumented
    # pipeline. This is a simple workaround for that. This keeps the DAG nodes to be inserted very simple.
    script_scope = {}

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

        initial_wir = WirExtractor().extract_wir(parsed_ast)
        print(initial_wir)

        parsed_ast = CallCaptureTransformer().visit(parsed_ast)
        parsed_ast = ast.fix_missing_locations(parsed_ast)

        func_import_node = ast.ImportFrom(module='mlinspect.instrumentation.pipeline_executor',
                                          names=[ast.alias(name='instrumented_call_used',
                                                           asname=None)],
                                          level=0)
        parsed_ast.body.insert(2, func_import_node)
        inspect_import_node = ast.Import(names=[ast.alias(name='inspect', asname=None)])
        parsed_ast.body.insert(3, inspect_import_node)
        parsed_ast = ast.fix_missing_locations(parsed_ast)

        # self.output_parsed_ast(parsed_ast)

        exec(compile(parsed_ast, filename="<ast>", mode="exec"), PipelineExecutor.script_scope)

        return "test"

    @staticmethod
    def instrumented_call_used(arg_values, args_code, node, code):
        """
        This is the method we want to insert into the DAG
        """
        print(code)
        if node is not None:
            function = code.split("(", 1)[0]
            print(eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope))
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
