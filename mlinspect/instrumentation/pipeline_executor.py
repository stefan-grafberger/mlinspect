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


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    script_scope = {}

    def __init__(self):
        self.ast_call_node_id_to_module = {}

    def run(self, notebook_path: str or None, python_path: str or None, python_code: str or None):
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use
        PipelineExecutor.script_scope = {}

        source_code = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(source_code)

        parsed_modified_ast = self.instrument_pipeline(parsed_ast)

        exec(compile(parsed_modified_ast, filename="<ast>", mode="exec"), PipelineExecutor.script_scope)

        initial_wir = WirExtractor(parsed_ast).extract_wir(self.ast_call_node_id_to_module)
        print(initial_wir)

        return "test"

    @staticmethod
    def instrument_pipeline(parsed_ast):
        """
        Instrument the pipeline AST to instrument function calls
        """
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

    def instrumented_call_used(self, subscript, arg_values, args_code, node, code, ast_lineno, ast_col_offset):
        """
        This is the method we want to insert into the DAG
        """
        # pylint: disable=too-many-arguments
        if not subscript:
            print(code)

            function = str(code.split("(", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, str(function.split(".")[-1]))
            self.ast_call_node_id_to_module[(ast_lineno, ast_col_offset)] = function_info

            print(len(arg_values))
            for arg_code in args_code:
                print(arg_code)
        else:
            # pylint: disable=too-many-arguments
            print(code)

            function = str(code.split("[", 1)[0])
            module_info = eval("inspect.getmodule(" + function + ")", PipelineExecutor.script_scope)

            function_info = (module_info.__name__, "__getitem__")
            self.ast_call_node_id_to_module[(ast_lineno, ast_col_offset)] = function_info

            print(len(arg_values))
            print("caller: {}, key: {}".format(args_code[0], args_code[1]))

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
singleton = PipelineExecutor()


def instrumented_call_used(subscript, arg_values, args_code, node, code, ast_lineno, ast_col_offset):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    return singleton.instrumented_call_used(subscript, arg_values, args_code, node, code, ast_lineno, ast_col_offset)
