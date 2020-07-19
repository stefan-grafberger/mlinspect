"""
Instrument and executes the pipeline
"""
import ast
import astunparse
# import astpretty
import nbformat
from nbconvert import PythonExporter


def print_and_exec(args, args_code, node, code):
    """
    This is the method we want to insert into the DAG
    """
    print(code)
    if node is not None:
        function = code.split("(", 1)[0]
        print(eval("inspect.getmodule(" + function + ")"))
    print(len(args))
    for arg_code in args_code:
        print(eval(arg_code))
    # print(node)
    # return exec(compile(node, "test", mode="exec"))
    # Maybe we also need to deal with subscripts
    # Here we could replace function calls or replace return types, e.g. subclassing pandas.Dataframe orr
    # sklearn pipeline functions that return special vectors
    return node


class MyTransformer(ast.NodeTransformer):
    """
    Inserts function call capturing into the DAG
    """

    def visit_call(self, node):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use
        print("test")
        if isinstance(node, ast.Call):
            code = astunparse.unparse(node)
            # name = node.func.id
            # astpretty.pprint(node)
            args = ast.List(node.args, ctx=ast.Load())
            args_code = ast.List([ast.Str(astunparse.unparse(arg).split("\n", 1)[0]) for arg in node.args],
                                 ctx=ast.Load())

            test = ast.Call(func=ast.Name(id='print_and_exec', ctx=ast.Load()),
                            args=[args, args_code, node, ast.Str(s=code)], keywords=[])
            # test = ast.Call(func=ast.Name(id='print_and_exec', ctx=ast.Load()), args=[ast.Str(s=code)],keywords=[])
            # test = ast.Expr(value=ast.Call(id="print", func="print", args=[ast.Str(s='Hello World')], keywords=[]))
            ast.copy_location(test, node)

            # to deal with both projections and joins as black-box operators, we need different
            # storage formats. (alternatively, we could overwrite operators, but it would be better to avoid this)
            # we want to store the annotations separately. but for joins and select, we need to add them to the
            # dataframe, to discover input/output mappings quickly. for projects, nothing changes.
            # in case of selects, deduplicates and joins no changes occur in input/output.
            # because of this, input is the same as output if we remember which column belongs to which table
            # only for extended projects do we need to scan the input again. to do this efficiently,
            # we can add the old input as column to the table and remove it afterwards.
            # basically, we want to add all information before the operation to the dataframe itself, and then
            # delete it again afterwards and move it to a separate storage format. we do not want to have
            # two dataframes in-memory at the same time.

            # do we need to ensure that function calls that appear in parameters are visited again?
            # example: print(income_pipeline.predict(data)): income_pipeline.predict(data)
            # probably yes!

            # todo: warn if unrecognized function call, if defined within same file, maybe inline this in some way?
            return test
        return node


def run(notebook_path: str or None, python_path: str or None):
    """
    Instrument and execute the pipeline
    """
    source_code = ""
    print("test")
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
    transformer = MyTransformer()
    parsed_ast = transformer.visit(parsed_ast)
    # FuncLister().visit(parsed_ast)
    # print(ast.dump(parsed_ast))
    parsed_ast = ast.fix_missing_locations(parsed_ast)
    # astpretty.pprint(parsed_ast)
    exec(compile(parsed_ast, filename="<ast>", mode="exec"))
    return "test"


class FuncLister(ast.NodeVisitor):
    """
    NodeVisitor that lists function calls
    """

    def generic_visit(self, node):
        """
        Visit and analyze DAG nodes
        """
        # print(type(node).__name__)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                print(node.func.attr)  # we need to instrument call and determine module at runtime
            if isinstance(node.func, ast.Name):
                print(node.func.id)
        ast.NodeVisitor.generic_visit(self, node)
