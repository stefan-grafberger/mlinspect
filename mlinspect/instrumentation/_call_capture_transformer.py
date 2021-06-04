"""
Inserts function call capturing into the DAG
"""
import ast


class CallCaptureTransformer(ast.NodeTransformer):
    """
    ast.NodeTransformer to replace calls with captured calls
    """

    def visit_Call(self, node):
        """
        Instrument all function calls
        """
        # pylint: disable=invalid-name
        ast.NodeTransformer.generic_visit(self, node)
        self.call_add_set_code_reference(node)
        return node

    def visit_Subscript(self, node):
        """
        Instrument all subscript calls
        """
        # pylint: disable=invalid-name
        ast.NodeTransformer.generic_visit(self, node)
        if isinstance(node.ctx, ast.Store):
            # Needed to get the parent assign node for subscript assigns.
            #  Without this, "pandas_df['baz'] = baz + 1" would only be "pandas_df['baz']"
            code_reference_from_node = node.parents[0]
        else:
            code_reference_from_node = node
        self.subscript_add_set_code_reference(node, code_reference_from_node)
        return node

    @staticmethod
    def call_add_set_code_reference(node):
        """
        When a method is called, capture the arguments of the method before executing it
        """
        # We need to use a keyword argument call to capture stuff because the eval order and because
        #  keyword arguments may contain function calls.
        #  https://stackoverflow.com/questions/17948369/is-it-safe-to-rely-on-python-function-arguments-evaluation-order
        # Here we can consider instrumenting only functions we patch based on the name
        #  But the detection based on the static function name is unreliable, so we will skip this for now
        kwargs = node.keywords
        call_node = CallCaptureTransformer.create_set_code_reference_node_call(node, kwargs)
        new_kwargs_node = ast.keyword(value=call_node, arg=None)
        node.keywords = [new_kwargs_node]

    @staticmethod
    def create_set_code_reference_node_call(node, kwargs):
        """
        Create the set_code_reference function call ast node that then gets inserted into the AST
        """
        call_node = ast.Call(func=ast.Name(id='set_code_reference_call', ctx=ast.Load()),
                             args=[ast.Constant(n=node.lineno, kind=None),
                                   ast.Constant(n=node.col_offset, kind=None),
                                   ast.Constant(n=node.end_lineno, kind=None),
                                   ast.Constant(n=node.end_col_offset, kind=None)],
                             keywords=kwargs)
        return call_node

    @staticmethod
    def create_set_code_reference_node_subscript(node, kwargs):
        """
        Create the set_code_reference function call ast node that then gets inserted into the AST
        """
        call_node = ast.Call(func=ast.Name(id='set_code_reference_subscript', ctx=ast.Load()),
                             args=[ast.Constant(n=node.lineno, kind=None),
                                   ast.Constant(n=node.col_offset, kind=None),
                                   ast.Constant(n=node.end_lineno, kind=None),
                                   ast.Constant(n=node.end_col_offset, kind=None),
                                   kwargs],
                             keywords=[])
        return call_node

    @staticmethod
    def subscript_add_set_code_reference(node, code_reference_from_node):
        """
        When the __getitem__ method of some object is called, capture the arguments of the method before executing it
        """
        subscript_arg = node.slice
        call_node = CallCaptureTransformer.create_set_code_reference_node_subscript(code_reference_from_node,
                                                                                    subscript_arg)
        node.slice = call_node
