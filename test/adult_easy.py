import unittest
import ast
import nbformat
from nbconvert import PythonExporter

class MyTestCase(unittest.TestCase):
    file_py = 'pipelines/adult_easy.py'
    file_nb = 'pipelines/adult_easy.ipynb'

    def test_py_pipeline_runs(self):
        with open(self.file_py) as file:  # Use file to refer to the file object
            text = file.read()
            parsed_ast = ast.parse(text)
            exec(compile(parsed_ast, filename="<ast>", mode="exec"))

    def test_nb_pipeline_runs(self):
        with open(self.file_nb) as file:  # Use file to refer to the file object
            nb = nbformat.reads(file.read(), nbformat.NO_CONVERT)
            exporter = PythonExporter()

            # source is a tuple of python source code
            # meta contains metadata
            source, meta = exporter.from_notebook_node(nb)
            parsed_ast = ast.parse(source)
            exec(compile(parsed_ast, filename="<ast>", mode="exec"))

if __name__ == '__main__':
    unittest.main()
