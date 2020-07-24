"""
User-facing API for inspecting the pipeline
"""
from .instrumentation.pipeline_executor import pipeline_executor


class PipelineInspectorBuilder:
    """
    The fluent API builder to build an inspection run
    """

    def __init__(self, notebook_path: str or None = None,
                 python_path: str or None = None,
                 python_code: str or None = None
                 ) -> None:
        self.notebook_path = notebook_path
        self.python_path = python_path
        self.python_code = python_code

    def add_analyzer(self, analyzer):
        """
        Add an analyzer
        """
        print(str(analyzer))
        return self

    def execute(self):
        """
        Instrument and execute the pipeline
        """
        return pipeline_executor.run(self.notebook_path, self.python_path, self.python_code)


class PipelineInspector:
    """
    The entry point to the fluent API to build an inspection run
    """
    @staticmethod
    def on_pipeline_from_py_file(path: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a .py file."""
        return PipelineInspectorBuilder(python_path=path)

    @staticmethod
    def on_pipeline_from_ipynb_file(path: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a .ipynb file."""
        return PipelineInspectorBuilder(notebook_path=path)

    @staticmethod
    def on_pipeline_from_string(code: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a string."""
        return PipelineInspectorBuilder(python_code=code)
