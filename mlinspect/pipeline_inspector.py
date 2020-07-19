"""
User-facing API for inspecting the pipeline
"""
from mlinspect.instrumentation.pipeline_executor import PipelineExecutor


class PipelineInspectorBuilder:
    """
    The fluent API builder to build an inspection run
    """

    def __init__(self, notebook_path: str or None, python_path: str or None) -> None:
        self.notebook_path = notebook_path
        self.python_path = python_path

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
        return PipelineExecutor().run(self.notebook_path, self.python_path)


class PipelineInspector:
    """
    The entry point to the fluent API to build an inspection run
    """
    @staticmethod
    def on_python_pipeline(path: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a .py file."""
        return PipelineInspectorBuilder(None, path)

    @staticmethod
    def on_jupyter_pipeline(path: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a .ipynb file."""
        return PipelineInspectorBuilder(path, None)
