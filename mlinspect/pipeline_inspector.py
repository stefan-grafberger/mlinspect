"""
User-facing API for inspecting the pipeline
"""
from typing import Iterable

from .instrumentation.analyzers.analyzer import Analyzer
from .instrumentation.inspection_result import InspectionResult
from .instrumentation.pipeline_executor import singleton


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
        self.analyzers = []

    def add_analyzer(self, analyzer: Analyzer):
        """
        Add an analyzer
        """
        self.analyzers.append(analyzer)
        return self

    def add_analyzers(self, analyzers: Iterable[Analyzer]):
        """
        Add a list of analyzers
        """
        self.analyzers.extend(analyzers)
        return self

    def execute(self) -> InspectionResult:
        """
        Instrument and execute the pipeline
        """
        return singleton.run(self.notebook_path, self.python_path, self.python_code, self.analyzers)


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
