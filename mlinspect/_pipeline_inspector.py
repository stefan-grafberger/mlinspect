"""
User-facing API for inspecting the pipeline
"""
from typing import Iterable, Dict, List

from pandas import DataFrame

from mlinspect.inspections._inspection import Inspection
from .checks._check import Check, CheckResult
from ._inspector_result import InspectorResult
from .instrumentation._pipeline_executor import singleton


class PipelineInspectorBuilder:
    """
    The fluent API builder to build an inspection run
    """

    def __init__(self, notebook_path: str or None = None,
                 python_path: str or None = None,
                 python_code: str or None = None
                 ) -> None:
        self.track_code_references = True
        self.monkey_patching_modules = []
        self.notebook_path = notebook_path
        self.python_path = python_path
        self.python_code = python_code
        self.inspections = []
        self.checks = []

    def add_required_inspection(self, inspection: Inspection):
        """
        Add an analyzer
        """
        self.inspections.append(inspection)
        return self

    def add_required_inspections(self, inspections: Iterable[Inspection]):
        """
        Add a list of inspections
        """
        self.inspections.extend(inspections)
        return self

    def add_check(self, check: Check):
        """
        Add an analyzer
        """
        self.checks.append(check)
        return self

    def add_checks(self, checks: Iterable[Check]):
        """
        Add a list of inspections
        """
        self.checks.extend(checks)
        return self

    def set_code_reference_tracking(self, track_code_references: bool):
        """
        Set whether to track code references. The default is tracking them.
        """
        self.track_code_references = track_code_references
        return self

    def add_custom_monkey_patching_modules(self, module_list: List):
        """
        Add additional monkey patching modules.
        """
        self.monkey_patching_modules.extend(module_list)
        return self

    def add_custom_monkey_patching_module(self, module: any):
        """
        Add an additional monkey patching module.
        """
        self.monkey_patching_modules.append(module)
        return self

    def execute(self) -> InspectorResult:
        """
        Instrument and execute the pipeline
        """
        return singleton.run(notebook_path=self.notebook_path,
                             python_path=self.python_path,
                             python_code=self.python_code,
                             inspections=self.inspections,
                             checks=self.checks,
                             custom_monkey_patching=self.monkey_patching_modules)


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

    @staticmethod
    def check_results_as_data_frame(check_to_check_results: Dict[Check, CheckResult]) -> DataFrame:
        """
        Get a pandas DataFrame with an overview of the CheckResults
        """
        check_names = []
        status = []
        descriptions = []
        for check_result in check_to_check_results.values():
            check_names.append(check_result.check)
            status.append(check_result.status)
            descriptions.append(check_result.description)
        return DataFrame(zip(check_names, status, descriptions), columns=["check_name", "status", "description"])
