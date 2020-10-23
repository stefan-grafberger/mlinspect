"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Tuple, OrderedDict

import networkx

from mlinspect.checks._check import Check, CheckResult
from mlinspect.inspections._inspection import Inspection


@dataclasses.dataclass
class InspectorResult:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    inspection_to_annotations: OrderedDict[Inspection, OrderedDict[Tuple[int, int], any]]
    check_to_check_results: OrderedDict[Check, CheckResult]
