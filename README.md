mlinspect
================================

[![mlinspect](https://img.shields.io/badge/🔎-mlinspect-green)](https://github.com/stefan-grafberger/MLInspect)
[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://github.com/stefan-grafberger/MLInspect/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/stefan-grafberger/mlinspect.svg?token=x1zHsibRoiV8cZwxNVsj&branch=master)](https://travis-ci.com/stefan-grafberger/MLInspect)
[![codecov](https://codecov.io/gh/stefan-grafberger/MLInspect/branch/master/graph/badge.svg?token=KTMNPBV1ZZ)](https://codecov.io/gh/stefan-grafberger/MLInspect)

Inspect ML Pipelines in Python in the form of a DAG

## Run mlinspect locally

Prerequisite: python >=  3.8

1. Clone this repository
2. Set up the environment

	`cd mlinspect` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>
	
3. Install pip dependencies 

    `pip install -e .[dev]` <br>
    
4. If you want to use the visualisation functions we provide, install graphviz which can not be installed via pip

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>

5. Run the tests

    `python setup.py test` <br>
    
## Vision
Make it easy to analyze your pipeline and automatically check for common issues.
```python
from mlinspect.pipeline_inspector import PipelineInspector
from mlinspect.inspections.materialize_first_rows_inspection import MaterializeFirstRowsInspection

IPYNB_PATH = ...

inspection_result = PipelineInspector \
        .on_pipeline_from_ipynb_file(IPYNB_PATH) \
        .add_inspection(MaterializeFirstRowsInspection(2)) \
        .execute()

extracted_dag = inspection_result.dag
analyzer_results = inspection_result.analyzer_to_annotations
```
    
## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
* This is a research project, so comprehensive coverage of all possible ML APIs will not be possible in the current initial step. We will try to tell you if we encounter APIs we can not handle yet.

## License
This library is licensed under the Apache 2.0 License.
