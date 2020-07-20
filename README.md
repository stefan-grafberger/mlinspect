MLInspec
================================

[![GitHub license](https://img.shields.io/github/license/stefan-grafberger/mlinspect.svg)](https://github.com/stefan-grafberger/MLInspect/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/stefan-grafberger/mlinspect.svg)](https://github.com/stefan-grafberger/MLInspect/issues)
[![Build Status](https://travis-ci.com/stefan-grafberger/mlinspect.svg?token=x1zHsibRoiV8cZwxNVsj&branch=master)](https://travis-ci.com/stefan-grafberger/MLInspect)
[![codecov](https://codecov.io/gh/stefan-grafberger/MLInspect/branch/master/graph/badge.svg?token=KTMNPBV1ZZ)](https://codecov.io/gh/stefan-grafberger/MLInspect)

Inspect ML Pipelines in Python in the form of a DAG

## Run MLInspect locally

Prerequisite: python >=  3.7

1. Clone this repository
2. Set up the environment

	`cd MLInspect` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>
	
3. Install dependencies 

    `python setup.py develop` <br>

3. Run the tests

    `python setup.py test` <br>
    
## Vision
Make it easy to analyze your pipeline and automatically check for common issues.
```
from mlinspect.pipeline_inspector import PipelineInspector

IPYNB_PATH = ...

extracted_annotated_dag = PipelineInspector\
        .on_jupyter_pipeline(IPYNB_PATH)\
        .add_analyzer("test")\
        .execute()
```
    
## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
* This is a research project, so comprehensive coverage of all possible ML APIs will not be possible in the current initial step. We will try to tell you if we encounter APIs we can not handle yet.
