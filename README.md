mlinspect
===

[![mlinspect](https://img.shields.io/badge/🔎-mlinspect-green)](https://github.com/stefan-grafberger/MLInspect)
[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://github.com/stefan-grafberger/MLInspect/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/stefan-grafberger/mlinspect.svg?token=x1zHsibRoiV8cZwxNVsj&branch=master)](https://travis-ci.com/stefan-grafberger/MLInspect)
[![codecov](https://codecov.io/gh/stefan-grafberger/MLInspect/branch/master/graph/badge.svg?token=KTMNPBV1ZZ)](https://codecov.io/gh/stefan-grafberger/MLInspect)

Inspect ML Pipelines in Python in the form of a DAG

Run mlinspect locally
---

Prerequisite: Python 3.8

1. Clone this repository
2. Set up the environment

	cd mlinspect
	python -m venv venv
	source venv/bin/activate

3. If you want to use the visualisation functions we provide, install graphviz which can not be installed via pip

	# Linux
	apt-get install graphviz libgraphviz-dev
	# MAC OS
	brew install graphviz
	
4. Install pip dependencies 

	pip install -e .

5. To ensure everything works, you can run the tests (without graphviz, the visualisation test will fail)

	python setup.py test

### Docker Development Environment

	docker build . -t mlinspect

	# Run with DEBUG mode enabled and local code changes available
	docker run -ti --rm --name mlinspect-demo -v $PWD:/mlinspect -w /mlinspect -p 8050:8050 -e DEBUG=1 mlinspect

	# Run with DEBUG mode disabled and local code changes unavailable
	docker run -ti --rm --name mlinspect-demo -p 8050:8050 mlinspect

Vision
---

Make it easy to analyze your pipeline and automatically check for common issues.

```python
from mlinspect import PipelineInspector
from mlinspect.inspections import MaterializeFirstOutputRows
from mlinspect.checks import NoBiasIntroducedFor

IPYNB_PATH = ...

inspector_result = PipelineInspector\
		.on_pipeline_from_ipynb_file(IPYNB_PATH)\
		.add_required_inspection(MaterializeFirstOutputRows(5))\
		.add_check(NoBiasIntroducedFor(['race']))\
		.execute()

extracted_dag = inspector_result.dag
inspection_to_annotations = inspector_result.inspection_to_annotations
check_to_check_results = inspector_result.check_to_check_results
```

Detailed Example
---

We prepared a [demo notebook](demo/feature_overview/feature_overview.ipynb) to showcase mlinspect and its features.

Notes
---

* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
* This is a research project, so comprehensive coverage of all possible ML APIs will not be possible in the current initial step. We will try to tell you if we encounter APIs we can not handle yet.

License
---

This library is licensed under the Apache 2.0 License.
