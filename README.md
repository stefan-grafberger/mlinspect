# MLInspect
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
    
## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
