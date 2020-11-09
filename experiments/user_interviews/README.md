# User Study

## Building and uploading the Docker Container
* Run `docker build . -t mlinspect` in this directory
* `docker tag mlinspect stefangrafberger/mlinspect`
* `docker login`
* `docker push`

## Running the Docker Container
* `docker run -t --rm -p 8899:8899 stefangrafberger/mlinspect`

## Links to the task files
* `http://localhost:8899/notebooks/experiments/user_interviews/example-task-with-solution.ipynb`
* `http://localhost:8899/notebooks/experiments/user_interviews/task-1.ipynb`
* `http://localhost:8899/notebooks/experiments/user_interviews/task-2.ipynb`
