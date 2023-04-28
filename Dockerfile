ARG PYTHON_VERSION=3.9.16

FROM python:${PYTHON_VERSION}-slim-bullseye as base

RUN apt-get clean  \
  && apt-get update \
  #  psycopg2 dependencies
  && apt-get install -y libpq5=13.9-0+deb11u1 --no-install-recommends \
  && apt-get install -y libpq-dev=13.9-0+deb11u1 --no-install-recommends \
  # dependencies for building Python packages
  && apt-get install -y build-essential --no-install-recommends \
  # for visualization purposes
  && apt-get install -y graphviz  --no-install-recommends \
  && apt-get install -y graphviz-dev  --no-install-recommends \
  # cleaning up unused files
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PLATFORM_DIRS 1
ENV SETUPTOOLS_USE_DISTUTILS stdlib

# create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


FROM base as builder

# set work directory
WORKDIR /project

# install poetry
COPY poetry-requirements.txt poetry-requirements.txt
RUN pip install -r poetry-requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# install local (dev) dependencies
RUN poetry export -f requirements.txt --output requirements.txt --with dev --without-hashes
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY mlinspect/ mlinspect/
COPY README.md README.md
RUN poetry build
RUN pip install dist/*.whl

FROM builder as final

COPY  --from=builder /opt/venv /opt/venv

RUN jupyter --paths
RUN ipython kernel install --user --name="venv"

COPY example_pipelines/ example_pipelines/
COPY demo/ demo/
COPY test/ test/
COPY .pylintrc .pylintrc

ENV PYTHONPATH /project

EXPOSE 8888