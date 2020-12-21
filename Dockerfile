FROM python:3.8

RUN apt-get update -y \
&& apt-get install -y graphviz libgraphviz-dev

ENV TF_CPP_MIN_LOG_LEVEL="3"

COPY . /mlinspect
WORKDIR "/mlinspect"

RUN pip install -e .[dev] dash dash-bootstrap-components

EXPOSE 8050
ENTRYPOINT [ "python", "app.py" ]
