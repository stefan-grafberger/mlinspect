FROM python:3.9

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config

RUN git clone https://github.com/stefan-grafberger/mlinspect.git
WORKDIR "/mlinspect"

RUN pip install -e .[dev]

CMD ["jupyter", "notebook", "--port", "8899", "--allow-root", "--ip", "0.0.0.0"]
