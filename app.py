from ast import literal_eval
import numpy as np
import os
import time

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from inspect import cleandoc

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

import plotly.graph_objects as go
import pygraphviz as pgv

from mlinspect import PipelineInspector
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows
from mlinspect.visualisation import save_fig_to_path


# Initialize Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=[
    # Dash CSS
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    # Loading screen CSS
    "https://codepen.io/chriddyp/pen/brPBPO.css",
    # Bootstrap theme CSS
    dbc.themes.BOOTSTRAP,  # pro: CSS classes; con: tiny font size
    # dbc.themes.GRID,  # pro: grid layouts, large enough font size; con: no other dbc elements or CSS classes
])
app.config.suppress_callback_exceptions = True


# Create HTML layout
CODE_FONT = {"font-family": "'Courier New', monospace"}
app.title = "mlinspect"
app.layout = dbc.Container([
    # Header and description
    html.H1("mlinspect", style=CODE_FONT),
    html.P("Inspect ML Pipelines in Python in the form of a DAG."),

    # Body
    dbc.Row([  # Grid only works with dbc.themes in external_stylesheets.
        dbc.Col([
            dbc.Form([
                dbc.FormGroup([
                    # Pipeline definition
                    dbc.Label("Pipeline definition:", html_for="pipeline"),
                    dbc.Textarea(id="pipeline", className="mb-3", style={"width": "450px", "height": "500px", **CODE_FONT}),
                ]),
                dbc.FormGroup([
                    # Add checks
                    dbc.Label("Add checks:", html_for="checks"),
                    dbc.Checklist(
                        id="checks",
                        options=[
                            {"label": "No Bias Introduced For", "value": "NoBiasIntroducedFor"},
                            {"label": "No Illegal Features", "value": "NoIllegalFeatures"},
                        ],
                        switch=True,
                        value=[],
                    ),
                ]),
                dbc.FormGroup([
                    # Add inspections
                    dbc.Label("Add required inspections:", html_for="inspections"),
                    dbc.Checklist(
                        id="inspections",
                        options=[
                            {"label": "Histogram For Columns", "value": "HistogramForColumns"},
                            {"label": "Row Lineage", "value": "RowLineage"},
                            {"label": "Materialize First Output Rows", "value": "MaterializeFirstOutputRows"},
                        ],
                        switch=True,
                        value=[],
                    ),
                ]),
            ]),
            # Execute inspection
            dbc.Button("Inspect pipeline", id="execute", color="primary", size="lg", className="mr-1"),
        ], width=6),
        dbc.Col([
            # Display DAG
            dbc.Label("Extracted DAG:", html_for="dag"),
            dcc.Graph(id="dag"),
            # html.Img(id="dag", hidden=True, className="mb-3", style={"width": "450px"}),
        ], width=6),
    ]),
])

# Flask server (for gunicorn)
server = app.server


@app.callback(
    # [
        Output("dag", "figure"),
        # Output("dag", "src"),
        # Output("dag", "hidden"),
    # ],
    [
        Input("execute", "n_clicks"),
    ],
    state=[
        State("pipeline", "value"),
        State("checks", "value"),
        State("inspections", "value"),
    ])
def update_figure(n_clicks, pipeline, checks, inspections):
    """Dash callback function to show extracted DAG of ML pipeline."""
    if n_clicks is None:
        return dash.no_update

    digraph, _, _ = extract_dag(pipeline, checks, inspections)

    return nx2go(digraph)


def extract_dag(pipeline, checks=None, inspections=None):
    """Extract DAG the original way, i.e. by creating a PipelineInspectorBuilder."""
    start = time.time()
    builder = PipelineInspector.on_pipeline_from_string(pipeline)
    if checks:
        builder = builder.add_checks(checks)
    if inspections:
        builder = builder.add_required_inspections(inspections)
    inspector_result = builder.execute()
    print(f"Total time in seconds: {time.time() - start}")

    extracted_dag = inspector_result.dag
    inspection_results = inspector_result.inspection_to_annotations
    check_results = inspector_result.check_to_check_results
    return extracted_dag, inspection_results, check_results


def nx2png(extracted_dag):
    """Convert networkx.DiGraph to a pygraphviz.agraph.AGraph, save to file, and return filename.
    Also return boolean of whether HTML Image element is hidden."""
    filename = os.path.join(os.getcwd(), app.get_asset_url("image.png"))
    save_fig_to_path(extracted_dag, filename)
    return filename, False


def nx2agraph(extracted_dag):
    """Convert networkx.DiGraph to a pygraphviz.agraph.AGraph."""
    extracted_dag = nx.relabel_nodes(extracted_dag, lambda node: cleandoc("""
        {} (L{})
        {}
        """.format(node.operator_type.value, node.code_reference.lineno, node.description or "")))
    agraph = to_agraph(extracted_dag)
    agraph.layout('dot')
    return agraph


def _plotly_graph(E, pos):
    """
    E is the list of tuples representing the graph edges.
    pos is the list of node coordinates.

    Source: https://chart-studio.plotly.com/~empet/14007/graphviz-networks-plotted-with-plotly/#/
    """
    N = len(pos)
    Xn = [pos[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [pos[k][1] for k in range(N)]  # y-coordnates of nodes

    Xe = []
    Ye = []
    for e0, e1 in E:
        Xe += [pos[e0][0],pos[e1][0], None]  # x coordinates of the nodes defining the edge e
        Ye += [pos[e0][1],pos[e1][1], None]  # y - " -

    return Xn, Yn, Xe, Ye


def _set_annotation(x, y, anno_text,  textangle, fontsize=11, color='rgb(10,10,10)'): 
    """
    Source: https://chart-studio.plotly.com/~empet/14007/graphviz-networks-plotted-with-plotly/#/
    """
    return dict(x=x,
                y=y,
                text=anno_text,
                textangle=textangle,  # angle with horizontal line through (x,y), in degrees;
                                      # + =clockwise, -=anti-clockwise
                font={'size': fontsize, 'color': color},
                showarrow=False)


def _get_pos(G):
    pos_dict = nx.nx_agraph.graphviz_layout(G, 'dot')

    # Define the tree  as a networkx graph
    V = G.nodes()
    E = G.edges()

    labels = [f"{node.operator_type.value} (L{node.code_reference.lineno})\n{node.description}" for node in pos_dict.keys()]

    Xn = []
    Yn = []
    for v in V:
        x, y = pos_dict[v]
        Xn += [x]
        Yn += [y]

    Xe = []
    Ye = []
    for e0, e1 in E:
        x0, y0 = pos_dict[e0]
        x1, y1 = pos_dict[e1]
        Xe += [x0, x1]
        Ye += [y0, y1]

    return Xn, Yn, Xe, Ye, labels


def nx2go(G):
    """
    Convert networkx.DiGraph to a plotly.graph_objects.Figure.

    Source: https://chart-studio.plotly.com/~empet/14007/graphviz-networks-plotted-with-plotly/#/
    """
    # pos_dict = nx.nx_agraph.graphviz_layout(G, 'dot')

    # # Define the tree  as a networkx graph
    # V = G.nodes()
    # E = G.edges()

    # # pygraphviz tree H and its layout
    # H = pgv.AGraph(strict=True, directed=False)
    # H.add_nodes_from(V)
    # H.add_edges_from(E)
    # H.layout(prog='dot')
    # # H.layout(prog='twopi')

    # # Get node positions in the tree H:
    # # pos = np.array([literal_eval(n.attr['pos']) for n in V])
    # # pos = np.array([literal_eval(H.get_node(k).attr['pos']) for  k in range(N)])
    # # Rotate node positions with pi/2 counter-clockwise
    # # pos[:, [0, 1]] = pos[:, [1, 0]]
    # # pos[:, 0] =- pos[:,0]

    # pos_values = list(pos_dict.values())
    # labels = [f"{k.operator_type.value} (L{k.code_reference.lineno})\n{k.description}" for k in pos_dict.keys()]

    # # Define the Plotly objects that represent the tree
    # Xn, Yn, Xe, Ye = _plotly_graph(E, pos_values)
    Xn, Yn, Xe, Ye, labels = _get_pos(G)

    edges = go.Scatter(x=Xe, y=Ye, mode='lines', hoverinfo='none',
                       line={'color': 'rgb(160,160,160)', 'width': 0.75})
    nodes = go.Scatter(x=Xn, y=Yn, mode='markers', name='', hoverinfo='text', text=labels,
                       marker={
                           'size': 8,
                           'color': '#85b6b6',
                           'line': {
                               'color': 'rgb(100,100,100)',
                               'width': 0.5,
                            },
                        })
    layout = go.Layout(
                title="Pipeline execution DAG",
                font={'family': 'Balto'},
                width=650,
                height=650,
                showlegend=False,
                xaxis={'visible': False},
                yaxis={'visible': False},
                margin={'t': 100},
                hovermode='closest')
    # layout.annotations = annotations

    fig = go.Figure(data=[edges, nodes], layout=layout)

    return fig


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
