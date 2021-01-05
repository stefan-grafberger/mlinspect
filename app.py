from ast import literal_eval
import json
import numpy as np
import os
import time

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from inspect import cleandoc

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

import plotly.graph_objects as go
import pygraphviz as pgv

from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
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
with open("example_pipelines/healthcare/healthcare.py") as f:
    default_pipeline = f.read()
sensitive_columns = ["age_group", "race"]
inspection_switcher = {
    "HistogramForColumns": lambda: HistogramForColumns(sensitive_columns),
    "RowLineage": lambda: RowLineage(5),
    "MaterializeFirstOutputRows": lambda: MaterializeFirstOutputRows(5),
}
check_switcher = {
    "NoBiasIntroducedFor": lambda: NoBiasIntroducedFor(sensitive_columns),
    "NoIllegalFeatures": NoIllegalFeatures,
    "NoMissingEmbeddings": NoMissingEmbeddings,
}
app.layout = dbc.Container([
    # Header and description
    html.H1("mlinspect", style={"font-size": "24px", **CODE_FONT}),
    html.P("Inspect ML Pipelines in Python in the form of a DAG."),

    dcc.Tabs([
        # Inspection definition tab
        dcc.Tab([
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            # Pipeline definition
                            dbc.Label("Pipeline definition:", html_for="pipeline"),
                            dbc.Textarea(
                                id="pipeline",
                                className="mb-3",
                                style={"width": "700px", "height": "500px", **CODE_FONT},
                                value=default_pipeline,
                            ),
                        ]),
                    ]),
                    dbc.Col([
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
                        dbc.FormGroup([
                            # Add checks
                            dbc.Label("Add checks:", html_for="checks"),
                            dbc.Checklist(
                                id="checks",
                                options=[
                                    {"label": "No Bias Introduced For", "value": "NoBiasIntroducedFor"},
                                    {"label": "No Illegal Features", "value": "NoIllegalFeatures"},
                                    {"label": "No Missing Embeddings", "value": "NoMissingEmbeddings"},
                                ],
                                switch=True,
                                value=[],
                            ),
                        ]),
                        # Execute inspection
                        dbc.Button("Inspect pipeline", id="execute", color="primary", size="lg", className="mr-1"),
                    ]),
                ]),
            ], style={"margin": "auto", "padding": "20px"}),
        ], label="INSPECTION DEFINITION", value="definition-tab"),

        # Inspection results tab
        dcc.Tab([
            dbc.Row([
                dbc.Col([
                    # Display code
                    html.P(
                        id="pipeline-output",
                        className="mb-3",
                        style={**CODE_FONT, "font-size": "12px", "white-space": "pre-line"},
                        children=default_pipeline,
                    ),
                ], width=6),
                dbc.Col([
                    # Display DAG
                    dcc.Graph(id="dag", figure=go.Figure(
                        # layout_width=650,
                        layout_height=650,
                        layout_showlegend=False,
                        layout_xaxis={'visible': False},
                        layout_yaxis={'visible': False},
                    )),
                    html.Div(id="results-detail"),
                ], width=6),
            ], style={"margin": "auto", "padding": "20px", "font-size": "12px"}),
        ], label="INSPECTION RESULTS", value="results-tab"),
    ], id="tabs", style={"display": "none"}),
], style={"font-size": "14px"})


# Flask server (for gunicorn)
server = app.server


@app.callback(
    Output("pipeline-output", "children"),
    Input("execute", "n_clicks"),
    state=[
        State("pipeline", "value")
    ])
def update_pipeline_output(n_clicks, pipeline):
    if n_clicks is None:
        return dash.no_update

    # TODO: Add formatting, e.g. red text color for problem lines

    return pipeline


@app.callback([
        Output("dag", "figure"),
        Output("tabs", "value"),
        Output("results-detail", "children"),
    ],
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
        return dash.no_update, "definition-tab", None

    extracted_dag, inspection_results, _ = extract_dag(pipeline, checks, inspections)

    active_tab = "results-tab"

    # === DAG figure ===
    fig = nx2go(extracted_dag)

    # === Inspection results ===
    fig, output_rows_results = materialize_first_output_rows(fig, extracted_dag, inspection_results)

    # Display first output rows (results of MaterializeFirstOutputRows(5) inspection)
    details = []
    for node, df in output_rows_results:
        description = html.Div(
            "\n\033{} ({})\033\n{}\n{}".format(
                node.operator_type,
                node.description,
                node.source_code,
                node.code_reference,
            ),
            style=CODE_FONT,
        )
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            # style_cell={
            #     'width': '{}%'.format(len(df.columns)),
            #     'textOverflow': 'ellipsis',
            #     'overflow': 'hidden'
            # },
            # css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
        )
        details += [description, table]

    return fig, active_tab, details


def extract_dag(pipeline, checks=None, inspections=None):
    """Extract DAG the original way, i.e. by creating a PipelineInspectorBuilder."""
    start = time.time()
    builder = PipelineInspector.on_pipeline_from_string(pipeline)
    for inspection in inspections:
        builder = builder.add_required_inspection(inspection_switcher[inspection]())
    for check in checks:
        builder = builder.add_check(check_switcher[check]())
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


def get_new_node_label(node):
    """From mlinspect.visualisation._visualisation."""
    label = cleandoc("""
            {} (L{})
            {}
            """.format(node.operator_type.value, node.code_reference.lineno, node.description or ""))
    return label


def _get_pos(G):
    pos_dict = nx.nx_agraph.graphviz_layout(G, 'dot')
    # pos_json = {k.node_id: {'pos': v, 'node': k.to_dict()} for k, v in pos_dict.items()}
    # with open('pos_dict_with_checks.json', 'w') as f:
    #     json.dump(pos_json, f, indent="\t", ensure_ascii=True)

    nodes = G.nodes()
    edges = G.edges()

    Xn = []
    Yn = []
    for node in nodes:
        x, y = pos_dict[node]
        Xn += [x]
        Yn += [y]

    Xe = []
    Ye = []
    from addEdge import add_edge
    for edge0, edge1 in edges:
        Xe, Ye = add_edge(
            pos_dict[edge0],
            pos_dict[edge1],
            Xe,
            Ye,
            length_frac=0.8,
            arrow_pos='end',
            arrow_length=130,
            arrow_angle=5,
            # dot_size=15,
        )

    labels = []
    annotations = []
    for node, pos in pos_dict.items():
        labels += [get_new_node_label(node)]
        annotations += [{
            'x': pos[0],
            'y': pos[1],
            'text': node.operator_type.short_value,
            'showarrow': False,
        }]

    return Xn, Yn, Xe, Ye, labels, annotations


def nx2go(G):
    """
    Convert networkx.DiGraph to a plotly.graph_objects.Figure.

    Adapted from: https://chart-studio.plotly.com/~empet/14007/graphviz-networks-plotted-with-plotly/#/
    """
    Xn, Yn, Xe, Ye, labels, annotations = _get_pos(G)

    edges = go.Scatter(x=Xe, y=Ye, mode='lines', hoverinfo='none',
                       line={
                           'color': 'rgb(160,160,160)',
                           'width': 0.75,
                        })
    nodes = go.Scatter(x=Xn, y=Yn, mode='markers', name='', hoverinfo='text', text=labels,
                       marker={
                           'size': 15,
                           'color': '#85b6b6',
                           'line': {
                               'color': 'rgb(100,100,100)',
                               'width': 0.5,
                            },
                        })
    layout = go.Layout(
                title="Pipeline execution DAG",
                # font={'family': 'Balto'},
                # font={'family': "'Courier New', monospace"},
                font={'family': "Courier New"},
                # width=650,
                height=650,
                showlegend=False,
                xaxis={'visible': False},
                yaxis={'visible': False},
                # margin={'t': 100},
                hovermode='closest',
    )
    layout.annotations = annotations

    fig = go.Figure(data=[edges, nodes], layout=layout)

    return fig


def materialize_first_output_rows(figure, extracted_dag, inspection_results):
    try:
        first_rows_inspection_result = inspection_results[MaterializeFirstOutputRows(5)]
    except KeyError:
        return figure, []

    relevant_nodes = [node for node in extracted_dag.nodes if node.description in {
        "Imputer (SimpleImputer), Column: 'county'", "Categorical Encoder (OneHotEncoder), Column: 'county'"}]

    # Create scatter plot of these nodes
    pos_dict = nx.nx_agraph.graphviz_layout(extracted_dag, 'dot')  # TODO: Reuse from before rather than rerunning
    Xn = []
    Yn = []
    labels = []
    output_rows_results = []
    for dag_node in relevant_nodes:
        if dag_node in first_rows_inspection_result and first_rows_inspection_result[dag_node] is not None:
            x, y = pos_dict[dag_node]
            Xn += [x]
            Yn += [y]
            labels += [get_new_node_label(dag_node)]
            output_rows_results += [(dag_node, first_rows_inspection_result[dag_node])]
    nodes = go.Scatter(x=Xn, y=Yn, mode='markers', name='', hoverinfo='text', text=labels,
                       marker={
                           'size': 15,
                           'color': 'red',
                           'line': {
                               'color': 'red',
                               'width': 0.5,
                            },
                        })

    # Append scatter plot to figure
    figure.add_trace(nodes)

    return figure, output_rows_results


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
