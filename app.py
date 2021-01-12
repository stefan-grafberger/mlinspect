from ast import literal_eval
from contextlib import redirect_stdout
import io
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

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from mlinspect import PipelineInspector
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows
from mlinspect.visualisation import save_fig_to_path

# Initialize Dash app
app = dash.Dash(__name__,
                title="mlinspect",
                external_stylesheets=[
                    # Dash CSS
                    "https://codepen.io/chriddyp/pen/bWLwgP.css",

                    # Loading screen CSS
                    "https://codepen.io/chriddyp/pen/brPBPO.css",

                    # Bootstrap theme CSS
                    dbc.themes.BOOTSTRAP,  # pro: CSS classes; con: tiny font size
                    # dbc.themes.GRID,  # pro: grid layouts, large enough font size; con: no other dbc elements or CSS classes

                    # https://cdnjs.com/libraries/codemirror
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.32.0/codemirror.min.css",
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.59.1/theme/twilight.min.css"
                ],
                external_scripts=[
                    # https://cdnjs.com/libraries/codemirror
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.32.0/codemirror.min.js",
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.59.1/mode/python/python.min.js"
                ])
app.config.suppress_callback_exceptions = True
INSPECTOR_RESULT, POS_DICT = None, None


# Create HTML layout
CODE_FONT = {"fontFamily": "'Courier New', monospace"}
with open("example_pipelines/healthcare/healthcare.py") as f:
    default_pipeline = f.read()
patients = pd.read_csv("example_pipelines/healthcare/patients.csv", na_values='?')
histories = pd.read_csv("example_pipelines/healthcare/histories.csv", na_values='?')
healthcare_data = patients.merge(histories, on=['ssn'])
inspection_switcher = {
    "HistogramForColumns": HistogramForColumns(['age_group', 'race']),
    "RowLineage": lambda: RowLineage(5),
    "MaterializeFirstOutputRows": lambda: MaterializeFirstOutputRows(5),
}
check_switcher = {
    "NoBiasIntroducedFor": NoBiasIntroducedFor,
    "NoIllegalFeatures": NoIllegalFeatures,
    "NoMissingEmbeddings": NoMissingEmbeddings,
}
# app.layout = html.Div([     # for no margin
app.layout = dbc.Container([  # for more margin
    # Header and description
    html.Div([
        html.H1("mlinspect", style=CODE_FONT),
        html.P("Inspect ML Pipelines in Python in the form of a DAG."),
    ], id="header-container", className="container"),

    dbc.Row([
        dbc.Col([
            # Inspection definition
            dbc.Form([
                html.Div([
                    html.H3("Pipeline Definition"),
                    dbc.FormGroup([
                        # Pipeline definition
                        # dbc.Label("Pipeline definition:", html_for="pipeline"),
                        dbc.Textarea(
                            id="pipeline-textarea",
                            value=default_pipeline,
                            className="mb-3",
                        ),
                    ]),
                ], id="pipeline-definition-container", className="container"),
                html.Div([
                    html.H3("Pipeline Output"),
                    html.Pre(html.Code(id="pipeline-output")),
                ], id="pipeline-output-container", className="container", hidden=True),
                html.Div([
                    html.H3("Inspector Definition"),
                    dbc.FormGroup([
                        # Add inspections
                        dbc.Label("Run inspections:", html_for="inspections"),
                        dbc.Checklist(
                            id="inspections",
                            options=[
                                # {"label": "Histogram For Columns", "value": "HistogramForColumns"},
                                {"label": "Row Lineage", "value": "RowLineage"},
                                {"label": "Materialize First Output Rows", "value": "MaterializeFirstOutputRows"},
                            ],
                            switch=True,
                            value=[],
                        ),
                    ]),
                    dbc.FormGroup([
                        # Add checks
                        dbc.Label("Run checks:", html_for="checks"),
                        html.Div([
                            html.Div([
                                dbc.Checkbox(id="nobiasintroduced-checkbox",
                                             className="custom-control-input"),
                                dbc.Label("No Bias Introduced For",
                                          html_for="nobiasintroduced-checkbox",
                                          className="custom-control-label"),
                                dbc.Checklist(id="sensitive-columns",
                                              options=[{"label": column, "value": column}
                                                       for column in healthcare_data.columns],
                                              style={"display": "none"}),
                            ], className="custom-switch custom-control"),
                            html.Div([
                                dbc.Checkbox(id="noillegalfeatures-checkbox",
                                             className="custom-control-input"),
                                dbc.Label("No Illegal Features",
                                          html_for="noillegalfeatures-checkbox",
                                          className="custom-control-label"),
                            ], className="custom-switch custom-control"),
                            html.Div([
                                dbc.Checkbox(id="nomissingembeddings-checkbox",
                                             className="custom-control-input"),
                                dbc.Label("No Missing Embeddings",
                                          html_for="nomissingembeddings-checkbox",
                                          className="custom-control-label"),
                            ], className="custom-switch custom-control"),
                        ], id="checks"),
                    ]),
                ], id="inspector-definition-container", className="container"),
                # Execute inspection
                dbc.Button("Execute", id="execute", color="primary", size="lg", className="mr-1"),
            ]),
        ], width=6),
        dbc.Col([
            # Extracted DAG
            html.Div([
                html.H3("Extracted DAG"),
                dcc.Graph(
                    id="dag",
                    figure=go.Figure(
                        layout_height=650,
                        layout_showlegend=False,
                        layout_xaxis={'visible': False},
                        layout_yaxis={'visible': False},
                        layout_plot_bgcolor='rgb(255,255,255)',
                    ),
                ),
            ], id="dag-container", className="container"),
            html.Div([
                html.Div(id="hovered-code-reference"),
                html.Div(id="selected-code-reference"),
            ], id="code-reference-container", className="container", hidden=True),
            html.Br(),
            # Operator details
            html.Div([
                html.H3("Operator Details"),
                html.Div("Select an operator in the DAG to see details", id="operator-details"),
            ], id="operator-details-container", className="container"),
        ], width=6),
    ]),
], style={"fontSize": "14px"}, id="app-container")


# Flask server (for gunicorn)
server = app.server


@app.callback(
    Output("sensitive-columns", "style"),
    Input("nobiasintroduced-checkbox", "checked"),
)
def on_nobiasintroduced_checked(checked):
    """Show checklist of sensitive columns if NoBiasIntroducedFor is selected."""
    if checked:
        return {"display": "block", **CODE_FONT}
    return {"display": "none"}


@app.callback(
    [
        Output("dag", "figure"),
        Output("pipeline-output", "children"),
        Output("pipeline-output-container", "hidden"),
    ],
    Input("execute", "n_clicks"),
    state=[
        State("pipeline-textarea", "value"),
        State("nobiasintroduced-checkbox", "checked"),
        State("sensitive-columns", "value"),
        State("noillegalfeatures-checkbox", "checked"),
        State("nomissingembeddings-checkbox", "checked"),
        State("inspections", "value"),
    ]
)
def on_execute(execute_clicks, pipeline, nobiasintroduced, sensitive_columns,
               noillegalfeatures, nomissingembeddings, inspections):
    """
    When user clicks 'execute' button, show extracted DAG including potential
    problem nodes in red.
    """
    if not execute_clicks:
        return dash.no_update, dash.no_update, dash.no_update

    # Execute pipeline and inspections
    checks = {
        "NoBiasIntroducedFor": (nobiasintroduced, [sensitive_columns]),
        "NoIllegalFeatures": (noillegalfeatures, []),
        "NoMissingEmbeddings": (nomissingembeddings, []),
    }
    pipeline_output = execute_inspector_builder(pipeline, checks, inspections)
    hide_output = False

    # Convert extracted DAG into plotly figure
    figure = nx2go(INSPECTOR_RESULT.dag)

    # Highlight problematic nodes
    figure = highlight_problem_nodes(figure, sensitive_columns)

    return figure, pipeline_output, hide_output


@app.callback(
    Output("hovered-code-reference", "children"),
    Input("dag", "hoverData"),
)
def on_dag_node_hover(hoverData):
    """
    When user hovers on DAG node, show node label and emphasize corresponding
    source code.
    """
    # Un-highlight source code
    if not hoverData:
        return []

    # Find DagNode object at this position
    point = hoverData['points'][0]
    x = point['x']
    y = point['y']
    try:
        node = [node for node, pos in POS_DICT.items() if pos == (x, y)][0]
    except IndexError:
        print(f"[hover] Could not find node with pos {x} and {y}")
        return dash.no_update

    # Highlight source code
    code_ref = node.code_reference
    return json.dumps(code_ref.__dict__)


@app.callback(
    [
        Output("selected-code-reference", "children"),
        Output("operator-details", "children"),
    ],
    [
        Input("dag", "selectedData"),
    ],
)
def on_dag_node_select(selectedData):
    """
    When user selects DAG node, show detailed check and inspection results
    and emphasize corresponding source code.
    """
    # Un-highlight source code
    if not selectedData:
        return [], dash.no_update

    # Find DagNode object at this position
    point = selectedData['points'][0]
    x = point['x']
    y = point['y']
    try:
        node = [node for node, pos in POS_DICT.items() if pos == (x, y)][0]
    except IndexError:
        print(f"[select] Could not find node with pos {x} and {y}")
        return dash.no_update, dash.no_update, dash.no_update

    # Highlight source code
    code_ref = node.code_reference

    # Populate and show div(s)
    operator_details = get_operator_details(node)

    return json.dumps(code_ref.__dict__), operator_details


def execute_inspector_builder(pipeline, checks=None, inspections=None):
    """Extract DAG the original way, i.e. by creating a PipelineInspectorBuilder."""
    global INSPECTOR_RESULT

    start = time.time()

    builder = PipelineInspector.on_pipeline_from_string(pipeline)
    for inspection in inspections:
        builder = builder.add_required_inspection(inspection_switcher[inspection]())
    for check_name, (check_bool, check_args) in checks.items():
        if check_bool:
            builder = builder.add_check(check_switcher[check_name](*check_args))

    f = io.StringIO()
    with redirect_stdout(f):
        INSPECTOR_RESULT = builder.execute()
    out = f.getvalue()

    print(f"Total time in seconds: {time.time() - start}")

    return out


def get_new_node_label(node):
    """From mlinspect.visualisation._visualisation."""
    label = cleandoc("""
            {} (L{})
            {}
            """.format(node.operator_type.value, node.code_reference.lineno, node.description or ""))
    return label


def _get_pos(G):
    global POS_DICT
    POS_DICT = nx.nx_agraph.graphviz_layout(G, 'dot')

    nodes = G.nodes()
    edges = G.edges()

    Xn, Yn = [], []
    for node in nodes:
        x, y = POS_DICT[node]
        Xn += [x]
        Yn += [y]

    Xe, Ye = [], []
    from addEdge import add_edge
    for edge0, edge1 in edges:
        Xe, Ye = add_edge(
            POS_DICT[edge0],
            POS_DICT[edge1],
            Xe,
            Ye,
            length_frac=1,
            # arrow_pos='end',
            arrow_length=130,
            arrow_angle=5,
            # dot_size=15,
        )

    labels = []
    annotations = []
    for node, pos in POS_DICT.items():
        labels += [get_new_node_label(node)]
        annotations += [{
            'x': pos[0],
            'y': pos[1],
            'text': node.operator_type.short_value,
            'showarrow': False,
            'font': {
                "size": 16,
            },
        }]

    return Xn, Yn, Xe, Ye, labels, annotations


def nx2go(G):
    """
    Convert networkx.DiGraph to a plotly.graph_objects.Figure.

    Adapted from: https://chart-studio.plotly.com/~empet/14007/graphviz-networks-plotted-with-plotly/#/
    """
    Xn, Yn, Xe, Ye, labels, annotations = _get_pos(G)

    edges = go.Scatter(
        x=Xe, y=Ye, mode='lines', hoverinfo='none',
        line={
            'color': 'rgb(160,160,160)',
            'width': 0.75,
        },
    )
    nodes = go.Scatter(
        x=Xn, y=Yn, mode='markers', name='', hoverinfo='text', text=labels,
        marker={
            'size': 20,
            'color': 'rgb(200,200,200)',
            'line': {
                'color': 'black',
                'width': 0.5,
            },
        },
    )
    layout = go.Layout(
        font={'family': "Courier New"},
        font_color='black',
        height=650,
        showlegend=False,
        xaxis={'visible': False},
        yaxis={'visible': False},
        margin= {
            'l': 1,
            'r': 1,
            'b': 1,
            't': 1,
            'pad': 1,
        },
        hovermode='closest',
        plot_bgcolor='white',
    )
    layout.annotations = annotations

    fig = go.Figure(data=[edges, nodes], layout=layout)
    fig.update_layout(clickmode='event+select')

    return fig#.to_dict()


def highlight_dag_node_in_figure(dag_node, figure):
    # Create scatter plot of this node
    Xn, Yn = POS_DICT[dag_node]
    label = get_new_node_label(dag_node)
    nodes = go.Scatter(
        x=[Xn], y=[Yn], mode='markers', name='', hoverinfo='text', text=[label],
        marker={
            'size': 20,
            'color': 'red',
            'line': {
                'color': 'red',
                'width': 0.5,
            },
        },
    )

    # Append scatter plot to figure
    if isinstance(figure, dict):
        figure['data'].append(nodes)
    else:
        figure.add_trace(nodes)

    return figure


def highlight_problem_nodes(fig_dict, sensitive_columns):
    """From mlinspect.checks._no_bias_introduced_for:NoBiasIntroducedFor.plot_distribution_change_histograms."""
    try:
        no_bias_check_result = INSPECTOR_RESULT.check_to_check_results[NoBiasIntroducedFor(sensitive_columns)]
    except (KeyError, TypeError):
        return fig_dict

    for node_dict in no_bias_check_result.bias_distribution_change.values():
        for distribution_change in node_dict.values():
            # Check if distribution change is acceptable
            if distribution_change.acceptable_change:
                continue

            # Highlight this node in figure
            fig_dict = highlight_dag_node_in_figure(distribution_change.dag_node, fig_dict)

    return fig_dict


def create_histogram(column, distribution_change):
    keys = distribution_change.before_and_after_df["sensitive_column_value"]
    keys = [str(key) for key in keys]  # Necessary because of null values
    before_values = distribution_change.before_and_after_df["count_before"]
    after_values = distribution_change.before_and_after_df["count_after"]
    trace1 = go.Bar(x=keys, y=before_values, name="Before")
    trace2 = go.Bar(x=keys, y=after_values, name="After")

    data = [trace1, trace2]
    title = {
        "text": f"Line {distribution_change.dag_node.code_reference.lineno}, "\
                f"Operator '{distribution_change.dag_node.operator_type.value}', "\
                f"Column '{column}'",
        "font_size": 12,
    }
    margin = {"l": 20, "r": 20, "t": 20, "b": 20}

    layout = go.Layout(title=title, margin=margin, autosize=False, width=350, height=300)
    figure = go.Figure(data=data, layout=layout)
    return dcc.Graph(figure=figure)


def get_operator_details(node):
    details = []

    # Show inspection results
    for inspection, result_dict in INSPECTOR_RESULT.inspection_to_annotations.items():
        if isinstance(inspection, MaterializeFirstOutputRows):
            df = result_dict[node]
            columns = [{"name": i, "id": i} for i in df.columns]
            data = [
                {
                    k: np.array2string(v) if isinstance(v, np.ndarray) else v
                    for k, v in record.items()
                }
                for record in df.to_dict('records')
            ]
            element = html.Div([
                html.H4(f"{inspection}"),
                dash_table.DataTable(columns=columns, data=data)
            ], className="result-item")
            details += [element]
        else:
            print("inspection:", inspection)

    # Show check results
    for check, result_obj in INSPECTOR_RESULT.check_to_check_results.items():
        if isinstance(check, NoBiasIntroducedFor):
            if node not in result_obj.bias_distribution_change:
                print("check:", check)
                continue
            graphs = []
            for column, distribution_change in result_obj.bias_distribution_change[node].items():
                graphs += [create_histogram(column, distribution_change)]
            element = html.Div([
                html.H4(f"{check}"),
                html.Div(graphs),
            ], className="result-item")
            details += [element]
        else:
            print("check:", check)

    return details


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
