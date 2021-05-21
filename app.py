# pylint: disable=import-error
import os
import time

from contextlib import redirect_stdout
import io
import json
import numpy as np

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from inspect import cleandoc

import networkx as nx

import pandas as pd

import plotly.graph_objects as go

from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings
from mlinspect import PipelineInspector
from mlinspect.checks import NoBiasIntroducedFor, NoIllegalFeatures
from mlinspect.inspections import HistogramForColumns, RowLineage, MaterializeFirstOutputRows


# === Initialize Dash app ===
app = dash.Dash(__name__,
                title="mlinspect",
                external_stylesheets=[
                    # Dash CSS
                    "https://codepen.io/chriddyp/pen/bWLwgP.css",

                    # Loading screen CSS
                    "https://codepen.io/chriddyp/pen/brPBPO.css",

                    # Bootstrap theme CSS
                    dbc.themes.BOOTSTRAP,

                    # CodeMirror stylesheets: https://cdnjs.com/libraries/codemirror
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.32.0/codemirror.min.css",
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.59.1/theme/twilight.min.css"
                ],
                external_scripts=[
                    # CodeMirror scripts: https://cdnjs.com/libraries/codemirror
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.32.0/codemirror.min.js",
                    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.59.1/mode/python/python.min.js"
                ])
app.config.suppress_callback_exceptions = True
INSPECTOR_RESULT, POS_DICT = None, None


# === Create HTML layout ===
CODE_FONT = {"fontFamily": "'Courier New', monospace"}
STYLE_HIDDEN = {"display": "none"}
STYLE_SHOWN = {"display": "block"}

with open("example_pipelines/healthcare/healthcare.py") as healthcare_file:
    healthcare_pipeline = healthcare_file.read()
with open("example_pipelines/adult_demo/adult_demo.py") as adult_file:
    adult_pipeline = adult_file.read()

patients = pd.read_csv("example_pipelines/healthcare/patients.csv", na_values='?')
histories = pd.read_csv("example_pipelines/healthcare/histories.csv", na_values='?')
healthcare_data = patients.merge(histories, on=['ssn'])
adult_data = pd.read_csv("example_pipelines/adult_demo/train.csv",
                         na_values='?', index_col=0)

inspection_switcher = {
    "HistogramForColumns": HistogramForColumns,
    "RowLineage": RowLineage,
    "MaterializeFirstOutputRows": MaterializeFirstOutputRows,
}
check_switcher = {
    "NoBiasIntroducedFor": NoBiasIntroducedFor,
    "NoIllegalFeatures": NoIllegalFeatures,
    "NoMissingEmbeddings": NoMissingEmbeddings,
}

app.layout = dbc.Container([
    # Header and description
    dbc.Row([
        dbc.Col([
            html.H2("mlinspect"),
        ], width=4),
        dbc.Col([
            html.H2("Inspect ML Pipelines in Python in the form of a DAG.", style={"textAlign": "right"}),
        ], width=8),
    ], id="header-container", className="container", style=CODE_FONT),
    html.Hr(),

    dbc.Row([
        # Pipeline
        dbc.Col([
            # Pipeline definition
            html.Div([
                html.H3("Pipeline Definition"),
                dbc.FormGroup([
                    # Paste text from pipeline: Healthcare Adult
                    html.Div("Paste text from pipeline:"),
                    dbc.Button("Healthcare", id="healthcare-pipeline", color="secondary", size="lg", className="mr-1"),
                    dbc.Button("Adult Income", id="adult-pipeline", color="secondary", size="lg", className="mr-1"),
                    dbc.Textarea(id="pipeline-textarea", className="mb-3"),
                    html.Div(healthcare_pipeline, id="healthcare-pipeline-text", hidden=True),
                    html.Div(adult_pipeline, id="adult-pipeline-text", hidden=True),
                ]),
            ], id="pipeline-definition-container", className="container"),
            # Pipeline execution output
            html.Div([
                html.H3("Pipeline Output"),
                html.Pre(html.Code(id="pipeline-output"), id="pipeline-output-cell"),
            ], id="pipeline-output-container", className="container", hidden=True),
        ], width=7),
        # DAG
        dbc.Col([
            # Extracted DAG
            html.Div([
                html.H3("Extracted DAG"),
                dcc.Graph(
                    id="dag",
                    figure=go.Figure(
                        layout_width=500,
                        layout_height=500,
                        layout_showlegend=False,
                        layout_xaxis={'visible': False},
                        layout_yaxis={'visible': False},
                        layout_plot_bgcolor='rgb(255,255,255)',
                    ),
                ),
            ], id="dag-container", className="container"),
            # Code references for highlighting source code (hidden)
            html.Div([
                html.Div(id="hovered-code-reference"),
                html.Div(id="selected-code-reference"),
            ], id="code-reference-container", className="container", hidden=True),
        ], width=5),
    ]),
    dbc.Row([
        # Inspections
        dbc.Col([
            dbc.FormGroup([
                # Add inspections
                html.H3("Inspections"),
                html.Div([
                    html.Div([  # Histogram For Columns
                        dbc.Checkbox(id="histogramforcolumns-checkbox",
                                     className="custom-control-input"),
                        dbc.Label("Histogram For Columns",
                                  html_for="histogramforcolumns-checkbox",
                                  className="custom-control-label"),
                        dbc.Checklist(id="histogram-sensitive-columns",
                                      options=[{"label": "label1", "value": "value1"},
                                               {"label": "label2", "value": "value2"}],
                                      style=STYLE_HIDDEN),
                    ], className="custom-switch custom-control"),
                    html.Div([  # Row Lineage
                        dbc.Checkbox(id="rowlineage-checkbox",
                                     className="custom-control-input"),
                        dbc.Label("Row Lineage",
                                  html_for="rowlineage-checkbox",
                                  className="custom-control-label"),
                        dbc.Input(id="rowlineage-num-rows", type="number",
                                  min=0, step=1, placeholder="Number of rows (default 5)",
                                  style=STYLE_HIDDEN),
                    ], className="custom-switch custom-control"),
                    html.Div([  # Materialize First Output Rows
                        dbc.Checkbox(id="materializefirstoutputrows-checkbox",
                                     className="custom-control-input"),
                        dbc.Label("Materialize First Output Rows",
                                  html_for="materializefirstoutputrows-checkbox",
                                  className="custom-control-label"),
                        dbc.Input(id="materializefirstoutputrows-num-rows", type="number",
                                  min=0, step=1, placeholder="Number of rows (default 5)",
                                  style=STYLE_HIDDEN),
                    ], className="custom-switch custom-control"),
                ]),
            ]),
        ], width=3),
        # Checks
        dbc.Col([
            dbc.FormGroup([
                # Add checks
                html.H3("Checks"),
                html.Div([
                    html.Div([  # No Bias Introduced For
                        dbc.Checkbox(id="nobiasintroduced-checkbox",
                                     className="custom-control-input"),
                        dbc.Label("No Bias Introduced For",
                                  html_for="nobiasintroduced-checkbox",
                                  className="custom-control-label"),
                        #   min_allowed_relative_ratio_change=-0.3
                        dbc.Input(id="nobiasintroduced-ratio-threshold", type="number",
                                  min=-100, max=0, step=1,
                                  placeholder="Min ratio change -30%",
                                  style=STYLE_HIDDEN),
                        #   max_allowed_probability_difference=2.0
                        dbc.Input(id="nobiasintroduced-probability-threshold", type="number",
                                  min=0, step=1,
                                  placeholder="Max prob diff 200%",
                                  style=STYLE_HIDDEN),
                        dbc.Checklist(id="nobiasintroduced-sensitive-columns",
                                      options=[{"label": "label1", "value": "value1"},
                                               {"label": "label2", "value": "value2"}],
                                      style=STYLE_HIDDEN),
                    ], className="custom-switch custom-control"),
                    html.Div([  # No Illegal Features
                        dbc.Checkbox(id="noillegalfeatures-checkbox",
                                     className="custom-control-input"),
                        dbc.Label("No Illegal Features",
                                  html_for="noillegalfeatures-checkbox",
                                  className="custom-control-label"),
                    ], className="custom-switch custom-control"),
                    html.Div([  # No Missing Embeddings
                        dbc.Checkbox(id="nomissingembeddings-checkbox",
                                     className="custom-control-input"),
                        dbc.Label("No Missing Embeddings",
                                  html_for="nomissingembeddings-checkbox",
                                  className="custom-control-label"),
                    ], className="custom-switch custom-control"),
                ], id="checks"),
            ]),
        ], width=3),
        # Execute
        dbc.Col([
            # Execute inspection
            html.Br(),
            html.Br(),
            dbc.Button(id="execute", color="primary", size="lg", className="mr-1 play-button"),
        ], width=1),
        # Details
        dbc.Col([
            # Summary
            html.Div([
                html.H3("Summary", id="results-summary-header"),
                html.Div("Select inspections and/or checks and execute to see results",
                         id="results-summary"),
            ], id="results-summary-container", className="container"),
            # Details
            html.Div([
                html.H3("Details", id="results-details-header"),
                html.Div("Select an operator in the DAG to see operator-specific details",
                         id="results-details"),
            ], id="results-details-container", className="container"),
        ], id="results-container", width=5, align="end")
    ], id="inspector-definition-container", className="container"),
    html.Div(id="clientside-pipeline-code", hidden=True)
], style={"fontSize": "14px"}, id="app-container")


# === Flask server (for gunicorn) ===
server = app.server


# === Set up callback functions ===
app.clientside_callback(
    """
    function(n_clicks) {
        var editor = document.querySelector('#pipeline-textarea');
        return editor.value;
    }
    """,
    Output('clientside-pipeline-code', 'children'),
    Input("execute", "n_clicks")
)


app.clientside_callback(
    """
    function(healthcare_clicked, adult_clicked) {
        const ctx = dash_clientside.callback_context;
        if (ctx.triggered.length === 0) {
            return '';
        }
        const prop_id = ctx.triggered[0]["prop_id"];

        var text = '';
        if (prop_id.startsWith("healthcare")) {
            const healthcareElem = document.getElementById('healthcare-pipeline-text');
            text = healthcareElem.textContent;
        } else if (prop_id.startsWith("adult")) {
            const adultElem = document.getElementById('adult-pipeline-text');
            text = adultElem.textContent;
        }

        var editor = document.querySelector('.CodeMirror').CodeMirror;
        editor.setValue(text);

        return text;
    }
    """,
    Output("pipeline-textarea", "value"),
    [
        Input("healthcare-pipeline", "n_clicks"),
        Input("adult-pipeline", "n_clicks"),
    ]
)


@app.callback(
    [
        Output("histogram-sensitive-columns", "options"),
        Output("nobiasintroduced-sensitive-columns", "options"),
    ],
    [
        Input("healthcare-pipeline", "n_clicks"),
        Input("adult-pipeline", "n_clicks"),
    ],
)
def on_sensitive_column_options_changed(healthcare_clicked, adult_clicked):
    if not healthcare_clicked and not adult_clicked:
        return dash.no_update

    ctx = dash.callback_context
    elem_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if elem_id == "healthcare-pipeline":
        columns = healthcare_data.columns
    elif elem_id == "adult-pipeline":
        columns = adult_data.columns
    else:
        columns = []

    options = [{"label": c, "value": c} for c in columns]
    return options, options


@app.callback(
    Output("histogram-sensitive-columns", "style"),
    Input("histogramforcolumns-checkbox", "checked"),
)
def on_histogramforcolumns_checked(checked):
    """Show checklist of sensitive columns if HistogramForColumns is selected."""
    if checked:
        return {**STYLE_SHOWN, **CODE_FONT}
    return STYLE_HIDDEN


@app.callback(
    Output("rowlineage-num-rows", "style"),
    Input("rowlineage-checkbox", "checked"),
)
def on_rowlineage_checked(checked):
    """Show input for number of rows if RowLineage is selected."""
    if checked:
        return STYLE_SHOWN
    return STYLE_HIDDEN


@app.callback(
    Output("materializefirstoutputrows-num-rows", "style"),
    Input("materializefirstoutputrows-checkbox", "checked"),
)
def on_materializefirstoutputrows_checked(checked):
    """Show input for number of rows if MaterializeFirstOutputRows is selected."""
    if checked:
        return STYLE_SHOWN
    return STYLE_HIDDEN


@app.callback(
    [
        Output("nobiasintroduced-ratio-threshold", "style"),
        Output("nobiasintroduced-probability-threshold", "style"),
        Output("nobiasintroduced-sensitive-columns", "style"),
    ],
    Input("nobiasintroduced-checkbox", "checked"),
)
def on_nobiasintroduced_checked(checked):
    """Show checklist of sensitive columns if NoBiasIntroducedFor is selected."""
    if checked:
        return STYLE_SHOWN, STYLE_SHOWN, {**STYLE_SHOWN, **CODE_FONT}
    return STYLE_HIDDEN, STYLE_HIDDEN, STYLE_HIDDEN


@app.callback(
    [
        Output("dag", "figure"),
        Output("pipeline-output", "children"),
        Output("pipeline-output-container", "hidden"),
        Output("dag", "selectedData"),
        Output("results-summary", "children"),
    ],
    Input("execute", "n_clicks"),
    Input("clientside-pipeline-code", "children"),
    state=[
        # HistogramForColumns
        State("histogramforcolumns-checkbox", "checked"),
        State("histogram-sensitive-columns", "value"),
        # RowLineage
        State("rowlineage-checkbox", "checked"),
        State("rowlineage-num-rows", "value"),
        # MaterializeFirstOutputRows
        State("materializefirstoutputrows-checkbox", "checked"),
        State("materializefirstoutputrows-num-rows", "value"),
        # NoBiasIntroducedFor
        State("nobiasintroduced-checkbox", "checked"),
        State("nobiasintroduced-sensitive-columns", "value"),
        State("nobiasintroduced-ratio-threshold", "value"),
        State("nobiasintroduced-probability-threshold", "value"),
        # NoIllegalFeatures
        State("noillegalfeatures-checkbox", "checked"),
        # NoMissingEmbeddings
        State("nomissingembeddings-checkbox", "checked"),
    ]
)
def on_execute(execute_clicks, pipeline,
               # Inspections
               histogramforcolumns, histogramforcolumns_sensitive_columns,
               rowlineage, rowlineage_num_rows,
               materializefirstoutputrows, materializefirstoutputrows_num_rows,
               # Checks
               nobiasintroduced, nobiasintroduced_sensitive_columns,
               nobiasintroduced_ratio_threshold, nobiasintroduced_probability_threshold,
               noillegalfeatures, nomissingembeddings):
    """
    When user clicks 'execute' button, show extracted DAG including potential
    problem nodes in red.
    """
    if not execute_clicks or not pipeline:
        return [dash.no_update]*5

    ### Execute pipeline and inspections
    # params for NoBiasIntroducedFor
    if nobiasintroduced_ratio_threshold:
        # convert percentage to decimal
        nobiasintroduced_ratio_threshold = nobiasintroduced_ratio_threshold/100.
    else:
        # use default value if None
        nobiasintroduced_ratio_threshold = -0.3
    if nobiasintroduced_probability_threshold:
        # convert percentage to decimal
        nobiasintroduced_probability_threshold = nobiasintroduced_probability_threshold/100.
    else:
        # use default value if None
        nobiasintroduced_probability_threshold = 2.0
    # if both RowLineage and MaterializeFirstOutputRows are enabled, display the higher number of rows
    if materializefirstoutputrows and rowlineage:
        rowlineage_num_rows = max(rowlineage_num_rows, materializefirstoutputrows_num_rows)
    # don't include MaterializeFirstOutputRows if RowLineage is also checked
    materializefirstoutputrows = materializefirstoutputrows and not rowlineage
    # construct arguments for inspector builder
    inspections = {
        "HistogramForColumns": (histogramforcolumns, [histogramforcolumns_sensitive_columns]),
        "RowLineage": (rowlineage, [rowlineage_num_rows or 5]),
        "MaterializeFirstOutputRows": (materializefirstoutputrows,
                                       [materializefirstoutputrows_num_rows or 5]),
    }
    checks = {
        "NoBiasIntroducedFor": (nobiasintroduced, [
            nobiasintroduced_sensitive_columns,
            nobiasintroduced_ratio_threshold,
            nobiasintroduced_probability_threshold,
        ]),
        "NoIllegalFeatures": (noillegalfeatures, []),
        "NoMissingEmbeddings": (nomissingembeddings, []),
    }
    pipeline_output = execute_inspector_builder(pipeline, checks, inspections)
    hide_output = False

    ### Convert extracted DAG into plotly figure
    figure = nx2go(INSPECTOR_RESULT.dag)

    ### Highlight problematic nodes
    figure = highlight_problem_nodes(figure, nobiasintroduced_sensitive_columns)

    ### De-select any DAG nodes and trigger callback to reset details div
    selected_data = {}

    ### Summary results
    summary = get_result_summary()

    return figure, pipeline_output, hide_output, selected_data, summary


@app.callback(
    Output("hovered-code-reference", "children"),
    Input("dag", "hoverData"),
)
def on_dag_node_hover(hover_data):
    """
    When user hovers on DAG node, show node label and emphasize corresponding
    source code.
    """
    # Un-highlight source code
    if not hover_data:
        return []

    # Find DagNode object at this position
    point = hover_data['points'][0]
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
        Output("results-details", "children"),
        Output("results-details-header", "children"),
    ],
    [
        Input("dag", "selectedData"),
    ],
)
def on_dag_node_select(selected_data):
    """
    When user selects DAG node, show detailed check and inspection results
    and emphasize corresponding source code.
    """
    # Un-highlight source code
    if not selected_data:
        return [], "Select an operator in the DAG to see operator-specific details", "Details"

    # Find DagNode object at this position
    point = selected_data['points'][0]
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
    header = "Details: Operator '{operator}', Line {code_ref}".format(
        operator=node.operator_type.value,
        code_ref=node.code_reference.lineno,
    )
    operator_details = get_result_details(node)

    return json.dumps(code_ref.__dict__), operator_details, header


# === Utility functions ===
def execute_inspector_builder(pipeline, checks=None, inspections=None):
    """Extract DAG the original way, i.e. by creating a PipelineInspectorBuilder."""
    global INSPECTOR_RESULT

    builder = PipelineInspector.on_pipeline_from_string(pipeline)
    for inspection_name, (inspection_bool, inspection_args) in inspections.items():
        if inspection_bool:
            builder = builder.add_required_inspection(inspection_switcher[inspection_name](*inspection_args))
    for check_name, (check_bool, check_args) in checks.items():
        if check_bool:
            builder = builder.add_check(check_switcher[check_name](*check_args))

    start = time.time()

    f = io.StringIO()
    with redirect_stdout(f):
        INSPECTOR_RESULT = builder.execute()
    out = f.getvalue()

    print(f"Execution time: {time.time() - start:.02f} seconds")

    return out


def get_new_node_label(node):
    """From mlinspect.visualisation._visualisation."""
    label = cleandoc("""
            {} (L{})
            {}
            """.format(
                node.operator_type.value,
                node.code_reference.lineno,
                node.description or ""
            ))
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
        width=500,
        height=500,
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

    return fig


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
            if distribution_change.acceptable_change and distribution_change.acceptable_probability_difference:
                continue

            # Highlight this node in figure
            fig_dict = highlight_dag_node_in_figure(distribution_change.dag_node, fig_dict)

    return fig_dict


def create_distribution_histogram(column, distribution_dict):
    keys = list(distribution_dict.keys())
    counts = list(distribution_dict.values())
    data = go.Bar(x=keys, y=counts, text=counts, hoverinfo="text")
    title = {
        "text": f"Column '{column}' Distribution",
        "font_size": 12,
    }
    margin = {"l": 20, "r": 20, "t": 20, "b": 20}

    layout = go.Layout(title=title, margin=margin, hovermode="x",
                       autosize=False, width=380, height=300)
    figure = go.Figure(data=data, layout=layout)
    return dcc.Graph(figure=figure)


def create_distribution_change_histograms(column, distribution_change):
    keys = distribution_change.before_and_after_df["sensitive_column_value"]
    keys = [str(key) for key in keys]  # Necessary because of null values
    before_values = distribution_change.before_and_after_df["count_before"]
    after_values = distribution_change.before_and_after_df["count_after"]
    before_text = distribution_change.before_and_after_df["ratio_before"]
    after_text = distribution_change.before_and_after_df["ratio_after"]
    before = go.Bar(x=keys, y=before_values, name="before", text=before_text, hoverinfo="text")
    after = go.Bar(x=keys, y=after_values, name="after", text=after_text, hoverinfo="text")

    data = [before, after]
    title = {
        "text": f"Column '{column}' Distribution Change",
        "font_size": 12,
    }
    margin = {"l": 20, "r": 20, "t": 20, "b": 20}

    layout = go.Layout(title=title, margin=margin, hovermode="x",
                       autosize=False, width=380, height=300)
    figure = go.Figure(data=data, layout=layout)
    figure.update_traces(hovertemplate="%{text:.2f}")
    return dcc.Graph(figure=figure)


def create_removal_probability_histograms(column, distribution_change):
    keys = distribution_change.before_and_after_df["sensitive_column_value"]
    keys = [str(key) for key in keys]  # Necessary because of null values
    removal_probabilities = distribution_change.before_and_after_df["removal_probability"]
    data = go.Bar(
        x=keys, y=removal_probabilities,
        text=removal_probabilities,
        hoverinfo="text",
        hovertemplate="%{text:.2f}",
    )
    title = {
        "text": f"Column '{column}' Removal Probabilities",
        "font_size": 12,
    }
    margin = {"l": 20, "r": 20, "t": 20, "b": 20}

    layout = go.Layout(title=title, margin=margin, hovermode="x",
                       autosize=False, width=380, height=300)
    figure = go.Figure(data=data, layout=layout)
    # figure.update_traces(marker_color="orange")
    return dcc.Graph(figure=figure)


def convert_dataframe_to_dash_table(df):
    columns = [{"name": i, "id": i} for i in df.columns]
    data = [
        {
            k: np.array2string(v, precision=2, threshold=2)
            if isinstance(v, np.ndarray) else str(v)
            for k, v in record.items()
        }
        for record in df.to_dict('records')
    ]
    # data = []
    # for record in df.to_dict('records'):
    #     record_dict = {}
    #     for k, v in record.items():
    #         if isinstance(v, np.ndarray):
    #             record_dict[k] = np.array2string(v, precision=2, threshold=2)
    #         elif isinstance(v, set):
    #             record_dict[k] = "{" + "\n".join(map(str, v)) + "}"
    #         else:
    #             record_dict[k] = str(v)
    #     data += [record_dict]
    return dash_table.DataTable(
        columns=columns,
        data=data,
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
    )


def get_result_summary():
    check_results = INSPECTOR_RESULT.check_to_check_results
    check_result_df = PipelineInspector.check_results_as_data_frame(check_results)
    return convert_dataframe_to_dash_table(check_result_df)


def get_result_details(node):
    details = []

    # Show inspection results
    for inspection, result_dict in INSPECTOR_RESULT.inspection_to_annotations.items():
        if isinstance(inspection, RowLineage) or isinstance(inspection, MaterializeFirstOutputRows):
            output_df = result_dict[node]
            output_table = convert_dataframe_to_dash_table(output_df)
            input_tables = [
                convert_dataframe_to_dash_table(result_dict[input_node])
                for input_node in INSPECTOR_RESULT.dag.predecessors(node)
            ]
            if input_tables:
                input_tables.insert(0, dbc.Label("Input Rows"))
            element = html.Div([
                html.H4(f"{inspection}", className="result-item-header"),
                *input_tables,
                dbc.Label("Output Rows"),
                output_table,
            ], className="result-item")
            details += [element]
        elif isinstance(inspection, HistogramForColumns):
            if node not in result_dict:
                continue

            distribution_dicts = result_dict[node]
            graphs = []
            for column, distribution in distribution_dicts.items():
                graphs += [create_distribution_histogram(column, distribution)]
            element = html.Div([
                html.H4(f"{inspection}", className="result-item-header"),
                html.Div(graphs, className="result-item-content"),
            ], className="result-item")
            details += [element]
        else:
            print("inspection not implemented:", inspection)

    # Show check results
    for check, result_obj in INSPECTOR_RESULT.check_to_check_results.items():
        if isinstance(check, NoBiasIntroducedFor):
            if node not in result_obj.bias_distribution_change:
                continue
            dist_changes = result_obj.bias_distribution_change[node]
            graphs = []
            for column, distribution_change in dist_changes.items():
                graphs += [
                    create_distribution_change_histograms(column, distribution_change),
                    create_removal_probability_histograms(column, distribution_change),
                ]
            element = html.Div([
                html.H4(f"{check}", className="result-item-header"),
                html.Div(graphs, className="result-item-content"),
            ], className="result-item")
            details += [element]
        elif isinstance(check, NoIllegalFeatures):
            # already shown in results summary
            pass
        elif isinstance(check, NoMissingEmbeddings):
            if node not in result_obj.dag_node_to_missing_embeddings:
                continue
            info = result_obj.dag_node_to_missing_embeddings[node]
            element = html.Div([
                html.H4(f"{check}", className="result-item-header"),
                html.Div(info.missing_embeddings_examples,
                         className="result-item-content"),
            ], className="result-item")
            details += [element]
        else:
            print("check not implemented:", check)

    return details


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    debug = "DEBUG" in os.environ
    app.run_server(host="0.0.0.0", debug=debug)
