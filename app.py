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

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
INSPECTOR_RESULT, POS_DICT = None, None


# Create HTML layout
CODE_FONT = {"font-family": "'Courier New', monospace"}
app.title = "mlinspect"
with open("example_pipelines/healthcare/healthcare.py") as f:
    default_pipeline = f.read()
patients = pd.read_csv("example_pipelines/healthcare/healthcare_patients.csv", na_values='?')
histories = pd.read_csv("example_pipelines/healthcare/healthcare_histories.csv", na_values='?')
data = patients.merge(histories, on=['ssn'])
inspection_switcher = {
    "HistogramForColumns": HistogramForColumns,
    "RowLineage": lambda: RowLineage(5),
    "MaterializeFirstOutputRows": lambda: MaterializeFirstOutputRows(5),
}
check_switcher = {
    "NoBiasIntroducedFor": NoBiasIntroducedFor,
    "NoIllegalFeatures": NoIllegalFeatures,
    "NoMissingEmbeddings": NoMissingEmbeddings,
}
app.layout = dbc.Container([
    # Header and description
    html.H1("mlinspect", style={"font-size": "24px", **CODE_FONT}),
    html.P("Inspect ML Pipelines in Python in the form of a DAG."),

    dbc.Row([
        dbc.Col([
            # Inspection definition
            dbc.Form([
                dbc.FormGroup([
                    # Pipeline definition
                    # dbc.Label("Pipeline definition:", html_for="pipeline"),
                    dbc.Textarea(
                        id="pipeline-textarea",
                        value=default_pipeline,
                        className="mb-3",
                        style={"height": "500px"},
                    ),
                    html.Pre(
                        html.Code(
                            # default_pipeline,
                            id="pipeline-code",
                            # contentEditable="true",
                            hidden=True,
                        ),
                    )
                ]),
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
                            dbc.Checkbox(id="nobiasintroduced-checkbox", className="custom-control-input"),
                            dbc.Label("No Bias Introduced For", html_for="nobiasintroduced-checkbox", className="custom-control-label"),
                            dbc.Checklist(id="sensitive-columns",
                                          options=[{"label": column, "value": column} for column in data.columns],
                                          style={"display": "none"}),
                        ], className="custom-switch custom-control"),
                        html.Div([
                            dbc.Checkbox(id="noillegalfeatures-checkbox", className="custom-control-input"),
                            dbc.Label("No Illegal Features", html_for="noillegalfeatures-checkbox", className="custom-control-label"),
                        ], className="custom-switch custom-control"),
                        html.Div([
                            dbc.Checkbox(id="nomissingembeddings-checkbox", className="custom-control-input"),
                            dbc.Label("No Missing Embeddings", html_for="nomissingembeddings-checkbox", className="custom-control-label"),
                        ], className="custom-switch custom-control"),
                    ], id="checks"),
                ]),
                # Execute inspection
                dbc.Button("Inspect pipeline", id="execute", color="primary", size="lg", className="mr-1"),
            ]),
        ], width=6),
        dbc.Col([
            # Extracted DAG
            # dbc.Label("Extracted DAG:", html_for="dag"),
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
            html.Br(),
            # Inspection details
            dbc.Button("Show first output rows", id="show-outputs", color="primary", size="lg", className="mr-1"),
            html.Div(id="results-detail"),
        ], width=6),
    ]),
], style={"font-size": "14px"})


# Flask server (for gunicorn)
server = app.server


@app.callback(
    [
        Output("pipeline-code", "children"),
        Output("pipeline-code", "hidden"),
        Output("pipeline-textarea", "hidden"),
    ],
    [
        Input("pipeline-textarea", "n_blur"),
        Input("pipeline-code", "n_clicks"),
        Input("execute", "n_clicks"),
    ],
    state=[
        State("pipeline-textarea", "value"),
    ],
)
def toggle_editable(textarea_blur, code_clicks, execute_clicks, pipeline):
    """
    When textarea loses focus, hide textarea and show code instead
    (with children updated with textarea value).

    When user clicks on code, hide code and show textarea instead.
    """
    # user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if textarea_blur or execute_clicks:
        return pipeline, False, True

    if code_clicks:  # Doesn't work :/
        return pipeline, True, False

    return pipeline, dash.no_update, dash.no_update


@app.callback(
    Output("sensitive-columns", "style"),
    Input("nobiasintroduced-checkbox", "checked"),
)
def show_subchecklist(checked):
    """Show checklist of sensitive columns if NoBiasIntroducedFor is selected."""
    if checked:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    [
        Output("dag", "figure"),
        Output("results-detail", "children"),
    ],
    [
        Input("execute", "n_clicks"),
        Input("show-outputs", "n_clicks"),
    ],
    state=[
        State("pipeline-textarea", "value"),
        # State("checks", "value"),
        State("nobiasintroduced-checkbox", "checked"),
        State("sensitive-columns", "value"),
        State("noillegalfeatures-checkbox", "checked"),
        State("nomissingembeddings-checkbox", "checked"),
        State("inspections", "value"),
        State("dag", "figure"),
    ]
)
# def update_outputs(execute_clicks, outputs_clicks, pipeline, checks, inspections, figure):
def update_outputs(execute_clicks, outputs_clicks, pipeline, nobiasintroduced,
                   sensitive_columns, noillegalfeatures, nomissingembeddings,
                   inspections, figure):
    """When user clicks 'execute' button, show extracted DAG and inspection results."""
    user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if not user_click:
        return dash.no_update, dash.no_update

    if user_click == "execute":
        checks = {
            "NoBiasIntroducedFor": (nobiasintroduced, [sensitive_columns]),
            "NoIllegalFeatures": (noillegalfeatures, []),
            "NoMissingEmbeddings": (nomissingembeddings, []),
        }
        execute_inspector_builder(pipeline, checks, inspections)
        figure = nx2go(INSPECTOR_RESULT.dag)
        figure, details = show_distribution_changes(figure, sensitive_columns)
        return figure, details

    elif user_click == "show-outputs":
        figure, details = show_one_hot_encoder_details(figure)
        return figure, details


def execute_inspector_builder(pipeline, checks=None, inspections=None):
    """Extract DAG the original way, i.e. by creating a PipelineInspectorBuilder."""
    global INSPECTOR_RESULT

    start = time.time()
    builder = PipelineInspector.on_pipeline_from_string(pipeline)
    for inspection in inspections:
        builder = builder.add_required_inspection(inspection_switcher[inspection]())
    for check_name, (check_bool, check_args) in checks.items():
        # builder = builder.add_check(check_switcher[check]())
        if check_bool:
            builder = builder.add_check(check_switcher[check_name](*check_args))
    INSPECTOR_RESULT = builder.execute()
    print(f"Total time in seconds: {time.time() - start}")

    # extracted_dag = INSPECTOR_RESULT.dag
    # inspection_results = INSPECTOR_RESULT.inspection_to_annotations
    # check_results = INSPECTOR_RESULT.check_to_check_results
    # return extracted_dag, inspection_results, check_results


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
    # pos_json = {k.node_id: {'pos': v, 'node': k.to_dict()} for k, v in POS_DICT.items()}
    # with open('pos_dict_with_checks.json', 'w') as f:
    #     json.dump(pos_json, f, indent="\t", ensure_ascii=True)

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
            length_frac=0.8,
            arrow_pos='end',
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
            'size': 15,
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
    # Obviously this won't work because it's for Jupyter!
    # fig = go.FigureWidget(data=[edges, nodes], layout=layout)

    # create our callback function
    def update_point(trace, points, selector):
        c = list(nodes.marker.color)
        s = list(nodes.marker.size)
        for i in points.point_inds:
            c[i] = 'red'
            s[i] = 20
            with fig.batch_update():
                nodes.marker.color = c
                nodes.marker.size = s

    nodes.on_click(update_point)

    return fig.to_dict()


def highlight_dag_node_in_figure(dag_node, fig_dict):
    # Create scatter plot of this node
    Xn, Yn = POS_DICT[dag_node]
    label = get_new_node_label(dag_node)
    nodes = go.Scatter(
        x=[Xn], y=[Yn], mode='markers', name='', hoverinfo='text', text=[label],
        marker={
            'size': 15,
            'color': 'red',
            'line': {
                'color': 'red',
                'width': 0.5,
            },
        },
    )

    # Append scatter plot to figure
    fig_dict['data'].append(nodes)  # if it's a dict
    # figure.add_trace(nodes)  # if it's a tuple

    return fig_dict


def show_one_hot_encoder_details(fig_dict):
    try:
        first_rows_inspection_result = INSPECTOR_RESULT.inspection_to_annotations[MaterializeFirstOutputRows(5)]
    except KeyError:
        return fig_dict, []

    # Display first output rows (results of MaterializeFirstOutputRows(5) inspection)
    details = []
    node_list = list(INSPECTOR_RESULT.dag.nodes)
    for idx, desc in [[23, "Input"], [29, "Output"]]:
        node = node_list[idx]
        df = first_rows_inspection_result[node]
        # operator = html.Div(f"{node.operator_type}", style=CODE_FONT)
        # description = html.Div(f"{node.description}", style=CODE_FONT)
        data = df.to_dict('records')
        if idx == 29:
            for record in data:
                record['county'] = np.array2string(record['county'])
        label = dbc.Label(desc, html_for=f"table-{idx}", style=CODE_FONT)
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=data,
            id=f"table-{idx}",
        )
        # details += [html.Br(), operator, description, table]
        details += [html.Br(), label, table]

    dag_node = node_list[29]
    fig_dict = highlight_dag_node_in_figure(dag_node, fig_dict)

    return fig_dict, details


def show_distribution_changes(fig_dict, sensitive_columns):
    """From mlinspect.checks._no_bias_introduced_for:NoBiasIntroducedFor.plot_distribution_change_histograms."""
    try:
        no_bias_check_result = INSPECTOR_RESULT.check_to_check_results[NoBiasIntroducedFor(sensitive_columns)]
    except (KeyError, TypeError):
        return fig_dict, []

    details = []
    for node_dict in no_bias_check_result.bias_distribution_change.values():
        for column, distribution_change in node_dict.items():
            # check if distribution change is acceptable
            if distribution_change.acceptable_change:
                continue

            # create histogram
            keys = distribution_change.before_and_after_df["sensitive_column_value"]
            keys = [str(key) for key in keys]  # Necessary because of null values
            before_values = distribution_change.before_and_after_df["count_before"]
            after_values = distribution_change.before_and_after_df["count_after"]
            trace1 = go.Bar(x=keys, y=before_values, name="Before")
            trace2 = go.Bar(x=keys, y=after_values, name="After")
            details.append(
                dcc.Graph(
                    figure=go.Figure(
                        data=[trace1, trace2],
                        layout_title_text=f"Line {distribution_change.dag_node.code_reference.lineno}, "\
                                          f"Operator '{distribution_change.dag_node.operator_type.value}', "\
                                          f"Column '{column}'",
                    )
                )
            )

            # highlight this node in figure
            fig_dict = highlight_dag_node_in_figure(distribution_change.dag_node, fig_dict)

    return fig_dict, details


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
