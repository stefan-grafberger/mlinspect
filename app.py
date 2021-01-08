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
                            dbc.Label("Add checks:", html_for="checks"),
                            dbc.Checklist(
                                id="checks",
                                options=[
                                    {"label": "No Bias Introduced For", "value": "NoBiasIntroducedFor"},  # TODO: Sub checklist with data.columns
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
                    # TODO: Add button "EDIT PIPELINE" to go back to invisible first tab?
                    # html.P(
                    #     id="pipeline-output",
                    #     className="mb-3",
                    #     style={
                    #         **CODE_FONT,
                    #         "font-size": "12px",
                    #         "white-space": "pre-line",
                    #     },
                    #     children=default_pipeline,
                    # ),
                    dcc.Markdown("""
                        Pipeline definition:
                        ```python
                        {}
                        ```
                        """.format(default_pipeline), id="pipeline-output",
                        dangerously_allow_html=True,
                        style={
                            "background": "white",
                            # "pre": {
                            #     ".line-number": {
                            #         "display": "block",
                            #         "float": "left",
                            #         "margin": "0 1em 0 -1em",
                            #         "border-right": "1px solid #ddd",
                            #         "text-align": "right",
                            #         "span": {
                            #             "display": "block",
                            #             "padding": "0 .5em 0 1em",
                            #             "color": "#ccc",
                            #         },
                            #     },
                            # },
                        }),
                ], width=6),
                dbc.Col([
                    # Display DAG
                    dbc.Label("Extracted DAG:", html_for="dag"),
                    dcc.Graph(id="dag", figure=go.Figure(
                        # layout_width=650,
                        layout_height=650,
                        layout_showlegend=False,
                        layout_xaxis={'visible': False},
                        layout_yaxis={'visible': False},
                    )),
                    html.Br(),
                    # dbc.Button("Show first output rows", id="show-outputs", color="primary", size="lg", className="mr-1"),
                    # dbc.Button("Show histograms", id="show-histograms", color="primary", size="lg", className="mr-1"),
                    # TODO: Maybe even tabs for different details, first output rows vs. histograms, instead of one button
                    html.Div(id="results-detail"),
                ], width=6),
            ], style={"margin": "auto", "padding": "20px", "font-size": "12px"}),
        ], label="INSPECTION RESULTS", value="results-tab"),
    ], id="tabs", style={"display": "none"}),
], style={"font-size": "14px"})


# Flask server (for gunicorn)
server = app.server


@app.callback(
    [
        Output("tabs", "value"),
        Output("pipeline-output", "children"),
        Output("dag", "figure"),
        Output("results-detail", "children"),
    ],
    [
        Input("execute", "n_clicks"),
        Input("dag", "n_clicks"),
        # Input("show-outputs", "n_clicks"),
        # Input("show-histograms", "n_clicks"),
    ],
    state=[
        State("pipeline", "value"),
        State("checks", "value"),
        State("inspections", "value"),
        State("dag", "figure"),
    ]
)
def update_outputs(execute_clicks, dag_clicks, pipeline, checks, inspections, figure):
    """Dash callback function to show extracted DAG of ML pipeline."""
    user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if not user_click:
        return "definition-tab", dash.no_update, dash.no_update, dash.no_update

    active_tab = "results-tab"
    pipeline_output = """
```python
{}
```
""".format(pipeline)

    if user_click == "execute":
        execute_inspector_builder(pipeline, checks, inspections)
        figure = nx2go(INSPECTOR_RESULT.dag)
        details = []
        return active_tab, pipeline_output, figure, details

    elif user_click == "dag":
        # Update figure with highlighted (red) nodes
        figure, output_rows_results = materialize_first_output_rows(figure)

        # Display first output rows (results of MaterializeFirstOutputRows(5) inspection)
        details, code_linenos = [], []
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
            )
            details += [html.Br(), description, table]
            code_linenos += [node.code_reference.lineno]

        # Highlight problematic code
        lines = pipeline.splitlines(keepends=True)
        for lineno in code_linenos:
            print("Formatting code line:", lineno)
            lines[lineno-1] = "<b>" + lines[lineno-1] + "</b>"
        pipeline_output = """
<pre>
```python
{}
```
</pre>
""".format("".join(lines))

        return active_tab, pipeline_output, figure, details

    elif user_click == "show-histograms":
        details = show_distribution_changes()
        return active_tab, pipeline_output, figure, details


def execute_inspector_builder(pipeline, checks=None, inspections=None):
    """Extract DAG the original way, i.e. by creating a PipelineInspectorBuilder."""
    global INSPECTOR_RESULT

    start = time.time()
    builder = PipelineInspector.on_pipeline_from_string(pipeline)
    for inspection in inspections:
        builder = builder.add_required_inspection(inspection_switcher[inspection]())
    for check in checks:
        builder = builder.add_check(check_switcher[check]())
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
            'color': '#85b6b6',
            'line': {
                'color': 'rgb(100,100,100)',
                'width': 0.5,
            },
        },
    )
    layout = go.Layout(
        # title="Pipeline execution DAG",
        # font={'family': 'Balto'},
        # font={'family': "'Courier New', monospace"},
        font={'family': "Courier New"},
        # width=650,
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
        plot_bgcolor='rgb(255,255,255)',
    )
    layout.annotations = annotations

    fig = go.Figure(data=[edges, nodes], layout=layout)

    return fig


def materialize_first_output_rows(figure):
    try:
        first_rows_inspection_result = INSPECTOR_RESULT.inspection_to_annotations[MaterializeFirstOutputRows(5)]
    except KeyError:
        return figure, []

    relevant_nodes = [node for node in INSPECTOR_RESULT.dag.nodes if node.description in {
        "Imputer (SimpleImputer), Column: 'county'", "Categorical Encoder (OneHotEncoder), Column: 'county'"}]

    # Create scatter plot of these nodes
    Xn, Yn, labels, results = [], [], [], []
    for dag_node in relevant_nodes:
        if dag_node in first_rows_inspection_result and first_rows_inspection_result[dag_node] is not None:
            x, y = POS_DICT[dag_node]
            Xn += [x]
            Yn += [y]
            labels += [get_new_node_label(dag_node)]
            results += [(dag_node, first_rows_inspection_result[dag_node])]
    nodes = go.Scatter(
        x=Xn, y=Yn, mode='markers', name='', hoverinfo='text', text=labels,
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
    figure['data'].append(nodes)

    return figure, results


def show_distribution_changes():
    """From mlinspect.checks._no_bias_introduced_for:NoBiasIntroducedFor.plot_distribution_change_histograms."""
    # --- distribution changes
    no_bias_check_result = INSPECTOR_RESULT.check_to_check_results[NoBiasIntroducedFor(["age_group", "race"])]
    dag_node_distribution_changes_list = list(no_bias_check_result.bias_distribution_change.items())

    # --- histograms
    _, join_distribution_changes = dag_node_distribution_changes_list[0]
    graphs = []
    for column, distribution_change in join_distribution_changes.items():
        keys = distribution_change.before_and_after_df["sensitive_column_value"]
        keys = [str(key) for key in keys]  # Necessary because of null values
        before_values = distribution_change.before_and_after_df["count_before"]
        after_values = distribution_change.before_and_after_df["count_after"]

        # --- plotly figure
        # --- (with subplots)
        # figure = make_subplots(rows=1, cols=2, subplot_titles=["before", "after"])
        # figure.append_trace(go.Bar(x=keys, y=before_values), 1, 1)
        # figure.append_trace(go.Bar(x=keys, y=after_values), 1, 1)
        # figure.update_layout(title_text=f"Column '{column}'")
        # --- (without subplots)
        trace1 = go.Bar(x=keys, y=before_values, name="Before")
        trace2 = go.Bar(x=keys, y=after_values, name="After")
        graphs += [dcc.Graph(figure=go.Figure(data=[trace1, trace2], layout=go.Layout(title_text=f"Column '{column}'")))]

    return graphs


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
