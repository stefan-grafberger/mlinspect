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
    # Link taken from https://highlightjs.org/download/
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.5.0/styles/default.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.5.0/styles/a11y-dark.min.css",
    # jquery
    "https://code.jquery.com/jquery-3.5.1.min.js"
    # dbc.themes.GRID,  # pro: grid layouts, large enough font size; con: no other dbc elements or CSS classes
],  # Link taken from https://highlightjs.org/download/
   external_scripts=["//cdnjs.cloudflare.com/ajax/libs/highlight.js/10.5.0/highlight.min.js",
                     "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.5.0/languages/python.min.js"])
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
    html.Div([html.Pre([html.Code(["print('hello world')"], className="Python")])], id="highlightjs-test",),

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
                ]),
                # html.Pre(
                    dcc.Markdown(
                        id="pipeline-md",
                        style={"display": "none"},
                        dangerously_allow_html=True,  # to enable <b> tags for highlighting source code sections
                    ),
                # ),
                dbc.Button("Edit pipeline", id="edit", color="primary", size="lg", className="mr-1"),
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
            html.Div(id="first-outputs"),
            html.Div(id="problems"),
            html.Div([
                html.H4("Click data"),
                html.Div(id="click-data")
            ]),
        ], width=6),
    ]),
], style={"font-size": "14px"})


# Flask server (for gunicorn)
server = app.server


@app.callback(
    [
        # Output("pipeline-md", "children"),
        Output("pipeline-md", "style"),
        Output("pipeline-textarea", "hidden"),
    ],
    [
        Input("pipeline-textarea", "n_blur"),
        Input("edit", "n_clicks"),
        Input("execute", "n_clicks"),
    ],
    state=[
        State("pipeline-textarea", "value"),
    ],
)
def toggle_editable(textarea_blur, edit_clicks, execute_clicks, pipeline):
    """
    When textarea loses focus or when user clicks execute button,
    hide textarea and show markdown instead.
    Handle update of markdown content in main callback instead of here.

    When user clicks on edit button, hide markdown and show textarea instead.
    """
    user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if user_click == "edit":
        md_style = {"display": "none"}
        hide_textarea = False
        return md_style, hide_textarea

    if user_click == "execute" or textarea_blur:
        md_style = {"display": "block", "background": "#ffffff"}
        hide_textarea = True
        return md_style, hide_textarea

    return dash.no_update, dash.no_update


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
    Output("click-data", "children"),
    Input("dag", "clickData"),
)
def on_graph_click(click_data):
    """React on graph clicks."""
    data1 = {
        "points": [
            {
                "curveNumber": 1,
                "pointNumber": 24,
                "pointIndex": 24,
                "x": 10153,
                "y": 428.07,
                "text": "Transformer (L47)\nImputer (SimpleImputer), Column: 'race'",
            },
        ],
    }
    data2 = {
        "points": [
            {
                "curveNumber": 1,
                "pointNumber": 29,
                "pointIndex": 29,
                "x": 6131,
                "y": 315.38,
                "text": "Transformer (L48)\nCategorical Encoder (OneHotEncoder), Column: 'county'",
            },
        ],
    }

    return json.dumps(click_data, indent=2)


@app.callback(
    [
        Output("dag", "figure"),
        Output("first-outputs", "children"),
        Output("problems", "children"),
        Output("pipeline-md", "children"),
    ],
    [
        Input("execute", "n_clicks"),
        Input("dag", "clickData"),
        Input("pipeline-textarea", "n_blur"),
    ],
    state=[
        State("pipeline-textarea", "value"),
        State("nobiasintroduced-checkbox", "checked"),
        State("sensitive-columns", "value"),
        State("noillegalfeatures-checkbox", "checked"),
        State("nomissingembeddings-checkbox", "checked"),
        State("inspections", "value"),
        State("dag", "figure"),
    ]
)
def update_outputs(execute_clicks, graph_click_data, textarea_blur, pipeline,
                   nobiasintroduced, sensitive_columns, noillegalfeatures,
                   nomissingembeddings, inspections, figure):
    """
    When user clicks 'execute' button, show extracted DAG and automatically flag
    potential problems: highlight DAG node, highlight source code, and show
    histograms of distribution changes.

    When textarea loses focus or when user clicks execute button,
    update markdown content with value from textarea instead.
    Handle toggling between textarea and code in separate callback.

    When user clicks on DAG node, output first rows of this operator.
    """
    user_click = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if not user_click:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if user_click == "execute":
        # Execute pipeline and inspections
        checks = {
            "NoBiasIntroducedFor": (nobiasintroduced, [sensitive_columns]),
            "NoIllegalFeatures": (noillegalfeatures, []),
            "NoMissingEmbeddings": (nomissingembeddings, []),
        }
        execute_inspector_builder(pipeline, checks, inspections)

        # Convert extracted DAG into plotly figure
        figure = nx2go(INSPECTOR_RESULT.dag)

        # Highlight problematic nodes and show histograms of distribution changes
        figure, problems, pipeline_md = show_distribution_changes(figure, sensitive_columns, pipeline)

        return figure, problems, dash.no_update, pipeline_md

    if user_click == "dag":
        # Output first rows and highlight code
        figure, output_rows, pipeline_md = show_one_hot_encoder_details(figure, graph_click_data, pipeline)

        return figure, dash.no_update, output_rows, pipeline_md

    if textarea_blur:
        # Simply update the markdown content from the textarea
        pipeline_md = convert_to_markdown(pipeline, add_line_numbers=True)

        return dash.no_update, dash.no_update, dash.no_update, pipeline_md


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

    return fig.to_dict()


def highlight_dag_node_in_figure(dag_node, fig_dict):
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
    fig_dict['data'].append(nodes)  # if it's a dict
    # figure.add_trace(nodes)  # if it's a tuple

    return fig_dict


def convert_to_markdown(pipeline, add_line_numbers=True, emphasis=[]):
    lines = pipeline.splitlines(keepends=True)

    # Add line numbers
    if add_line_numbers:
        for idx, line in enumerate(lines):
            lineno = idx + 1  # because line numbers should have one-based indexing
            line = f"{lineno: >3}. {line}"

    # Add emphasis on certain line(s)
    for lineno in emphasis:
        idx = lineno - 1  # due to zero-based python array indexing
        lines[idx] = f"<b>{lines[idx]}</b>"

    pipeline = "".join(lines)

    # Convert to markdown code block
    pipeline_md = """
```python
{}
```
""".format(pipeline)
    if emphasis:
        pipeline_md = f"<pre>{pipeline_md}</pre>"

    return pipeline_md


def show_one_hot_encoder_details(fig_dict, graph_click_data, pipeline):
    try:
        first_rows_inspection_result = INSPECTOR_RESULT.inspection_to_annotations[MaterializeFirstOutputRows(5)]
    except KeyError:
        return fig_dict, []

    # TODO: Actually use graph_click_data

    # Display first output rows (results of MaterializeFirstOutputRows(5) inspection)
    details = [html.H4("First output rows")]
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

    # highlight relevant lines in code
    pipeline_md = convert_to_markdown(pipeline, add_line_numbers=True, emphasis=[47, 48])

    return fig_dict, details, pipeline_md


def show_distribution_changes(fig_dict, sensitive_columns, pipeline):
    """From mlinspect.checks._no_bias_introduced_for:NoBiasIntroducedFor.plot_distribution_change_histograms."""
    try:
        no_bias_check_result = INSPECTOR_RESULT.check_to_check_results[NoBiasIntroducedFor(sensitive_columns)]
    except (KeyError, TypeError):
        return fig_dict, []

    details = [html.H4("Problematic distribution changes")]
    code_linenos = []
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

            code_linenos += [distribution_change.dag_node.code_reference.lineno]
    # highlight relevant lines in code
    pipeline_md = convert_to_markdown(pipeline, add_line_numbers=True, emphasis=code_linenos)

    return fig_dict, details, pipeline_md


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
