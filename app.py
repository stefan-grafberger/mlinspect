import os
import time

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

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
            # dcc.Graph(id="dag"),
            html.Img(id="dag", hidden=True, className="mb-3", style={"width": "450px"}),
        ], width=6),
    ]),
])

# Flask server (for gunicorn)
server = app.server


@app.callback([
        Output("dag", "src"),
        Output("dag", "hidden"),
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
        return dash.no_update

    extracted_dag, _, _ = extract_dag(pipeline, checks, inspections)

    # TODO: Take advantage of plotly and do not save image file
    # filename = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))
    filename = os.path.join(os.getcwd(), app.get_asset_url("image.png"))
    print("Saving DAG to image filename:", filename)
    save_fig_to_path(extracted_dag, filename)
    return filename, False


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


def networkx2plotly(dag):
    """Convert networkx.DiGraph to a plotly graph object."""
    pass


if __name__ == "__main__":
    # Disable TensorFlow warnings in logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Run Dash server
    app.run_server(host="0.0.0.0", debug=True)
