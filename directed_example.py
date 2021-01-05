#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:45:51 2020

@author: aransil
"""

import plotly.graph_objects as go
import networkx as nx
import dash
import dash_core_components as dcc
import dash_html_components as html
from addEdge import add_edge

# Controls for how the graph is drawn
nodeColor = 'Blue'
nodeSize = 20
lineWidth = 2
lineColor = '#000000'

# Make a random graph using networkx
G = nx.random_geometric_graph(5, .5)
pos = nx.layout.spring_layout(G)
for node in G.nodes:
    G.nodes[node]['pos'] = list(pos[node])
    
# Make list of nodes for plotly
node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    
# Make a list of edges for plotly, including line segments that result in arrowheads
edge_x = []
edge_y = []
for edge in G.edges():
    # addEdge(start, end, edge_x, edge_y, lengthFrac=1, arrowPos = None, arrowLength=0.025, arrowAngle = 30, dotSize=20)
    start = G.nodes[edge[0]]['pos']
    end = G.nodes[edge[1]]['pos']
    edge_x, edge_y = add_edge(start, end, edge_x, edge_y, .8, 'end', .04, 30, nodeSize)
    

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=lineWidth, color=lineColor), hoverinfo='none', mode='lines')


node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(showscale=False, color = nodeColor, size=nodeSize))

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
             
# Note: if you don't use fixed ratio axes, the arrows won't be symmetrical
fig.update_layout(yaxis = dict(scaleanchor = "x", scaleratio = 1), plot_bgcolor='rgb(255,255,255)')
    
app = dash.Dash()
app.layout = html.Div([dcc.Graph(figure=fig)])

app.run_server(host="0.0.0.0", debug=True, use_reloader=False)


