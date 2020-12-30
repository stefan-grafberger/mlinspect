#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:45:07 2020

@author: aransil
"""

import math
import numpy as np


def add_edge(start, end, edge_x, edge_y, length_frac=1, arrow_pos=None, arrow_length=0.025, arrow_angle=30, dot_size=20):
    """
    Start and end are lists defining start and end points.
    Edge x and y are lists used to construct the graph.
    arrow_angle and arrow_length define properties of the arrowhead.
    arrow_pos is None, 'middle' or 'end' based on where on the edge you want the arrow to appear.
    arrow_length is the length of the arrowhead.
    arrow_angle is the angle in degrees that the arrowhead makes with the edge.
    dot_size is the plotly scatter dot size you are using (used to even out line spacing when you have a mix of edge lengths).

    Source: https://github.com/redransil/plotly-dirgraph
    Note: length_frac should not really be a fraction of the whole length but rather based on dot_size?
    """

    # Get start and end cartesian coordinates
    x0, y0 = start
    x1, y1 = end

    # Incorporate the fraction of this segment covered by a dot into total reduction
    # length = math.sqrt( (x1-x0)**2 + (y1-y0)**2 )
    dot_size_conversion = .0565/20  # length units per dot size
    converted_dot_diameter = dot_size * dot_size_conversion
    # length_frac_reduction = converted_dot_diameter / length
    length_frac -= converted_dot_diameter

    # If the line segment should not cover the entire distance, get actual start and end coords
    skipX = (x1-x0)*(1-length_frac)
    skipY = (y1-y0)*(1-length_frac)
    x0 = x0 + skipX/2
    x1 = x1 - skipX/2
    y0 = y0 + skipY/2
    y1 = y1 - skipY/2

    # Append line corresponding to the edge
    # Adding None prevents a line being drawn from end of this edge to start of next edge
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

    # Draw arrow
    if arrow_pos:
        print("---> Drawing an arrow!")
        # Find the point of the arrow; assume is at end unless told middle
        pointx = x1
        pointy = y1
        eta = math.degrees(math.atan((x1-x0)/(y1-y0)))

        if arrow_pos in ['middle', 'mid']:
            pointx = x0 + (x1-x0)/2
            pointy = y0 + (y1-y0)/2

        # Find the directions the arrows are pointing
        signx = np.sign(x1-x0)
        if signx == 0:
            print("signx == 0!")
            signx = 0.000001
        signy = np.sign(y1-y0)
        multiplier = signx**2 * signy

        # Append first arrowhead
        dx = arrow_length * math.sin(math.radians(eta + arrow_angle))
        dy = arrow_length * math.cos(math.radians(eta + arrow_angle))
        print(f"-> First dx and dy: {dx} and {dy}")
        edge_x += [pointx, pointx + dx, None]
        edge_y += [pointy, pointy + dy, None]

        # And second arrowhead
        dx = arrow_length * math.sin(math.radians(eta - arrow_angle))
        dy = arrow_length * math.cos(math.radians(eta - arrow_angle))
        print(f"-> Second dx and dy: {dx} and {dy}")
        edge_x += [pointx, pointx + dx, None]
        edge_y += [pointy, pointy + dy, None]

    return edge_x, edge_y
