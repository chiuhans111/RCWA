"""
RCWA Package - Rigorous Coupled Wave Analysis for Python.
"""

from .rcwa import Simulation, get_k_from_angle, visualize_field, get_index_contour
from .layer import Layer
from .modes import Modes

__all__ = [
    "Simulation",
    "Layer",
    "Modes",
    "get_k_from_angle",
    "visualize_field",
    "get_index_contour"
]
