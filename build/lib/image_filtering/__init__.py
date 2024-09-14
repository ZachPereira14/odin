"""
image_filtering module.

This module provides functions for filtering images, including blurring, denoising,
edge detection, morphological operations, and rescaling.
"""

from .blur import gaussian_blur
from .denoising import denoising
from .edge_detection import edge_detection_func
from .morphological_operations import morphological_operations_func
from .rescaling import rescaling_func

__all__ = [
    'gaussian_blur',
    'denoising',
    'edge_detection_func',
    'morphological_operations_func',
    'rescaling_func'
]
