"""
Odin astrophotography image processing package.

This package provides various tools for astrophotography, including image enhancement,
filtering, and utility functions.
"""

from .image_enhancement import contrast, color_balance, histogram, sharpen
from .image_filtering import blur, denoising, edge_detection, morphological_operations, rescaling
from .utils import plotting

__all__ = [
    'contrast',
    'color_balance',
    'histogram',
    'sharpen',
    'blur',
    'denoising',
    'edge_detection',
    'morphological_operations',
    'rescaling',
    'plotting'
]
