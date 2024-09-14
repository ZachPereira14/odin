"""
image_enhancement module.

This module provides functions for enhancing images, including contrast adjustment,
color balancing, histogram stretching, and sharpening.
"""

from .contrast import contrast_enhancement_func
from .color_balance import color_balance_func
from .histogram import histogram_stretching
from .sharpen import sharpen_image

__all__ = [
    'contrast_enhancement_func',
    'color_balance_func',
    'histogram_stretching',
    'sharpen_image'
]
