"""Shared utilities for text-only control tasks.

Text controls provide the same information as the image task but as text,
to distinguish perceptual from reasoning failures.
"""

from PIL import Image


def placeholder_image() -> Image.Image:
    """Return a 64x64 white placeholder image (API requires an image)."""
    return Image.new("RGB", (64, 64), "white")
