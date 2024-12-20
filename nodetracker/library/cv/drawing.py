"""
Support module for image drawing.
"""
import cv2
import numpy as np

from nodetracker.library.cv import color_palette


def draw_text(
    image: np.ndarray,
    text: str,
    left: int,
    top: int,
    color: color_palette.ColorType,
    font_face: int = 1,
    font_scale: int = 1,
    thickness: int = 1,
    border_color: color_palette.ColorType = color_palette.BLACK,
    border_thickness: int = 2
) -> np.ndarray:
    """
    Draws text with border. With this heuristic it is easier to see image anywhere.

    Args:
        image: Image
        text: Text
        left: Text left position
        top: Text top position
        color: Text color
        font_face: Font face
        font_scale: Font scale
        thickness: Text thickness
        border_color: Text border color
        border_thickness: Text border thickness

    Returns:
        Image with text
    """
    # noinspection PyUnresolvedReferences
    image = cv2.putText(image, text, (left, top), font_face, font_scale, border_color, border_thickness)
    # noinspection PyUnresolvedReferences
    return cv2.putText(image, text, (left, top), font_face, font_scale, color, thickness)