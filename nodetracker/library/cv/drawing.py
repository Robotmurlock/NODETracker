import numpy as np
import cv2
from nodetracker.library.cv import color_palette

def draw_text(
    image: np.ndarray,
    text: str,
    left: int,
    top: int,
    color: color_palette.ColorType,
    border_color: color_palette.ColorType = color_palette.BLACK
) -> np.ndarray:
    """
    Draws text with border. With this heuristic it is easier to see image anywhere.

    Args:
        image: Image
        text: Text
        left: Text left position
        top: Text top position
        color: Text color
        border_color: Text border color

    Returns:
        Image with text
    """
    # noinspection PyUnresolvedReferences
    image = cv2.putText(image, text, (left, top), 1, 1, border_color, 2)
    # noinspection PyUnresolvedReferences
    return cv2.putText(image, text, (left, top), 1, 1, color, 1)