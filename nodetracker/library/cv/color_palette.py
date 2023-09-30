"""
Color palette
"""
from typing import Tuple

import matplotlib.colors as mcolors

ColorType = Tuple[int, int, int]

# BGR
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 128, 0)
ORANGE = (0, 185, 255)
PINK = (147, 20, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
DARK_GREEN = (0, 204, 0)
PURPLE = (51, 0, 51)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
TEAL = (128, 128, 0)
OLIVE = (0, 128, 128)
NAVY = (128, 0, 0)
LIME = (0, 255, 0)
AQUA = (212, 255, 127)
SKY_BLUE = (235, 206, 136)
INDIGO = (130, 0, 75)
BROWN = (19, 69, 139)
TURQUOISE = (208, 224, 64)
CORAL = (80, 127, 255)
TAN = (140, 180, 210)
VIOLET = (238, 130, 238)


ALL_COLORS = [
    RED, BLUE, GREEN, ORANGE, PINK,
    YELLOW, MAGENTA, WHITE, CYAN, DARK_GREEN,
    PURPLE, BLACK, GRAY, TEAL, OLIVE,
    NAVY, LIME, SKY_BLUE, INDIGO,
    BROWN, TURQUOISE, CORAL, TAN, VIOLET
]


MATPLOTLIB_COLORS = list(mcolors.BASE_COLORS) + list(mcolors.CSS4_COLORS)
