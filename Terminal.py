import math
import random
import numpy as np

def get_random_hex_color() -> 'str':
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

class CartesianPoint():
    def __init__(self, Cartx : 'float' = 0.0, Carty : 'float' = 0.0, Terminal_type : 'str' = "UWB", Terminal_name : 'str' = "Terminal"):
        self._x = Cartx
        self._y = Carty
        self._terminal_type = Terminal_type
        self._terminal_name = Terminal_name
        self._terminal_color = get_random_hex_color()

    def get_distance_from_origin(self)-> 'float':
        return math.sqrt(self._x**2 + self._y**2)

    def get_coord_array(self):
        return np.array([self._x, self._y])


class PolarPoint():
    pass