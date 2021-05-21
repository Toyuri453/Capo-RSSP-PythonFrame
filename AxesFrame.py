import Terminal

class Axes():
    def __init__(self, weak_terminal : 'Terminal.CartesianPoint'):
        # self._initiator_x = weak_terminal._x
        # self._initiator_y = weak_terminal._y
        self._initiator = Terminal.CartesianPoint(0.0, 0.0, "UWB", "initiator")
        self._weak_terminal = weak_terminal
        self._terminal_set = {self._initiator._terminal_name : self._initiator, self._weak_terminal._terminal_name : self._weak_terminal}
        self._terminal_measuring_point_set = {'Set' : {}} #Fill Later
        print(self._terminal_set)

    def add_terminal(self, terminal : 'Terminal.CartesianPoint'):
        self._terminal_set[terminal._terminal_name] = terminal

    def show_terminal_names(self):
        for key in self._terminal_set:
            print("[DATA] Terminal Name: {0}, Color: {1}".format(key, self._terminal_set[key]._terminal_color))