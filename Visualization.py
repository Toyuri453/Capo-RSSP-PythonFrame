import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import Calculation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import GenNorm
import Terminal
import AxesFrame
import random

def get_random_hex_color() -> 'str':
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def draw_variance_demo():
    x = np.linspace(0, 30, 100)
    ax1.plot(x, (Calculation.get_sigma(x)**2), label = "UWB", color = "blue")
    ax1.plot(x, (Calculation.get_sigma(x, "BlueTooth")**2), label="BlueTooth", color = 'red')
    ax1.legend()
    ax1.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax1.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax1.set_xlabel("Distance(cm)", fontsize=12)
    ax1.set_ylabel("Sigma", fontsize=12)
    ax1.set_title("Sigma Curve", fontsize=12, fontweight='light')

def draw_distribution_demo_by_distance(distance: 'float'):
    mu = 0
    UWB_sigma = Calculation.get_sigma(distance, "UWB")
    BT_sigma = Calculation.get_sigma(distance, "BlueTooth")
    #sigma = math.sqrt(variance)
    x = np.linspace(-10, 10, 100)
    ax2.plot(x, stats.norm.pdf(x, mu, UWB_sigma), label = "UWB", color = "blue")
    ax2.plot(x, stats.norm.pdf(x, mu, BT_sigma), label = "BlueTooth", color = 'red')
    ax2.legend()
    ax2.xaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='major')
    ax2.set_xlabel("Error Distance(cm)", fontsize=12)
    ax2.set_ylabel("Probability(%)", fontsize=12)
    ax2.set_title("Point Distribution(10cm)", fontsize=12, fontweight='light')

def draw_main_axes(main_axes):
    x = np.linspace(0, 50, 100)
    #ax3.scatter(main_axes._initiator._x, main_axes._initiator._x, None, 'blue')
    #ax3.scatter(main_axes._weak_terminal._x, main_axes._weak_terminal._y, None, 'red')
    # normal_set = GenNorm.gen_normal_point(50, main_axes._initiator, main_axes._weak_terminal)
    # for x in range(normal_set.shape[0]):
    #     ax3.scatter(normal_set[x, 0], normal_set[x, 1], c = 'g', s = 50*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], main_axes._weak_terminal), alpha=0.3)
    #     print("[Data] NormalPoint:" + str(10*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], main_axes._weak_terminal)))
    # draw_scatter_by_terminal(ax3, 50, main_axes._initiator, main_axes._weak_terminal)
    ax3.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax3.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax3.set_xlabel("X axis(cm)", fontsize=12)
    ax3.set_ylabel("Y axis(cm)", fontsize=12)
    ax3.set_title("Measuring Distribution", fontsize=12, fontweight='light')

def draw_terminal_circle(ax, terminal : 'Terminal.CartesianPoint', color : 'str', alpha : 'float'):
    out_area = [250]
    inner_area = [65]
    ax.scatter(terminal._x, terminal._y, s=out_area, color = color, alpha=alpha, edgecolors=color, linewidths=[1])
    ax.scatter(terminal._x, terminal._y, s=inner_area, color=color, edgecolors=color, label = terminal._terminal_name)
    ax.legend()

def draw_scatter_by_terminal(ax, number : 'int', terminal : 'Terminal.CartesianPoint', target_terminal : 'Terminal.CartesianPoint', edgecolor : 'str' = 'blue'):
    normal_set = GenNorm.gen_normal_point(number, terminal, target_terminal)
    for x in range(normal_set.shape[0]):
        ax.scatter(normal_set[x, 0], normal_set[x, 1], c = 'slateblue', s = 50*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], target_terminal), alpha=0.3, edgecolors=edgecolor, linewidths=[1])
        print("[Data] Distance:" + str(10*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], target_terminal)))

def draw_terminal_circle_all(ax, main_axes :'AxesFrame.Axes'):
    out_area = [250]
    inner_area = [65]
    for key in main_axes._terminal_set:
        ax.scatter(main_axes._terminal_set[key]._x, main_axes._terminal_set[key]._y, s=out_area, color = "crimson", alpha=0.4, linewidths=[1])
        ax.scatter(main_axes._terminal_set[key]._x, main_axes._terminal_set[key]._y, s=inner_area, label=main_axes._terminal_set[key]._terminal_name)
    ax.legend()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
weak = Terminal.CartesianPoint(5, 5, "BlueTooth", "weak_terminal")
main_axes = AxesFrame.Axes(weak)
draw_variance_demo()
draw_distribution_demo_by_distance(10)
draw_main_axes(main_axes)
# draw_terminal_circle(ax3, main_axes._initiator, 'blue', 0.3)
# draw_terminal_circle(ax3, main_axes._weak_terminal, 'red', 0.3)
draw_scatter_by_terminal(ax3, 20, main_axes._initiator, main_axes._weak_terminal)
main_axes.add_terminal(Terminal.CartesianPoint(10, 3, "UWB", "terminalB"))
draw_scatter_by_terminal(ax3, 10, main_axes._terminal_set["terminalB"], main_axes._weak_terminal)
draw_terminal_circle_all(ax3, main_axes)

plt.show()