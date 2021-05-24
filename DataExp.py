import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns
import Calculation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import GenNorm
import Terminal
import AxesFrame
import Visualization


fig, ((ax1,ax2),(ax8, ax9)) = plt.subplots(2,2)
fig2, (ax3, ax4) = plt.subplots(2,1)
weak = Terminal.CartesianPoint(5, 5, "BlueTooth", "weak_terminal")
main_axes = AxesFrame.Axes(weak)
main_axes.add_terminal(Terminal.CartesianPoint(12, 3, "UWB", "terminalB"))

# Visualization.draw_compare_error_fix_point(ax1, 5, 3, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
# Visualization.draw_compare_error_fix_point(ax2, 20, 3, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
# Visualization.draw_compare_error_fix_point(ax8, 5, 15, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
# Visualization.draw_compare_error_fix_point(ax9, 20, 15, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
Visualization.draw_compare_error_fix_point(ax3, 20, 10, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 2)
#
Visualization.draw_compare_error_fix_time(ax4, 25, 20, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
# Visualization.draw_compare_error_fix_time(ax5, 3, 20, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
#Visualization.draw_compare_error_fix_time(ax6, 15, 20, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)

Visualization.draw_compare_per_fix_point(ax1, 30, 10, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
Visualization.draw_compare_per_fix_point(ax2, 30, 3, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)

Visualization.draw_compare_per_fix_time(ax8, 30, 15, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)
Visualization.draw_compare_per_fix_time(ax9, 3, 15, main_axes._terminal_set["initiator"], main_axes._terminal_set["terminalB"], main_axes._weak_terminal, 1)


plt.show()