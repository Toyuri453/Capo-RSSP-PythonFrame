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

def draw_variance_demo():
    x = np.linspace(0, 50, 100)
    ax1.plot(x, (Calculation.get_sigma(x)**2), label = "UWB", color = "blue")
    ax1.plot(x, (Calculation.get_sigma(x, "BlueTooth")**2), label="BlueTooth", color = 'red')
    ax1.legend()
    ax1.xaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='major')
    ax1.set_xlabel("Distance(cm)", fontsize=12)
    ax1.set_ylabel("Sigma", fontsize=12)
    ax1.set_title("Sigma Curve", fontsize=12, fontweight='light')

def draw_distribution_demo_by_distance(distance : 'float', fname : 'str', sname : 'str', marker : "str"):
    mu = 0
    UWB_sigma = Calculation.get_sigma(distance, "UWB")
    BT_sigma = Calculation.get_sigma(distance, "BlueTooth")
    #sigma = math.sqrt(variance)
    x = np.linspace(-7, 7, 50)
    ax2.plot(x, stats.norm.pdf(x, mu, UWB_sigma), marker=marker, markevery=4,label = fname + " " + str(distance) + "cm", color = 'b')
    ax2.plot(x, stats.norm.pdf(x, mu, BT_sigma), marker=marker, markevery=4, label = sname + " " + str(distance) + "cm", color = 'r')
    ax2.legend()
    ax2.xaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='major')
    ax2.set_xlabel("Error Distance(cm)", fontsize=12)
    ax2.set_ylabel("Probability(%)", fontsize=12)
    #ax2.set_title("Point Distribution(10cm)", fontsize=12, fontweight='light')

def draw_main_axes(ax):
    x = np.linspace(0, 50, 100)
    #ax3.scatter(main_axes._initiator._x, main_axes._initiator._x, None, 'blue')
    #ax3.scatter(main_axes._weak_terminal._x, main_axes._weak_terminal._y, None, 'red')
    # normal_set = GenNorm.gen_normal_point(50, main_axes._initiator, main_axes._weak_terminal)
    # for x in range(normal_set.shape[0]):
    #     ax3.scatter(normal_set[x, 0], normal_set[x, 1], c = 'g', s = 50*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], main_axes._weak_terminal), alpha=0.3)
    #     print("[Data] NormalPoint:" + str(10*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], main_axes._weak_terminal)))
    # draw_scatter_by_terminal(ax3, 50, main_axes._initiator, main_axes._weak_terminal)

    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.set_xlabel("X axis(cm)", fontsize=12)
    ax.set_ylabel("Y axis(cm)", fontsize=12)
    ax.set_title("Measuring Distribution", fontsize=12, fontweight='light')

def draw_terminal_circle(ax, terminal : 'Terminal.CartesianPoint', color : 'str', alpha : 'float'):
    out_area = [250]
    inner_area = [65]
    ax.scatter(terminal._x, terminal._y, s=out_area, color = terminal._terminal_color, alpha=alpha, edgecolors=color, linewidths=[1])
    ax.scatter(terminal._x, terminal._y, s=inner_area, color = terminal._terminal_color, edgecolors=color, label = terminal._terminal_name)
    ax.legend()

def draw_scatter_by_terminal(ax, number : 'int', terminal : 'Terminal.CartesianPoint', target_terminal : 'Terminal.CartesianPoint', normal_set : 'np.array' = None):
    edge = terminal._terminal_color
    if normal_set is None:
        normal_set = GenNorm.gen_normal_point(number, terminal, target_terminal)
    else:
        edge = "red"
    #color = Terminal.get_random_hex_color()
    for x in range(normal_set.shape[0]):
        ax.scatter(normal_set[x, 0], normal_set[x, 1], c = terminal._terminal_color, s = 50*Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], target_terminal), alpha=0.45, edgecolors=edge , linewidths=[1])
        print("[DRAW] Draw Scatter In: (" + str(normal_set[x, 0]) + ", " + str(normal_set[x, 1]) + ") Distance: " + str(Calculation.get_distance_from_weak_terminal_by_coord(normal_set[x, 0], normal_set[x, 1], target_terminal)))
    return normal_set

def draw_scatter_by_set(ax, set_a : 'np.array',type : 'str' = "default"):
    out_area = [220]
    inner_area = [50]
    if type == "ideal":
        ax.scatter(set_a[0], set_a[1], edgecolors="firebrick",linewidths=[1])
    if type == "target":
        #ax.scatter(set_a[0], set_a[1], alpha=0.45, c = "seagreen")
        ax.scatter(set_a[0], set_a[1], s=out_area, color="seagreen", alpha=0.35, edgecolors="r",linewidths=[1])
        ax.scatter(set_a[0], set_a[1], s=inner_area, color="seagreen", alpha=0.55, label="target_point")
    if type == "expect":
        ax.scatter(set_a[0], set_a[1], c="purple", s=[55], alpha=0.55, linewidths=[1])
    ax.legend()

def draw_ideal_scatter_by_set(ax, ideal_set : 'np.array', terminal : 'Terminal.CartesianPoint', target_terminal : 'Terminal.CartesianPoint'):
    '''
    :param ax:
    :param ideal_set:
    :param terminal: only for color
    :param target_terminal: only for area
    :return:
    '''
    ax.scatter(ideal_set[0], ideal_set[1],
               s=50 * Calculation.get_distance_from_weak_terminal_by_coord(ideal_set[0], ideal_set[1],
                                                                           target_terminal), alpha=0.7,
               edgecolors='red', linewidths=[2], c = terminal._terminal_color)

def draw_terminal_circle_all(ax, main_axes :'AxesFrame.Axes'):
    out_area = [250]
    inner_area = [65]
    for key in main_axes._terminal_set:
        ax.scatter(main_axes._terminal_set[key]._x, main_axes._terminal_set[key]._y, s=out_area, color = main_axes._terminal_set[key]._terminal_color, alpha=0.4, linewidths=[1])
        ax.scatter(main_axes._terminal_set[key]._x, main_axes._terminal_set[key]._y, s=inner_area, color = main_axes._terminal_set[key]._terminal_color, label=main_axes._terminal_set[key]._terminal_name)
    ax.legend()

def draw_line_between_terminal(ax, terminalA :'Terminal.CartesianPoint', terminalB :'Terminal.CartesianPoint'):
    #[x1,y1][x2,y2] -> [x1,x2][y1,y2]
    #(terminalA.get_coord_array())[1], (terminalB.get_coord_array())[0] = (terminalA.get_coord_array())[0], (terminalB.get_coord_array())[1]
    #ax3.plot(terminalA.get_coord_array(), terminalB.get_coord_array())
    ax.plot([(terminalA.get_coord_array())[0],(terminalB.get_coord_array())[0]], [(terminalA.get_coord_array())[1],(terminalB.get_coord_array())[1]], color = 'red')
    #print((terminalA.get_coord_array())[1], (terminalB.get_coord_array())[0],(terminalA.get_coord_array())[0], (terminalB.get_coord_array())[1])

def draw_hist_demo(ax, measure_terminal : 'Terminal.CartesianPoint', target_terminal : 'Terminal.CartesianPoint', colormap : 'str' = "viridis"):
    point_number = 1000
    #ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    #ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.set_xlabel("Error Distance(cm)", fontsize=12)
    ax.set_ylabel("Total Number(cm)", fontsize=12)
    ax.set_title('Error Distribution Of {0} Point '.format(point_number), fontsize=12, fontweight='light')
    distance_set = GenNorm.gen_dis_set(point_number, measure_terminal, target_terminal)
    n, bins, patches = ax.hist(distance_set, bins=25, facecolor=Terminal.get_random_hex_color() if colormap == "random" else colormap, edgecolor='#6b6b6b',histtype='stepfilled', linewidth=0.5, alpha=0.30, label=measure_terminal._terminal_name+' ({:.2f} cm)'.format(Calculation.get_distance(measure_terminal, target_terminal)))
    n = n.astype('int')
    #for i in range(len(patches)):
        #patches[i].set_facecolor(plt.cm.viridis(n[i] / max(n)))
        #patches[i].set_facecolor(plt.cm.__getattribute__(colormap)(n[i] / max(n)))
    #sns.distplot(distance_set, ax = ax, color="red", label="Compact")
    ax.legend()

def draw_modified_distance_demo(ax, number : 'int' = 20):
    #normal_set = GenNorm.gen_normal_point(number, main_axes._terminal_set["initiator"], main_axes._weak_terminal)
    terminal = Terminal.CartesianPoint(5, 5)
    set = Calculation.get_modified_distance_by_set_and_type(GenNorm.gen_normal_point(number, main_axes._terminal_set["initiator"], terminal))
    x = np.linspace(0, 10, 100)
    dis_diff = -1
    ax.plot(x, (1 - (Calculation.get_sigma(x, "UWB") ** 3 / 15)), label = "UWB", color = "blue")
    #ax.plot(x, (1 - (Calculation.get_sigma(x, "BlueTooth") ** 3 / 10)), label="BlueTooth", color="red")
    for x in range(set[0].shape[0]):
        ax.scatter(set[0][x], set[0][x]/10+0.2, alpha=0.45, c = "b", label = "Mod Distance" if x == 0 else None)
        ax.scatter(set[1][x], set[0][x]/10+0.2, alpha=0.45, c = "r", label = "Raw Distance" if x == 0 else None)
        dis_diff += set[1][x] - set[0][x]
    #dis_diff /= len(set[0])
    ax.axvline(x = dis_diff/len(set[0]), c='r', ls='-.', lw=2,
               alpha=0.7,
               label="AVG Mod Dis")
    ax.axvline(x = Calculation.get_distance(main_axes._terminal_set["initiator"], terminal), c='purple', ls='--', lw = 2, alpha = 0.7,
               label = "Base Dis")
    ax.legend()
    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.set_xlabel("Distance(cm)", fontsize=12)
    ax.set_ylabel("Distance Ratio", fontsize=12)
    ax.set_title("Modified Distance Curve", fontsize=12, fontweight='light')

def draw_shift_parm_contour_demo(ax):
    x = np.linspace(0, 1, 50)
    y = np.linspace(-90, 90, 40)

    X, Y = np.meshgrid(x, y)
    Z = (X ** 2) * (5*Y / 2)
    con = ax.contourf(X, Y, Z, 50, cmap='RdPu')
    con = ax.contourf(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel("Mod-Dis Ratio", fontsize=12)
    ax.set_ylabel("Angle Diff", fontsize=12)
    fig.colorbar(con, ax=ax)

def draw_compare_error_fix_point(ax, time : 'int', point_number : 'int', terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint',
                       target : 'Terminal.CartesianPoint', step : 'int' = "1"): # 1, same time, diff mpoint
    x = np.linspace(0, time+5, 100)
    y = np.linspace(-5, 5, 200)
    step_count = 1
    former_exp_x = 0
    former_exp_y = 0
    former_opt_x = 0
    former_opt_x = 0
    for _ in range(int(time/step)): #20/1
        exp_dis = 0
        opt_dis = 0
        round_count = 0
        for __ in range(step_count):
            round_count = __ + 1
            setA = GenNorm.gen_normal_point(point_number, terminalA, target)
            setB = GenNorm.gen_normal_point(point_number, terminalB, target)
            exp_dis += (Calculation.get_distance_from_origin_by_set(Calculation.get_ideal_coord_by_set(setA)) + Calculation.get_distance_from_origin_by_set(Calculation.get_ideal_coord_by_set(setB)))/2

            op_set = Calculation.get_optimized_target(terminalA,
                                                      terminalB,
                                                      Calculation.get_ideal_coord_by_set(
                                                          Calculation.get_modified_coord_by_nor_set_and_terminal(setA,
                                                                                                                 terminalA)),
                                                      Calculation.get_ideal_coord_by_set(
                                                          Calculation.get_modified_coord_by_nor_set_and_terminal(setB,
                                                                                                                 terminalB))
                                                      )
            opt_dis += Calculation.get_distance_from_origin_by_set(op_set[2])
        step_count += step
        exp_dis /= round_count
        opt_dis /= round_count
        if round_count == 1 :
            ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", label = "exp")
            ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", label = "opt")
        else:
            ax.scatter(_*step, exp_dis, alpha=0.45, c = "b")
            ax.scatter(_*step, opt_dis, alpha=0.45, c = "r")
    ax.legend()

fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(2,4,figsize=(17,10))
fig1, (ax5,ax8) = plt.subplots(1,2)


weak = Terminal.CartesianPoint(5, 5, "BlueTooth", "weak_terminal")
main_axes = AxesFrame.Axes(weak)

draw_variance_demo()


draw_distribution_demo_by_distance(50, "UWB", "BlueTooth", "v")
draw_distribution_demo_by_distance(10, "UWB", "BlueTooth", "*")
draw_distribution_demo_by_distance(20, "UWB", "BlueTooth", "x")


draw_main_axes(ax3)
main_axes._terminal_set["initiator"]._terminal_color = "#fe5858"
main_axes._terminal_set["weak_terminal"]._terminal_color = "#3968e9"
draw_scatter_by_terminal(ax3, 20, main_axes._initiator, main_axes._weak_terminal)
main_axes.add_terminal(Terminal.CartesianPoint(12, 3, "UWB", "terminalB"))
main_axes._terminal_set["terminalB"]._terminal_color = "#01a79d"
#main_axes.add_terminal(Terminal.CartesianPoint(1, 7, "UWB", "terminalC"))
draw_scatter_by_terminal(ax3, 10, main_axes._terminal_set["terminalB"], main_axes._weak_terminal)
#draw_scatter_by_terminal(ax3, 10, main_axes._terminal_set["terminalC"], main_axes._weak_terminal)
draw_terminal_circle_all(ax3, main_axes)
main_axes.show_terminal_names()
draw_line_between_terminal(ax3, main_axes._initiator, main_axes._terminal_set['terminalB'])
#draw_line_between_terminal(ax3, main_axes._initiator, main_axes._terminal_set['terminalC'])
#draw_line_between_terminal(ax3, main_axes._terminal_set['terminalC'], main_axes._terminal_set['terminalB'])


draw_hist_demo(ax4, Terminal.CartesianPoint(15, 16, "UWB", "terminalE"), main_axes._weak_terminal,"r")
draw_hist_demo(ax4, main_axes._terminal_set["terminalB"], main_axes._weak_terminal,"#7a5199")
draw_hist_demo(ax4, Terminal.CartesianPoint(4, 2, "UWB", "terminalD"), main_axes._weak_terminal,"#2cc66c")


draw_main_axes(ax5)
set1 = draw_scatter_by_terminal(ax5, 3, main_axes._initiator, main_axes._weak_terminal)
set = draw_scatter_by_terminal(ax5, 3, main_axes._terminal_set["terminalB"], main_axes._weak_terminal)
draw_terminal_circle_all(ax5, main_axes)
main_axes.show_terminal_names()
draw_line_between_terminal(ax5, main_axes._initiator, main_axes._terminal_set['terminalB'])
#Calculation.get_modified_distance_by_set_and_type(set)

draw_scatter_by_terminal(ax5, 1, main_axes._terminal_set["terminalB"], main_axes._weak_terminal, Calculation.get_modified_coord_by_nor_set_and_terminal(set, main_axes._terminal_set["terminalB"]))
draw_scatter_by_terminal(ax5, 1, main_axes._terminal_set["initiator"], main_axes._weak_terminal, Calculation.get_modified_coord_by_nor_set_and_terminal(set1, main_axes._terminal_set["initiator"]))

draw_ideal_scatter_by_set(ax5, Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set, main_axes._terminal_set["terminalB"])), main_axes._terminal_set["terminalB"], main_axes._weak_terminal)
draw_ideal_scatter_by_set(ax5, Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set1, main_axes._terminal_set["initiator"])), main_axes._terminal_set["initiator"], main_axes._weak_terminal)
op_set = Calculation.get_optimized_target(main_axes._terminal_set["initiator"],
                                          main_axes._terminal_set["terminalB"],
                                          Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set1, main_axes._terminal_set["initiator"])),
                                          Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set, main_axes._terminal_set["terminalB"]))
                                          )

draw_scatter_by_set(ax5, op_set[0], "ideal")
draw_scatter_by_set(ax5, op_set[1], "ideal")
draw_scatter_by_set(ax5, op_set[2], "target")
draw_scatter_by_set(ax5, Calculation.get_ideal_coord_by_set(set1), "expect")
draw_scatter_by_set(ax5, Calculation.get_ideal_coord_by_set(set), "expect")

draw_modified_distance_demo(ax6, 50)
print(math.degrees(math.acos(1.41421/2)))

draw_shift_parm_contour_demo(ax7)
print(Calculation.get_shift_coord_by_radius_and_degree(([0, 1]), 1, 45))


# circle1 = plt.Circle((0, 0), 1, color='r', fill = False)
# ax8.add_patch(circle1)
# ax8.scatter(1, 0)
# draw_scatter_by_set(ax8, Calculation.get_shift_coord_by_radius_and_degree(([1, 0]), 1, 90, ([2, 0])))
# #ax8.scatter(Calculation.get_shift_coord_by_radius_and_degree(([1, 0]), 0.5, 10, ([0, 0]))[0], Calculation.get_shift_coord_by_radius_and_degree(([1, 0]), 0.5, 10, ([0, 0]))[1])
# ax8.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
# ax8.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
draw_main_axes(ax8)
set1 = draw_scatter_by_terminal(ax8, 7, main_axes._initiator, main_axes._weak_terminal)
set = draw_scatter_by_terminal(ax8, 7, main_axes._terminal_set["terminalB"], main_axes._weak_terminal)
draw_terminal_circle_all(ax8, main_axes)
main_axes.show_terminal_names()
draw_line_between_terminal(ax8, main_axes._initiator, main_axes._terminal_set['terminalB'])
#Calculation.get_modified_distance_by_set_and_type(set)

draw_scatter_by_terminal(ax8, 1, main_axes._terminal_set["terminalB"], main_axes._weak_terminal, Calculation.get_modified_coord_by_nor_set_and_terminal(set, main_axes._terminal_set["terminalB"]))
draw_scatter_by_terminal(ax8, 1, main_axes._terminal_set["initiator"], main_axes._weak_terminal, Calculation.get_modified_coord_by_nor_set_and_terminal(set1, main_axes._terminal_set["initiator"]))

draw_ideal_scatter_by_set(ax8, Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set, main_axes._terminal_set["terminalB"])), main_axes._terminal_set["terminalB"], main_axes._weak_terminal)
draw_ideal_scatter_by_set(ax8, Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set1, main_axes._terminal_set["initiator"])), main_axes._terminal_set["initiator"], main_axes._weak_terminal)
op_set = Calculation.get_optimized_target(main_axes._terminal_set["initiator"],
                                          main_axes._terminal_set["terminalB"],
                                          Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set1, main_axes._terminal_set["initiator"])),
                                          Calculation.get_ideal_coord_by_set(Calculation.get_modified_coord_by_nor_set_and_terminal(set, main_axes._terminal_set["terminalB"]))
                                          )
draw_scatter_by_set(ax8, op_set[0], "ideal")
draw_scatter_by_set(ax8, op_set[1], "ideal")
draw_scatter_by_set(ax8, op_set[2], "target")
draw_scatter_by_set(ax8, Calculation.get_ideal_coord_by_set(set1), "expect")
draw_scatter_by_set(ax8, Calculation.get_ideal_coord_by_set(set), "expect")


print(Calculation.get_degree_diff_by_vector(([1,1]), ([1,-1])))

# print(Calculation.get_distance_from_origin_by_set(Calculation.get_ideal_coord_by_set(GenNorm.gen_normal_point(3, main_axes._terminal_set["initiator"], main_axes._weak_terminal))))
# print(Calculation.get_distance_from_origin_by_set(op_set[2]))


plt.show()