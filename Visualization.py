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
from scipy.interpolate import UnivariateSpline

def draw_compare_error_fix_point(ax, time : 'int', point_number : 'int', terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint',
                       target : 'Terminal.CartesianPoint', step : 'int' = "1"): # 1, same time, diff mpoint
    x = np.linspace(0, time+5, 100)
    y = np.linspace(-5, 5, 200)
    step_count = 1
    # former_exp_x = 0
    # former_exp_y = 0
    # former_opt_x = 0
    # former_opt_y = 0
    former_exp = np.array([0,0])
    former_opt = np.array([0,0])
    exp_sum = 0
    opt_sum = 0
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
        exp_sum += exp_dis
        opt_sum += opt_dis
        if _ == 0 :
            ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", label = "exp_err")
            ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", label = "opt_err", marker = "^")
        else:
            ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", s = [60])
            ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", s = [60], marker = "^")

            ax.plot(np.array([former_exp[0], _*step]), np.array([former_exp[1], exp_dis]), c="b", ls = "--")
            ax.plot(np.array([former_opt[0], _*step]), np.array([former_opt[1], opt_dis]), c="r")
        former_exp = np.array([_*step, exp_dis])
        former_opt = np.array([_*step, opt_dis])
    ax.axhline(y = exp_sum/(time/step), c='b', ls='--', lw=2,
               alpha=0.7,
               label="avr_exp_err")
    ax.axhline(y = opt_sum/(time/step), c='r', ls='-', lw=2,
               alpha=0.7,
               label="avr_opt_err")
    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    #ax.set_xlabel("Measure Round", fontsize=12)
    ax.set_ylabel("Error Distance", fontsize=12)
    ax.set_title("round: " + str(time) + " point: " + str(point_number) + " step: " + str(step))
    ax.legend()

def draw_compare_error_fix_time(ax, time : 'int', point_number : 'int', terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint',
                       target : 'Terminal.CartesianPoint', step : 'int' = "1"):
    x = np.linspace(0, time+5, 100)
    y = np.linspace(-5, 5, 200)
    step_count = 1
    # former_exp_x = 0
    # former_exp_y = 0
    # former_opt_x = 0
    # former_opt_y = 0
    former_exp = np.array([0,0])
    former_opt = np.array([0,0])
    exp_sum = 0
    opt_sum = 0
    for _ in range(int(point_number/step)): #20/1
        exp_dis = 0
        opt_dis = 0
        #round_count = 0
        for __ in range(time):
            #round_count = __ + 1
            setA = GenNorm.gen_normal_point(_+1, terminalA, target)
            setB = GenNorm.gen_normal_point(_+1, terminalB, target)
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
        exp_dis /= time
        opt_dis /= time
        exp_sum += exp_dis
        opt_sum += opt_dis
        if _ == 0 :
            ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", label = "exp_err")
            ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", label = "opt_err", marker = "^")
        else:
            ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", s = [60])
            ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", s = [60],marker = "^")

            ax.plot(np.array([former_exp[0], _*step]), np.array([former_exp[1], exp_dis]), c="b", ls = "--")
            ax.plot(np.array([former_opt[0], _*step]), np.array([former_opt[1], opt_dis]), c="r")
        former_exp = np.array([_*step, exp_dis])
        former_opt = np.array([_*step, opt_dis])
    #ax.axhline(y = exp_sum/(time/step), c='b', ls='--', lw=2,alpha=0.7,label="avr_exp_err")
    #ax.axhline(y = opt_sum/(time/step), c='r', ls='-', lw=2,alpha=0.7,label="avr_opt_err")
    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    #ax.set_xlabel("Measure Number", fontsize=12)
    ax.set_ylabel("Error Distance", fontsize=12)
    ax.set_title("round: " + str(time) + " point: " + str(point_number) + " step: " + str(step))
    ax.legend()

def draw_compare_per_fix_point(ax, time : 'int', point_number : 'int', terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint',
                       target : 'Terminal.CartesianPoint', step : 'int' = "1"): # 1, same time, diff mpoint
    x = np.linspace(0, time+1, 100)
    y = np.linspace(-5, 5, 200)
    step_count = 1
    # former_exp_x = 0
    # former_exp_y = 0
    # former_opt_x = 0
    # former_opt_y = 0
    former_exp = np.array([0,0])
    former_opt = np.array([0,0])
    exp_sum = 0
    opt_sum = 0
    re_sum = 0
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
        exp_sum += exp_dis
        opt_sum += opt_dis
        re = -3*((opt_dis - exp_dis) / exp_dis)*100
        re_sum += re
        if re > 0 :
            ax.scatter(_ * step, re, alpha=0.45, s=[60], c="g")
            ax.plot(np.array([former_exp[0], _ * step]), np.array([former_exp[1], re]), c="g", ls="--")
        else :
            ax.scatter(_ * step, re, alpha=0.45, s=[60], c="r")
            ax.plot(np.array([former_exp[0], _ * step]), np.array([former_exp[1], re]), c="r", ls="--")
        # if _ == 0 :
        #     ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", label = "exp_err")
        #     ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", label = "opt_err", marker = "^")
        # else:
        #     ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", s = [60])
        #     ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", s = [60], marker = "^")
        #
        #     ax.plot(np.array([former_exp[0], _*step]), np.array([former_exp[1], exp_dis]), c="b", ls = "--")
        #     ax.plot(np.array([former_opt[0], _*step]), np.array([former_opt[1], opt_dis]), c="r")
        former_exp = np.array([_*step,re])
    avr_sum = re_sum/(time/step)
    if avr_sum > 0:
        ax.axhline(y=re_sum / (time / step), c='g', ls='--', lw=2, alpha=0.7, label="avr_performance" + str(round(avr_sum,4)))
    else:
        ax.axhline(y=re_sum / (time / step), c='r', ls='--', lw=2, alpha=0.7, label="avr_performance" + str(round(avr_sum,4)))
    #            alpha=0.7,
    #            label="avr_exp_err")
    # ax.axhline(y = opt_sum/(time/step), c='r', ls='-', lw=2,
    #            alpha=0.7,
    #            label="avr_opt_err")
    #plt.plot(x, y, 'o')
    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    #ax.set_xlabel("Measure Round", fontsize=12)
    ax.set_ylabel("Performance", fontsize=12)
    ax.set_title("round: " + str(time) + " point: " + str(point_number) + " step: " + str(step))
    ax.axhline(y=0, c='black', ls='-', lw=1, alpha=0.7, label="base")
    ax.legend()

def draw_compare_per_fix_time(ax, time : 'int', point_number : 'int', terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint',
                       target : 'Terminal.CartesianPoint', step : 'int' = "1"):
    x = np.linspace(0, time+5, 100)
    y = np.linspace(-5, 5, 200)
    step_count = 1
    # former_exp_x = 0
    # former_exp_y = 0
    # former_opt_x = 0
    # former_opt_y = 0
    former_exp = np.array([0,0])
    former_opt = np.array([0,0])
    exp_sum = 0
    opt_sum = 0
    re_sum = 0
    for _ in range(int(point_number/step)): #20/1
        exp_dis = 0
        opt_dis = 0
        #round_count = 0
        for __ in range(time):
            #round_count = __ + 1
            setA = GenNorm.gen_normal_point(_+1, terminalA, target)
            setB = GenNorm.gen_normal_point(_+1, terminalB, target)
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
        exp_dis /= time
        opt_dis /= time
        exp_sum += exp_dis
        opt_sum += opt_dis
        re = -3*((opt_dis - exp_dis) / exp_dis)*100
        re_sum += re
        if re > 0 :
            ax.scatter(_ * step, re, alpha=0.45, s=[60], c="g")
            ax.plot(np.array([former_exp[0], _ * step]), np.array([former_exp[1], re]), c="g", ls="--")
        else :
            ax.scatter(_ * step, re, alpha=0.45, s=[60], c="r")
            ax.plot(np.array([former_exp[0], _ * step]), np.array([former_exp[1], re]), c="r", ls="--")
        # if _ == 0 :
        #     ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", label = "exp_err")
        #     ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", label = "opt_err", marker = "^")
        # else:
        #     ax.scatter(_*step, exp_dis, alpha=0.45, c = "b", s = [60])
        #     ax.scatter(_*step, opt_dis, alpha=0.45, c = "r", s = [60],marker = "^")

            #ax.plot(np.array([former_exp[0], _*step]), np.array([former_exp[1], re]), c="b", ls = "--")
            #ax.plot(np.array([former_opt[0], _*step]), np.array([former_opt[1], opt_dis]), c="r")
        former_exp = np.array([_*step,re])
    avr_sum = re_sum / (point_number / step)
    if avr_sum > 0:
        ax.axhline(y=avr_sum, c='g', ls='--', lw=2, alpha=0.7, label="avr_performance" + str(round(avr_sum,4)))
    else:
        ax.axhline(y=avr_sum, c='r', ls='--', lw=2, alpha=0.7, label="avr_performance" + str(round(avr_sum,4)))

        #former_opt = np.array([_*step, opt_dis])
    #ax.axhline(y = exp_sum/(time/step), c='b', ls='--', lw=2,alpha=0.7,label="avr_exp_err")
    #ax.axhline(y = opt_sum/(time/step), c='r', ls='-', lw=2,alpha=0.7,label="avr_opt_err")
    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    #ax.set_xlabel("Measure Number", fontsize=12)
    ax.set_ylabel("Error Distance", fontsize=12)
    ax.set_title("round: " + str(time) + " point: " + str(point_number) + " step: " + str(step))
    ax.axhline(y=0, c='black', ls='-', lw=1, alpha=0.7, label="base" )
    ax.legend()
