import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import Calculation
import Terminal


def gen_normal_point(point_number : 'int', terminal : 'Terminal.CartesianPoint', target_terminal : 'Terminal.CartesianPoint'):
    Calculation.get_distance_from_weak_terminal_by_coord(terminal._x, terminal._y, target_terminal)
    mu = 0
    sigma = Calculation.get_sigma(Calculation.get_distance(terminal, target_terminal), terminal._terminal_type)
    normal_set = np.zeros((point_number,2))
    for x in range(normal_set.shape[0]):
        normal_set[x, 0] = target_terminal._x + np.random.normal(mu, math.sqrt(sigma))
        normal_set[x, 1] = target_terminal._y + np.random.normal(mu, math.sqrt(sigma))
    print("[DATA] %s PointSetGenerated As:" %point_number)
    print(normal_set)
    return normal_set

def gen_dis_set(point_number : 'int', terminal : 'Terminal.CartesianPoint', target_terminal : 'Terminal.CartesianPoint'):
    Calculation.get_distance_from_weak_terminal_by_coord(terminal._x, terminal._y, target_terminal)
    mu = 0
    sigma = Calculation.get_sigma(Calculation.get_distance(terminal, target_terminal), terminal._terminal_type)
    normal_set = np.zeros((point_number,2))
    dis_set = np.zeros((point_number,1))
    for x in range(normal_set.shape[0]):
        normal_set[x, 0] = np.random.normal(mu, math.sqrt(sigma))
        normal_set[x, 1] = np.random.normal(mu, math.sqrt(sigma))
        dis_set[x] = math.sqrt((normal_set[x, 0])**2 + (normal_set[x, 1])**2)
    print("[DATA] %sPointSetGenerated" %point_number)
    return np.reshape(dis_set, point_number)

