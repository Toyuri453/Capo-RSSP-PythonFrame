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
from matplotlib import rcParams
import pandas as pd

rcParams.update({'figure.autolayout': True})

type_factor_sigma = {'UWB' : 0.1, 'BlueTooth' : 5}

font1 = {
'weight' : 'normal',
'size'   : 12,
}

t_a = Terminal.CartesianPoint(-50, 0, "UWB", "TerminalA")
t_b = Terminal.CartesianPoint(10, 40, "UWB", "TerminalB")
t_c = Terminal.CartesianPoint(20, -30, "UWB", "TerminalC")
t_d = Terminal.CartesianPoint(50, 20, "UWB", "TerminalD")
t_e = Terminal.CartesianPoint(40, 30, "UWB", "TerminalE")
t_f = Terminal.CartesianPoint(-40, 30, "UWB", "TerminalF")
t_o = Terminal.CartesianPoint(5, 5, "BlueTooth", "Target")

e_t_a = Terminal.CartesianPoint(-50, 0, "UWB", "Error_TerminalA")
e_t_b = Terminal.CartesianPoint(10, 40, "UWB", "Error_TerminalB")
e_t_c = Terminal.CartesianPoint(20, -30, "UWB", "Error_TerminalC")


def get_rho(terminal, target):
    dis = Calculation.get_distance(terminal, target)
    factor = 1 + type_factor_sigma[target._terminal_type]
    rho = factor * math.log10(dis)
    #print (rho,dis,factor)
    return rho

def get_rho_from_dis(dis, terminal):
    factor = 1 + type_factor_sigma[terminal._terminal_type]
    rho = factor * math.log10(dis)
    return rho

def get_rho_by_sig(dis, sigma):
    factor = 1 + sigma
    rho = factor * math.log10(dis)
    return rho

avr_dis = Calculation.get_distance_triple(e_t_a, e_t_b, e_t_c)
error_rho = get_rho_from_dis(avr_dis, e_t_a)

def LSM(t_a: 'Terminal.CartesianPoint', t_b: 'Terminal.CartesianPoint', t_c: 'Terminal.CartesianPoint', t_o: 'Terminal.CartesianPoint'):
    d_a = Calculation.get_distance_sq(t_a, t_o)
    d_b = Calculation.get_distance_sq(t_b, t_o)
    d_c = Calculation.get_distance_sq(t_c, t_o)

    e_d_a = np.random.normal(math.sqrt(d_a), get_rho(t_a, t_o))
    e_d_b = np.random.normal(math.sqrt(d_b), get_rho(t_b, t_o))
    e_d_c = np.random.normal(math.sqrt(d_c), get_rho(t_c, t_o))

    print(e_d_a, e_d_b, e_d_c)

    m_a = np.array([[2*(t_a._x-t_c._x), 2*(t_a._y-t_c._y)], [2*(t_a._x-t_b._x), 2*(t_a._y-t_b._y)], [2*(t_b._x-t_c._x), 2*(t_b._y-t_c._y)]])
    m_aT = m_a.T
    m_aP1 = np.dot(m_aT,m_a)
    n_aI = np.linalg.inv(m_aP1)
    m_aP2 = np.dot(n_aI, m_aT)
    m_b = np.array([[(t_a._x ** 2 - t_c._x ** 2) + (t_a._y ** 2 - t_c._y ** 2) + e_d_c**2 - e_d_a**2],
                  [t_a._x ** 2 - t_b._x ** 2 + t_a._y ** 2 - t_b._y ** 2 + e_d_b**2 - e_d_a**2],
                  [t_b._x ** 2 - t_c._x ** 2 + t_b._y ** 2 - t_c._y ** 2 + e_d_c**2 - e_d_b**2]])
    m_x = np.dot(m_aP2,m_b)

    a_op_dis = math.sqrt((t_a._x - m_x[0]) ** 2 + (t_a._y - m_x[1]) ** 2)
    b_op_dis = math.sqrt((t_b._x - m_x[0]) ** 2 + (t_b._y - m_x[1]) ** 2)
    c_op_dis = math.sqrt((t_c._x - m_x[0]) ** 2 + (t_c._y - m_x[1]) ** 2)

    print(m_x)

    print(a_op_dis, e_d_a)
    print(b_op_dis, e_d_b)
    print(c_op_dis, e_d_c)
    d_a = Calculation.get_distance(t_a, t_o)
    d_b = Calculation.get_distance(t_b, t_o)
    d_c = Calculation.get_distance(t_c, t_o)

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), (abs(a_op_dis-d_a)+abs(b_op_dis-d_b)+abs(c_op_dis-d_c))/3, (abs(e_d_a-d_a)+abs(e_d_b-d_b)+abs(e_d_c-d_c))/3, m_x

def LSM_4(t_a: 'Terminal.CartesianPoint', t_b: 'Terminal.CartesianPoint', t_c: 'Terminal.CartesianPoint', t_d: 'Terminal.CartesianPoint', t_o: 'Terminal.CartesianPoint'):
    d_a = Calculation.get_distance_sq(t_a, t_o)
    d_b = Calculation.get_distance_sq(t_b, t_o)
    d_c = Calculation.get_distance_sq(t_c, t_o)
    d_d = Calculation.get_distance_sq(t_d, t_o)

    e_d_a = np.random.normal(math.sqrt(d_a), get_rho(t_a, t_o))
    e_d_b = np.random.normal(math.sqrt(d_b), get_rho(t_b, t_o))
    e_d_c = np.random.normal(math.sqrt(d_c), get_rho(t_c, t_o))
    e_d_d = np.random.normal(math.sqrt(d_d), get_rho(t_d, t_o))


    m_a = np.array([[2*(t_a._x-t_d._x), 2*(t_a._y-t_d._y)], [2*(t_a._x-t_b._x), 2*(t_a._y-t_b._y)], [2*(t_b._x-t_c._x), 2*(t_b._y-t_c._y)], [2*(t_c._x-t_d._x), 2*(t_c._y-t_d._y)]])
    m_aT = m_a.T
    m_aP1 = np.dot(m_aT,m_a)
    n_aI = np.linalg.inv(m_aP1)
    m_aP2 = np.dot(n_aI, m_aT)
    m_b = np.array([[(t_a._x ** 2 - t_d._x ** 2) + (t_a._y ** 2 - t_d._y ** 2) + e_d_d**2 - e_d_a**2],
                  [t_a._x ** 2 - t_b._x ** 2 + t_a._y ** 2 - t_b._y ** 2 + e_d_b**2 - e_d_a**2],
                  [t_b._x ** 2 - t_c._x ** 2 + t_b._y ** 2 - t_c._y ** 2 + e_d_c**2 - e_d_b**2],
                  [t_c._x ** 2 - t_d._x ** 2 + t_c._y ** 2 - t_d._y ** 2 + e_d_d**2 - e_d_c**2]])
    m_x = np.dot(m_aP2,m_b)

    a_op_dis = math.sqrt((t_a._x - m_x[0]) ** 2 + (t_a._y - m_x[1]) ** 2)
    b_op_dis = math.sqrt((t_b._x - m_x[0]) ** 2 + (t_b._y - m_x[1]) ** 2)
    c_op_dis = math.sqrt((t_c._x - m_x[0]) ** 2 + (t_c._y - m_x[1]) ** 2)
    d_op_dis = math.sqrt((t_d._x - m_x[0]) ** 2 + (t_d._y - m_x[1]) ** 2)

    print(m_x)

    print(a_op_dis, e_d_a)
    print(b_op_dis, e_d_b)
    print(c_op_dis, e_d_c)
    print(d_op_dis, e_d_d)
    d_a = Calculation.get_distance(t_a, t_o)
    d_b = Calculation.get_distance(t_b, t_o)
    d_c = Calculation.get_distance(t_c, t_o)
    d_d = Calculation.get_distance(t_d, t_o)

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), abs(d_op_dis-d_d), abs(e_d_d-d_d), (abs(a_op_dis-d_a)+abs(b_op_dis-d_b)+abs(c_op_dis-d_c)+abs(d_op_dis-d_d))/4

def LSM_5(t_a: 'Terminal.CartesianPoint', t_b: 'Terminal.CartesianPoint', t_c: 'Terminal.CartesianPoint', t_d: 'Terminal.CartesianPoint', t_e: 'Terminal.CartesianPoint', t_o: 'Terminal.CartesianPoint'):
    d_a = Calculation.get_distance_sq(t_a, t_o)
    d_b = Calculation.get_distance_sq(t_b, t_o)
    d_c = Calculation.get_distance_sq(t_c, t_o)
    d_d = Calculation.get_distance_sq(t_d, t_o)
    d_e = Calculation.get_distance_sq(t_e, t_o)

    e_d_a = np.random.normal(math.sqrt(d_a), get_rho(t_a, t_o))
    e_d_b = np.random.normal(math.sqrt(d_b), get_rho(t_b, t_o))
    e_d_c = np.random.normal(math.sqrt(d_c), get_rho(t_c, t_o))
    e_d_d = np.random.normal(math.sqrt(d_d), get_rho(t_d, t_o))
    e_d_e = np.random.normal(math.sqrt(d_e), get_rho(t_e, t_o))

    m_a = np.array([[2*(t_a._x-t_e._x), 2*(t_a._y-t_e._y)], [2*(t_a._x-t_b._x), 2*(t_a._y-t_b._y)], [2*(t_b._x-t_c._x), 2*(t_b._y-t_c._y)], [2*(t_c._x-t_d._x), 2*(t_c._y-t_d._y)]
                    ,[2*(t_d._x-t_e._x), 2*(t_d._y-t_e._y)]])
    m_aT = m_a.T
    m_aP1 = np.dot(m_aT,m_a)
    n_aI = np.linalg.inv(m_aP1)
    m_aP2 = np.dot(n_aI, m_aT)
    m_b = np.array([[(t_a._x ** 2 - t_e._x ** 2) + (t_a._y ** 2 - t_e._y ** 2) + e_d_e**2 - e_d_a**2],
                  [t_a._x ** 2 - t_b._x ** 2 + t_a._y ** 2 - t_b._y ** 2 + e_d_b**2 - e_d_a**2],
                  [t_b._x ** 2 - t_c._x ** 2 + t_b._y ** 2 - t_c._y ** 2 + e_d_c**2 - e_d_b**2],
                  [t_c._x ** 2 - t_d._x ** 2 + t_c._y ** 2 - t_d._y ** 2 + e_d_d**2 - e_d_c**2],
                  [t_d._x ** 2 - t_e._x ** 2 + t_d._y ** 2 - t_e._y ** 2 + e_d_e ** 2 - e_d_d ** 2]])
    m_x = np.dot(m_aP2,m_b)

    a_op_dis = math.sqrt((t_a._x - m_x[0]) ** 2 + (t_a._y - m_x[1]) ** 2)
    b_op_dis = math.sqrt((t_b._x - m_x[0]) ** 2 + (t_b._y - m_x[1]) ** 2)
    c_op_dis = math.sqrt((t_c._x - m_x[0]) ** 2 + (t_c._y - m_x[1]) ** 2)
    d_op_dis = math.sqrt((t_d._x - m_x[0]) ** 2 + (t_d._y - m_x[1]) ** 2)
    e_op_dis = math.sqrt((t_e._x - m_x[0]) ** 2 + (t_e._y - m_x[1]) ** 2)

    print(m_x)

    print(a_op_dis, e_d_a)
    print(b_op_dis, e_d_b)
    print(c_op_dis, e_d_c)
    print(d_op_dis, e_d_d)
    print(e_op_dis, e_d_e)

    d_a = Calculation.get_distance(t_a, t_o)
    d_b = Calculation.get_distance(t_b, t_o)
    d_c = Calculation.get_distance(t_c, t_o)
    d_d = Calculation.get_distance(t_d, t_o)
    d_e = Calculation.get_distance(t_e, t_o)

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), abs(d_op_dis-d_d), abs(e_d_d-d_d), abs(e_op_dis-d_e), abs(e_d_e-d_e), (abs(a_op_dis-d_a)+abs(b_op_dis-d_b)+abs(c_op_dis-d_c)+abs(d_op_dis-d_d)+abs(e_op_dis-d_e))/5

def LSM_6(t_a: 'Terminal.CartesianPoint', t_b: 'Terminal.CartesianPoint', t_c: 'Terminal.CartesianPoint', t_d: 'Terminal.CartesianPoint', t_e: 'Terminal.CartesianPoint', t_f: 'Terminal.CartesianPoint', t_o: 'Terminal.CartesianPoint'):
    d_a = Calculation.get_distance_sq(t_a, t_o)
    d_b = Calculation.get_distance_sq(t_b, t_o)
    d_c = Calculation.get_distance_sq(t_c, t_o)
    d_d = Calculation.get_distance_sq(t_d, t_o)
    d_e = Calculation.get_distance_sq(t_e, t_o)
    d_f = Calculation.get_distance_sq(t_f, t_o)

    e_d_a = np.random.normal(math.sqrt(d_a), get_rho(t_a, t_o))
    e_d_b = np.random.normal(math.sqrt(d_b), get_rho(t_b, t_o))
    e_d_c = np.random.normal(math.sqrt(d_c), get_rho(t_c, t_o))
    e_d_d = np.random.normal(math.sqrt(d_d), get_rho(t_d, t_o))
    e_d_e = np.random.normal(math.sqrt(d_e), get_rho(t_e, t_o))
    e_d_f = np.random.normal(math.sqrt(d_f), get_rho(t_f, t_o))

    m_a = np.array([[2*(t_a._x-t_f._x), 2*(t_a._y-t_f._y)], [2*(t_a._x-t_b._x), 2*(t_a._y-t_b._y)], [2*(t_b._x-t_c._x), 2*(t_b._y-t_c._y)], [2*(t_c._x-t_d._x), 2*(t_c._y-t_d._y)]
                    ,[2*(t_d._x-t_e._x), 2*(t_d._y-t_e._y)] ,[2*(t_e._x-t_f._x), 2*(t_e._y-t_f._y)]])
    m_aT = m_a.T
    m_aP1 = np.dot(m_aT,m_a)
    n_aI = np.linalg.inv(m_aP1)
    m_aP2 = np.dot(n_aI, m_aT)
    m_b = np.array([[(t_a._x ** 2 - t_f._x ** 2) + (t_a._y ** 2 - t_f._y ** 2) + e_d_f**2 - e_d_a**2],
                  [t_a._x ** 2 - t_b._x ** 2 + t_a._y ** 2 - t_b._y ** 2 + e_d_b**2 - e_d_a**2],
                  [t_b._x ** 2 - t_c._x ** 2 + t_b._y ** 2 - t_c._y ** 2 + e_d_c**2 - e_d_b**2],
                  [t_c._x ** 2 - t_d._x ** 2 + t_c._y ** 2 - t_d._y ** 2 + e_d_d**2 - e_d_c**2],
                  [t_d._x ** 2 - t_e._x ** 2 + t_d._y ** 2 - t_e._y ** 2 + e_d_e ** 2 - e_d_d ** 2],
                  [t_e._x ** 2 - t_f._x ** 2 + t_e._y ** 2 - t_f._y ** 2 + e_d_f ** 2 - e_d_e ** 2]])
    m_x = np.dot(m_aP2,m_b)

    a_op_dis = math.sqrt((t_a._x - m_x[0]) ** 2 + (t_a._y - m_x[1]) ** 2)
    b_op_dis = math.sqrt((t_b._x - m_x[0]) ** 2 + (t_b._y - m_x[1]) ** 2)
    c_op_dis = math.sqrt((t_c._x - m_x[0]) ** 2 + (t_c._y - m_x[1]) ** 2)
    d_op_dis = math.sqrt((t_d._x - m_x[0]) ** 2 + (t_d._y - m_x[1]) ** 2)
    e_op_dis = math.sqrt((t_e._x - m_x[0]) ** 2 + (t_e._y - m_x[1]) ** 2)
    f_op_dis = math.sqrt((t_f._x - m_x[0]) ** 2 + (t_f._y - m_x[1]) ** 2)

    print(m_x)

    print(a_op_dis, e_d_a)
    print(b_op_dis, e_d_b)
    print(c_op_dis, e_d_c)
    print(d_op_dis, e_d_d)
    print(e_op_dis, e_d_e)
    print(f_op_dis, e_d_f)

    d_a = Calculation.get_distance(t_a, t_o)
    d_b = Calculation.get_distance(t_b, t_o)
    d_c = Calculation.get_distance(t_c, t_o)
    d_d = Calculation.get_distance(t_d, t_o)
    d_e = Calculation.get_distance(t_e, t_o)
    d_f = Calculation.get_distance(t_f, t_o)

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), abs(d_op_dis-d_d), abs(e_d_d-d_d), abs(e_op_dis-d_e), abs(e_d_e-d_e), abs(f_op_dis-d_f), abs(e_d_f-d_f), (abs(a_op_dis-d_a)+abs(b_op_dis-d_b)+abs(c_op_dis-d_c)+abs(d_op_dis-d_d)+abs(e_op_dis-d_e)+abs(f_op_dis-d_f))/6

#LSM(t_a, t_b, t_c, t_o)

#ax.scatter(terminal._x, terminal._y, s=out_area, color = terminal._terminal_color, alpha=alpha, edgecolors=color, linewidths=[1])



a_op_dis_set = np.zeros((3000,1))
e_d_a_set = np.zeros((3000,1))
b_op_dis_set = np.zeros((3000,1))
e_d_b_set = np.zeros((3000,1))
c_op_dis_set = np.zeros((3000,1))
e_d_c_set = np.zeros((3000,1))

op_set = np.zeros((3000,1))
avr_d = np.zeros((3000,1))

lsm4_set = np.zeros((3000,1))
lsm5_set = np.zeros((3000,1))
lsm6_set = np.zeros((3000,1))

a_sum = 0
b_sum = 0
c_sum = 0
e_a_sum = 0
e_b_sum = 0
e_c_sum = 0

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []

lsm4_data = []
lsm5_data = []
lsm6_data = []

error_lsm3_data = []
error2_lsm3_data = []
error3_lsm3_data = []

sum_4, sum_5, sum_6 = 0, 0, 0

for _ in range(3000):
    e_t_a = Terminal.CartesianPoint(-50, 0, "UWB", "Error_TerminalA")
    e_t_b = Terminal.CartesianPoint(10, 40, "UWB", "Error_TerminalB")
    e_t_c = Terminal.CartesianPoint(20, -30, "UWB", "Error_TerminalC")
    error_rho = get_rho_from_dis(avr_dis, e_t_a)
    e_t_a._x += np.random.normal(avr_dis, error_rho) - avr_dis
    e_t_a._y += np.random.normal(avr_dis, error_rho) - avr_dis
    e_t_b._x += np.random.normal(avr_dis, error_rho) - avr_dis
    e_t_b._y += np.random.normal(avr_dis, error_rho) - avr_dis
    e_t_c._x += np.random.normal(avr_dis, error_rho) - avr_dis
    e_t_c._y += np.random.normal(avr_dis, error_rho) - avr_dis
    print(e_t_a._x, e_t_a._y, e_t_b._x, e_t_b._y, e_t_c._x, e_t_c._y)

    error_result = LSM(e_t_a, e_t_b, e_t_c, t_o)
    error_a = abs(math.sqrt(((error_result[8][0] - e_t_a._x)**2 + (error_result[8][1] - e_t_a._y)**2)) - Calculation.get_distance(t_a, t_o))
    error_b = abs(math.sqrt(((error_result[8][0] - e_t_b._x)**2 + (error_result[8][1] - e_t_b._y)**2)) - Calculation.get_distance(t_b, t_o))
    error_c = abs(math.sqrt(((error_result[8][0] - e_t_c._x)**2 + (error_result[8][1] - e_t_c._y)**2)) - Calculation.get_distance(t_c, t_o))
    error_lsm3_data.append((error_a + error_b + error_c)/3)


    e_t_a = Terminal.CartesianPoint(-50, 0, "UWB", "Error_TerminalA")
    e_t_b = Terminal.CartesianPoint(10, 40, "UWB", "Error_TerminalB")
    e_t_c = Terminal.CartesianPoint(20, -30, "UWB", "Error_TerminalC")
    error2_rho = get_rho_by_sig(avr_dis, 0.5)
    e_t_a._x += np.random.normal(avr_dis, error2_rho) - avr_dis
    e_t_a._y += np.random.normal(avr_dis, error2_rho) - avr_dis
    e_t_b._x += np.random.normal(avr_dis, error2_rho) - avr_dis
    e_t_b._y += np.random.normal(avr_dis, error2_rho) - avr_dis
    e_t_c._x += np.random.normal(avr_dis, error2_rho) - avr_dis
    e_t_c._y += np.random.normal(avr_dis, error2_rho) - avr_dis
    print(e_t_a._x, e_t_a._y, e_t_b._x, e_t_b._y, e_t_c._x, e_t_c._y)

    error2_result = LSM(e_t_a, e_t_b, e_t_c, t_o)
    error2_a = abs(math.sqrt(((error2_result[8][0] - e_t_a._x)**2 + (error2_result[8][1] - e_t_a._y)**2)) - Calculation.get_distance(t_a, t_o))
    error2_b = abs(math.sqrt(((error2_result[8][0] - e_t_b._x)**2 + (error2_result[8][1] - e_t_b._y)**2)) - Calculation.get_distance(t_b, t_o))
    error2_c = abs(math.sqrt(((error2_result[8][0] - e_t_c._x)**2 + (error2_result[8][1] - e_t_c._y)**2)) - Calculation.get_distance(t_c, t_o))
    error2_lsm3_data.append((error2_a + error2_b + error2_c)/3)


    e_t_a = Terminal.CartesianPoint(-50, 0, "UWB", "Error_TerminalA")
    e_t_b = Terminal.CartesianPoint(10, 40, "UWB", "Error_TerminalB")
    e_t_c = Terminal.CartesianPoint(20, -30, "UWB", "Error_TerminalC")
    error3_rho = get_rho_by_sig(avr_dis, 2)
    e_t_a._x += np.random.normal(avr_dis, error3_rho) - avr_dis
    e_t_a._y += np.random.normal(avr_dis, error3_rho) - avr_dis
    e_t_b._x += np.random.normal(avr_dis, error3_rho) - avr_dis
    e_t_b._y += np.random.normal(avr_dis, error3_rho) - avr_dis
    e_t_c._x += np.random.normal(avr_dis, error3_rho) - avr_dis
    e_t_c._y += np.random.normal(avr_dis, error3_rho) - avr_dis
    print(e_t_a._x, e_t_a._y, e_t_b._x, e_t_b._y, e_t_c._x, e_t_c._y)

    error3_result = LSM(e_t_a, e_t_b, e_t_c, t_o)
    error3_a = abs(math.sqrt(((error3_result[8][0] - e_t_a._x)**2 + (error3_result[8][1] - e_t_a._y)**2)) - Calculation.get_distance(t_a, t_o))
    error3_b = abs(math.sqrt(((error3_result[8][0] - e_t_b._x)**2 + (error3_result[8][1] - e_t_b._y)**2)) - Calculation.get_distance(t_b, t_o))
    error3_c = abs(math.sqrt(((error3_result[8][0] - e_t_c._x)**2 + (error3_result[8][1] - e_t_c._y)**2)) - Calculation.get_distance(t_c, t_o))
    error3_lsm3_data.append((error3_a + error3_b + error3_c)/3)

    result = LSM(t_a, t_b, t_c, t_o)
    result4 = LSM_4(t_a, t_b, t_c, t_d, t_o)
    result5 = LSM_5(t_a, t_b, t_c, t_d, t_e, t_o)
    result6 = LSM_6(t_a, t_b, t_c, t_d, t_e, t_f, t_o)
    a_op_dis_set[_] = result[0]
    e_d_a_set[_] = result[1]
    b_op_dis_set[_] = result[2]
    e_d_b_set[_] = result[3]
    c_op_dis_set[_] = result[4]
    e_d_c_set[_] = result[5]
    op_set[_] = result[6]
    avr_d[_] = result[7]
    a_sum += a_op_dis_set[_]
    b_sum += b_op_dis_set[_]
    c_sum += c_op_dis_set[_]
    e_a_sum += e_d_a_set[_]
    e_b_sum += e_d_b_set[_]
    e_c_sum += e_d_c_set[_]
    data1.append(result[1])
    data2.append(result[3])
    data3.append(result[5])
    data4.append(result[0])
    data5.append(result[2])
    data6.append(result[4])
    data7.append(result[6])
    data8.append(result[7])
    lsm4_set[_]=result4[8]
    lsm5_set[_]=result5[10]
    lsm6_set[_]=result6[12]
    lsm4_data.append(result4[8])
    lsm5_data.append(result5[10])
    lsm6_data.append(result6[12])
    print (result)

fig, (pax1) = plt.subplots(1,1, figsize=(5,4))
fig2, (pax2) = plt.subplots(1,1, figsize=(5,4))
fig3, (pax3) = plt.subplots(1,1, figsize=(5,4))
fig4, (ax5) = plt.subplots(1,1, figsize=(5,4))

fig5, (ax4) = plt.subplots(1,1, figsize=(5,4))
fig11, ((ax1, ax2, ax3)) = plt.subplots(1,3,figsize=(14,4))
fig6, ((ax6, ax7, ax8)) = plt.subplots(1,3,figsize=(14,4))

fig7, (pax6) = plt.subplots(1,1, figsize=(5,4))
fig8, (pax7) = plt.subplots(1,1, figsize=(5,4))
fig9, (pax8) = plt.subplots(1,1, figsize=(5,4))
fig10, (ax9) = plt.subplots(1,1, figsize=(5,4))
fig11, (pax9) = plt.subplots(1,1, figsize=(5,4))
fig12, (pax10) = plt.subplots(1,1, figsize=(5,4))
ax1.set_xlabel("Error distance", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
#ax1.set_title('Error distribution of Terminal A', fontsize=15, fontweight='light')
n, bins, patches = ax1.hist(e_d_a_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Original')
n, bins, patches = ax1.hist(a_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
ax1.legend(fontsize = 12)
#ax1.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax1.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

ax5.set_xlabel("Error distance", fontsize=14)
ax5.set_ylabel("Frequency", fontsize=14)
#ax1.set_title('Error distribution of Terminal A', fontsize=15, fontweight='light')
n, bins, patches = ax5.hist(avr_d, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Original')
n, bins, patches = ax5.hist(op_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
ax5.legend(fontsize = 12)
#ax5.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax5.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

ax2.set_xlabel("Error distance", fontsize=14)
ax2.set_ylabel("Frequency", fontsize=14)
#ax2.set_title('Error distribution of Terminal B', fontsize=15, fontweight='light')
n, bins, patches = ax2.hist(e_d_b_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Original')
n, bins, patches = ax2.hist(b_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
ax2.legend(fontsize = 12)
#ax2.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax2.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

ax3.set_xlabel("Error distance", fontsize=14)
ax3.set_ylabel("Frequency", fontsize=14)
#ax3.set_title('Error distribution of Terminal C', fontsize=15, fontweight='light')
n, bins, patches = ax3.hist(e_d_c_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='original')
n, bins, patches = ax3.hist(c_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
ax3.legend(fontsize = 12)
#ax3.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax3.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

#########################################################################
#########################################################################

pax1.set_xlabel("Error distance", fontsize=14)
pax1.set_ylabel("Frequency", fontsize=14)
#ax1.set_title('Error distribution of Terminal A', fontsize=15, fontweight='light')
n, bins, patches = pax1.hist(e_d_a_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Original')
n, bins, patches = pax1.hist(a_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
pax1.legend(fontsize = 12)
#ax1.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax1.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))


#ax5.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax5.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

pax2.set_xlabel("Error distance", fontsize=14)
pax2.set_ylabel("Frequency", fontsize=14)
#ax2.set_title('Error distribution of Terminal B', fontsize=15, fontweight='light')
n, bins, patches = pax2.hist(e_d_b_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Original')
n, bins, patches = pax2.hist(b_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
pax2.legend(fontsize = 12)
#ax2.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax2.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

pax3.set_xlabel("Error distance", fontsize=14)
pax3.set_ylabel("Frequency", fontsize=14)
#ax3.set_title('Error distribution of Terminal C', fontsize=15, fontweight='light')
n, bins, patches = pax3.hist(e_d_c_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='original')
n, bins, patches = pax3.hist(c_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='Capo')

n = n.astype('int')
pax3.legend(fontsize = 12)
#ax3.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
#ax3.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

#########################################################################
#########################################################################

a_sum /= 3000
b_sum /= 3000
c_sum /= 3000
e_a_sum /= 3000
e_b_sum /= 3000
e_c_sum /= 3000

print(a_sum, e_a_sum, b_sum, e_b_sum, c_sum, e_c_sum)
#print(a_op_dis_set[2999], e_d_a_set[2999], b_op_dis_set[2999], e_d_b_set[2999], c_op_dis_set[2999], e_d_c_set[2999])
print((a_sum+b_sum+c_sum)/3,(e_a_sum+e_b_sum+e_c_sum)/3)

#ax.scatter(terminal._x, terminal._y, s=out_area, color = terminal._terminal_color, alpha=alpha, edgecolors=color, linewidths=[1])
# ax4.set_xlabel("Round", fontsize=14)
# ax4.set_ylabel("Average Error/ m", fontsize=14)
# a3_error = 0
# a4_error = 0
# a5_error = 0
# a6_error = 0
# round_list = []
# a3_error_list = []
# a4_error_list = []
# a5_error_list = []
# a6_error_list = []
# for _ in range(3000):
#     r3 = LSM(t_a, t_b, t_c, t_o)
#     r4 = LSM_4(t_a, t_b, t_c, t_d, t_o)
#     r5 = LSM_5(t_a, t_b, t_c, t_d, t_e, t_o)
#     r6 = LSM_6(t_a, t_b, t_c, t_d, t_e, t_f, t_o)
#     a3_error += (r3[0]+r3[2]+r3[4])/3
#     a4_error += (r4[0]+r4[2]+r4[4]+r4[6])/4
#     a5_error += (r5[0]+r5[2]+r5[4]+r5[6]+r5[8])/5
#     a6_error += (r6[0]+r6[2]+r6[4]+r6[6]+r6[8]+r6[10])/6
#     if _ == 9 :
#         ax4.scatter(_, a3_error / 10, color = '#bf1722', linewidths=[1], label = "3 devices")
#         ax4.scatter(_, a4_error / 10, color='#17aebf', linewidths=[1], marker = "^", label = "4 devices")
#         ax4.scatter(_, a5_error / 10, color='#57ba06', linewidths=[1], marker = "s", label = "5 devices")
#         ax4.scatter(_, a6_error / 10, color='#8d17bf', linewidths=[1], marker = "x", label = "6 devices")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 10)
#         a4_error_list.append(a4_error / 10)
#         a5_error_list.append(a5_error / 10)
#         a6_error_list.append(a6_error / 10)
#     if _ == 29 :
#         ax4.scatter(_, a3_error / 30, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 30, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 30, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 30, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 30)
#         a4_error_list.append(a4_error / 30)
#         a5_error_list.append(a5_error / 30)
#         a6_error_list.append(a6_error / 30)
#     if _ == 99 :
#         ax4.scatter(_, a3_error / 100, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 100, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 100, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 100, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 100)
#         a4_error_list.append(a4_error / 100)
#         a5_error_list.append(a5_error / 100)
#         a6_error_list.append(a6_error / 100)
#     if _ == 199 :
#         ax4.scatter(_, a3_error / 200, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 200, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 200, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 200, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 200)
#         a4_error_list.append(a4_error / 200)
#         a5_error_list.append(a5_error / 200)
#         a6_error_list.append(a6_error / 200)
#     if _ == 499 :
#         ax4.scatter(_, a3_error / 500, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 500, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 500, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 500, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 500)
#         a4_error_list.append(a4_error / 500)
#         a5_error_list.append(a5_error / 500)
#         a6_error_list.append(a6_error / 500)
#     if _ == 999 :
#         ax4.scatter(_, a3_error / 1000, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 1000, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 1000, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 1000, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 1000)
#         a4_error_list.append(a4_error / 1000)
#         a5_error_list.append(a5_error / 1000)
#         a6_error_list.append(a6_error / 1000)
#     if _ == 1799 :
#         ax4.scatter(_, a3_error / 1800, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 1800, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 1800, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 1800, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 1800)
#         a4_error_list.append(a4_error / 1800)
#         a5_error_list.append(a5_error / 1800)
#         a6_error_list.append(a6_error / 1800)
#     if _ == 2999 :
#         ax4.scatter(_, a3_error / 3000, color = '#bf1722', linewidths=[1])
#         ax4.scatter(_, a4_error / 3000, color='#17aebf', linewidths=[1], marker = "^")
#         ax4.scatter(_, a5_error / 3000, color='#57ba06', linewidths=[1], marker = "s")
#         ax4.scatter(_, a6_error / 3000, color='#8d17bf', linewidths=[1], marker = "x")
#         round_list.append(_)
#         a3_error_list.append(a3_error / 3000)
#         a4_error_list.append(a4_error / 3000)
#         a5_error_list.append(a5_error / 3000)
#         a6_error_list.append(a6_error / 3000)
# ax4.xaxis.grid(True, which='major', linestyle=(0, (8, 8)))
# ax4.yaxis.grid(True, which='major', linestyle=(0, (8, 8)))
# ax4.plot(round_list, a3_error_list, c = '#bf1722', linewidth = 0.8)
# ax4.plot(round_list, a4_error_list, c = '#17aebf', linewidth = 0.8)
# ax4.plot(round_list, a5_error_list, c = '#57ba06', linewidth = 0.8)
# ax4.plot(round_list, a6_error_list, c = '#8d17bf', linewidth = 0.8)
# ax4.legend(fontsize = 10,framealpha=0)

#BLE
print(data1)
denominator=len(data1)#分母数量 e_d_a-d_a r[1] e_d_a_set
Data1=pd.Series(data1)#将数据转换为Series利用分组频数计算
Fre1=Data1.value_counts()
Fre1_sort=Fre1.sort_index(axis=0,ascending=True)
Fre1_df=Fre1_sort.reset_index()#将Series数据转换为DataFrame
Fre1_df[0]=Fre1_df[0]/denominator#转换成概率
Fre1_df.columns=['Rds','Fre1']
Fre1_df['cumsum']=np.cumsum(Fre1_df['Fre1'])

denominator=len(data2)
Data2=pd.Series(data2)
Fre2=Data2.value_counts()
Fre2_sort=Fre2.sort_index(axis=0,ascending=True)
Fre2_df=Fre2_sort.reset_index()#将Series数据转换为DataFrame
Fre2_df[0]=Fre2_df[0]/denominator#转换成概率
Fre2_df.columns=['Rds','Fre2']
Fre2_df['cumsum']=np.cumsum(Fre2_df['Fre2'])

denominator=len(data3)
Data3=pd.Series(data3)
Fre3=Data3.value_counts()
Fre3_sort=Fre3.sort_index(axis=0,ascending=True)
Fre3_df=Fre3_sort.reset_index()#将Series数据转换为DataFrame
Fre3_df[0]=Fre3_df[0]/denominator#转换成概率
Fre3_df.columns=['Rds','Fre3']
Fre3_df['cumsum']=np.cumsum(Fre3_df['Fre3'])

#Capo

denominator=len(data4)#分母数量 r[0] a_op_dis_set
Data4=pd.Series(data4)#将数据转换为Series利用分组频数计算
Fre4=Data4.value_counts()
Fre4_sort=Fre4.sort_index(axis=0,ascending=True)
Fre4_df=Fre4_sort.reset_index()#将Series数据转换为DataFrame
Fre4_df[0]=Fre4_df[0]/denominator#转换成概率
Fre4_df.columns=['Rds','Fre4']
Fre4_df['cumsum']=np.cumsum(Fre4_df['Fre4'])

denominator=len(data5)#分母数量 r[2] b_op_dis_set
Data5=pd.Series(data5)#将数据转换为Series利用分组频数计算
Fre5=Data5.value_counts()
Fre5_sort=Fre5.sort_index(axis=0,ascending=True)
Fre5_df=Fre5_sort.reset_index()#将Series数据转换为DataFrame
Fre5_df[0]=Fre5_df[0]/denominator#转换成概率
Fre5_df.columns=['Rds','Fre5']
Fre5_df['cumsum']=np.cumsum(Fre5_df['Fre5'])

denominator=len(data6)#分母数量 r[4] c_op_dis_set
Data6=pd.Series(data6)#将数据转换为Series利用分组频数计算
Fre6=Data6.value_counts()
Fre6_sort=Fre6.sort_index(axis=0,ascending=True)
Fre6_df=Fre6_sort.reset_index()#将Series数据转换为DataFrame
Fre6_df[0]=Fre6_df[0]/denominator#转换成概率
Fre6_df.columns=['Rds','Fre6']
Fre6_df['cumsum']=np.cumsum(Fre6_df['Fre6'])

#Average

denominator=len(data7)#分母数量 r[4] c_op_dis_set
Data7=pd.Series(data7)#将数据转换为Series利用分组频数计算
Fre7=Data7.value_counts()
Fre7_sort=Fre7.sort_index(axis=0,ascending=True)
Fre7_df=Fre7_sort.reset_index()#将Series数据转换为DataFrame
Fre7_df[0]=Fre7_df[0]/denominator#转换成概率
Fre7_df.columns=['Rds','Fre7']
Fre7_df['cumsum']=np.cumsum(Fre7_df['Fre7'])

denominator=len(data8)#分母数量 r[4] c_op_dis_set
Data8=pd.Series(data8)#将数据转换为Series利用分组频数计算
Fre8=Data8.value_counts()
Fre8_sort=Fre8.sort_index(axis=0,ascending=True)
Fre8_df=Fre8_sort.reset_index()#将Series数据转换为DataFrame
Fre8_df[0]=Fre8_df[0]/denominator#转换成概率
Fre8_df.columns=['Rds','Fre8']
Fre8_df['cumsum']=np.cumsum(Fre8_df['Fre8'])

ax6.plot(Fre1_df['Rds'],Fre1_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
ax6.plot(Fre4_df['Rds'],Fre4_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
ax6.set_xlabel("OA: Ranging error (m)", font1)
ax6.set_ylabel("Cumulative Probability", font1)
ax6.set_xlim(0,25)
ax6.legend(fontsize = 12, loc = 'upper left',framealpha=0)

ax7.plot(Fre2_df['Rds'],Fre2_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
ax7.plot(Fre5_df['Rds'],Fre5_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
ax7.set_xlabel("OB: Ranging error (m)", font1)
ax7.set_ylabel("Cumulative Probability", font1)
ax7.set_xlim(0,25)
ax7.legend(fontsize = 12, loc = 'upper left',framealpha=0)


ax8.plot(Fre3_df['Rds'],Fre3_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
ax8.plot(Fre6_df['Rds'],Fre6_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
ax8.set_xlabel("OC: Ranging error (m)", font1)
ax8.set_ylabel("Cumulative Probability", font1)
ax8.set_xlim(0,25)
ax8.legend(fontsize = 12, loc = 'upper left',framealpha=0)

ax9.plot(Fre8_df['Rds'],Fre8_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
ax9.plot(Fre7_df['Rds'],Fre7_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
ax9.set_xlabel("OC: Ranging error (m)", font1)
ax9.set_ylabel("Cumulative Probability", font1)
ax9.set_xlim(0,25)
ax9.legend(fontsize = 12, loc = 'upper left',framealpha=0)

##########################################################################################
pax6.plot(Fre1_df['Rds'],Fre1_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
pax6.plot(Fre4_df['Rds'],Fre4_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
pax6.set_xlabel("OA: Ranging error (m)", font1)
pax6.set_ylabel("Cumulative Probability", font1)
pax6.set_xlim(0,25)
pax6.legend(fontsize = 12, loc = 'upper left',framealpha=0)

pax7.plot(Fre2_df['Rds'],Fre2_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
pax7.plot(Fre5_df['Rds'],Fre5_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
pax7.set_xlabel("OB: Ranging error (m)", font1)
pax7.set_ylabel("Cumulative Probability", font1)
pax7.set_xlim(0,25)
pax7.legend(fontsize = 12, loc = 'upper left',framealpha=0)


pax8.plot(Fre3_df['Rds'],Fre3_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
pax8.plot(Fre6_df['Rds'],Fre6_df['cumsum'],color='#00cc33',label='Capo')
#ax1.set_title("CDF")
pax8.set_xlabel("OC: Ranging error (m)", font1)
pax8.set_ylabel("Cumulative Probability", font1)
pax8.set_xlim(0,25)
pax8.legend(fontsize = 12, loc = 'upper left',framealpha=0)



denominator=len(lsm4_data)#分母数量 r[4] c_op_dis_set
Data10=pd.Series(lsm4_data)#将数据转换为Series利用分组频数计算
Fre10=Data10.value_counts()
Fre10_sort=Fre10.sort_index(axis=0,ascending=True)
Fre10_df=Fre10_sort.reset_index()#将Series数据转换为DataFrame
Fre10_df[0]=Fre10_df[0]/denominator#转换成概率
Fre10_df.columns=['Rds','Fre10']
Fre10_df['cumsum']=np.cumsum(Fre10_df['Fre10'])

denominator=len(lsm5_data)#分母数量 r[4] c_op_dis_set
Data11=pd.Series(lsm5_data)#将数据转换为Series利用分组频数计算
Fre11=Data11.value_counts()
Fre11_sort=Fre11.sort_index(axis=0,ascending=True)
Fre11_df=Fre11_sort.reset_index()#将Series数据转换为DataFrame
Fre11_df[0]=Fre11_df[0]/denominator#转换成概率
Fre11_df.columns=['Rds','Fre11']
Fre11_df['cumsum']=np.cumsum(Fre11_df['Fre11'])

denominator=len(lsm6_data)#分母数量 r[4] c_op_dis_set
Data12=pd.Series(lsm6_data)#将数据转换为Series利用分组频数计算
Fre12=Data12.value_counts()
Fre12_sort=Fre12.sort_index(axis=0,ascending=True)
Fre12_df=Fre12_sort.reset_index()#将Series数据转换为DataFrame
Fre12_df[0]=Fre12_df[0]/denominator#转换成概率
Fre12_df.columns=['Rds','Fre12']
Fre12_df['cumsum']=np.cumsum(Fre12_df['Fre12'])

denominator=len(error_lsm3_data)
Data13=pd.Series(error_lsm3_data)
Fre13=Data13.value_counts()
Fre13_sort=Fre13.sort_index(axis=0,ascending=True)
Fre13_df=Fre13_sort.reset_index()
Fre13_df[0]=Fre13_df[0]/denominator
Fre13_df.columns=['Rds','Fre13']
Fre13_df['cumsum']=np.cumsum(Fre13_df['Fre13'])

denominator=len(error2_lsm3_data)
Data14=pd.Series(error2_lsm3_data)
Fre14=Data14.value_counts()
Fre14_sort=Fre14.sort_index(axis=0,ascending=True)
Fre14_df=Fre14_sort.reset_index()
Fre14_df[0]=Fre14_df[0]/denominator
Fre14_df.columns=['Rds','Fre14']
Fre14_df['cumsum']=np.cumsum(Fre14_df['Fre14'])

denominator=len(error3_lsm3_data)
Data15=pd.Series(error3_lsm3_data)
Fre15=Data15.value_counts()
Fre15_sort=Fre15.sort_index(axis=0,ascending=True)
Fre15_df=Fre15_sort.reset_index()
Fre15_df[0]=Fre15_df[0]/denominator
Fre15_df.columns=['Rds','Fre15']
Fre15_df['cumsum']=np.cumsum(Fre15_df['Fre15'])


pax9.plot(Fre7_df['Rds'],Fre7_df['cumsum'],color='#00cc33',label='3 Devices',linestyle = "solid")
pax9.plot(Fre10_df['Rds'],Fre10_df['cumsum'],color='#6699ff',label='4 Devices',linestyle = "dashed")
pax9.plot(Fre11_df['Rds'],Fre11_df['cumsum'],color='#57ba06',label='5 Devices',linestyle = "dashdot")
pax9.plot(Fre12_df['Rds'],Fre12_df['cumsum'],color='#8d17bf',label='6 Devices',linestyle = "dotted")

pax9.set_xlabel("OC: Ranging error (m)", font1)
pax9.set_ylabel("Cumulative Probability", font1)
pax9.set_xlim(0,15)
pax9.legend(fontsize = 12, loc = 'upper left',framealpha=0)

pax10.plot(Fre8_df['Rds'],Fre8_df['cumsum'],linestyle='solid', color='#dc143c',label='Original')
pax10.plot(Fre7_df['Rds'],Fre7_df['cumsum'],linestyle = "--", color='#00cc33',label='Capo_Sigma0.0')
pax10.plot(Fre13_df['Rds'],Fre13_df['cumsum'],linestyle = "dotted", color='#8d17bf',label='Capo_Sigma0.1')
pax10.plot(Fre14_df['Rds'],Fre14_df['cumsum'],linestyle = "dashdot", color='#57ba06',label='Capo_Sigma0.5')
pax10.plot(Fre15_df['Rds'],Fre15_df['cumsum'],linestyle = (0, (3, 1, 1, 1)), color='#30938d',label='Capo_Sigma2.0')

pax10.set_xlabel("OC: Ranging error (m)", font1)
pax10.set_ylabel("Cumulative Probability", font1)
pax10.set_xlim(0,20)
pax10.legend(fontsize = 12, loc = 'lower right',framealpha=0)

##########################################################################################

#print(a3_error_list[7], a4_error_list[7], a5_error_list[7], a6_error_list[7])
print(t_o._x, t_o._y)

plt.show()



