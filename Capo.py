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
rcParams.update({'figure.autolayout': True})

type_factor_sigma = {'UWB' : 0, 'BlueTooth' : 0.3}

t_a = Terminal.CartesianPoint(0, 0, "UWB", "TerminalA")
t_b = Terminal.CartesianPoint(15, 4, "UWB", "TerminalB")
t_c = Terminal.CartesianPoint(5, 11, "UWB", "TerminalC")
t_d = Terminal.CartesianPoint(10, 9, "UWB", "TerminalD")
t_e = Terminal.CartesianPoint(8, 10, "UWB", "TerminalE")
t_f = Terminal.CartesianPoint(-4, 3, "UWB", "TerminalF")
t_o = Terminal.CartesianPoint(4, 4, "BlueTooth", "Target")

def get_rho(terminal, target):
    dis = Calculation.get_distance(terminal, target)
    factor = 1 + type_factor_sigma[target._terminal_type]
    rho = factor * math.log10(dis)
    #print (rho,dis,factor)
    return rho

def LSM(t_a: 'Terminal.CartesianPoint', t_b: 'Terminal.CartesianPoint', t_c: 'Terminal.CartesianPoint', t_o: 'Terminal.CartesianPoint'):
    d_a = Calculation.get_distance_sq(t_a, t_o)
    d_b = Calculation.get_distance_sq(t_b, t_o)
    d_c = Calculation.get_distance_sq(t_c, t_o)

    e_d_a = np.random.normal(math.sqrt(d_a), get_rho(t_a, t_o))
    e_d_b = np.random.normal(math.sqrt(d_b), get_rho(t_b, t_o))
    e_d_c = np.random.normal(math.sqrt(d_c), get_rho(t_c, t_o))


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

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c)

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

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), abs(d_op_dis-d_d), abs(e_d_d-d_d)

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

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), abs(d_op_dis-d_d), abs(e_d_d-d_d), abs(e_op_dis-d_e), abs(e_d_e-d_e)

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

    return abs(a_op_dis-d_a), abs(e_d_a-d_a), abs(b_op_dis-d_b), abs(e_d_b-d_b), abs(c_op_dis-d_c), abs(e_d_c-d_c), abs(d_op_dis-d_d), abs(e_d_d-d_d), abs(e_op_dis-d_e), abs(e_d_e-d_e), abs(f_op_dis-d_f), abs(e_d_f-d_f)

#LSM(t_a, t_b, t_c, t_o)

#ax.scatter(terminal._x, terminal._y, s=out_area, color = terminal._terminal_color, alpha=alpha, edgecolors=color, linewidths=[1])



a_op_dis_set = np.zeros((3000,1))
e_d_a_set = np.zeros((3000,1))
b_op_dis_set = np.zeros((3000,1))
e_d_b_set = np.zeros((3000,1))
c_op_dis_set = np.zeros((3000,1))
e_d_c_set = np.zeros((3000,1))

a_sum = 0
b_sum = 0
c_sum = 0
e_a_sum = 0
e_b_sum = 0
e_c_sum = 0

for _ in range(3000):
    result = LSM(t_a, t_b, t_c, t_o)
    a_op_dis_set[_] = result[0]
    e_d_a_set[_] = result[1]
    b_op_dis_set[_] = result[2]
    e_d_b_set[_] = result[3]
    c_op_dis_set[_] = result[4]
    e_d_c_set[_] = result[5]
    a_sum += a_op_dis_set[_]
    b_sum += b_op_dis_set[_]
    c_sum += c_op_dis_set[_]
    e_a_sum += e_d_a_set[_]
    e_b_sum += e_d_b_set[_]
    e_c_sum += e_d_c_set[_]
    print (result)

fig, (ax1) = plt.subplots(1,1, figsize=(5,3))
fig2, (ax2) = plt.subplots(1,1, figsize=(5,3))
fig3, (ax3) = plt.subplots(1,1, figsize=(5,3))
fig4, (ax4) = plt.subplots(1,1, figsize=(5,3))

ax1.set_xlabel("Error distance", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
#ax1.set_title('Error distribution of Terminal A', fontsize=15, fontweight='light')
n, bins, patches = ax1.hist(e_d_a_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='original')
n, bins, patches = ax1.hist(a_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='CAPO')

n = n.astype('int')
ax1.legend(fontsize = 12)
ax1.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
ax1.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

ax2.set_xlabel("Error distance", fontsize=14)
ax2.set_ylabel("Frequency", fontsize=14)
#ax2.set_title('Error distribution of Terminal B', fontsize=15, fontweight='light')
n, bins, patches = ax2.hist(e_d_b_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='original')
n, bins, patches = ax2.hist(b_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='CAPO')

n = n.astype('int')
ax2.legend(fontsize = 12)
ax2.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
ax2.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

ax3.set_xlabel("Error distance", fontsize=14)
ax3.set_ylabel("Frequency", fontsize=14)
#ax3.set_title('Error distribution of Terminal C', fontsize=15, fontweight='light')
n, bins, patches = ax3.hist(e_d_c_set, bins=80, facecolor='#bf1722',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='original')
n, bins, patches = ax3.hist(c_op_dis_set, bins=80, facecolor='#17aebf',
                           edgecolor='#000000',histtype='stepfilled', linewidth=0.5, alpha=0.30, label='CAPO')

n = n.astype('int')
ax3.legend(fontsize = 12)
ax3.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
ax3.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

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
ax4.set_xlabel("Round", fontsize=14)
ax4.set_ylabel("Average Error/ m", fontsize=14)
a3_error = 0
a4_error = 0
a5_error = 0
a6_error = 0
round_list = []
a3_error_list = []
a4_error_list = []
a5_error_list = []
a6_error_list = []
for _ in range(3000):
    r3 = LSM(t_a, t_b, t_c, t_o)
    r4 = LSM_4(t_a, t_b, t_c, t_d, t_o)
    r5 = LSM_5(t_a, t_b, t_c, t_d, t_e, t_o)
    r6 = LSM_6(t_a, t_b, t_c, t_d, t_e, t_f, t_o)
    a3_error += (r3[0]+r3[2]+r3[4])/3
    a4_error += (r4[0]+r4[2]+r4[4]+r4[6])/4
    a5_error += (r5[0]+r5[2]+r5[4]+r5[6]+r5[8])/5
    a6_error += (r6[0]+r6[2]+r6[4]+r6[6]+r6[8]+r6[10])/6
    if _ == 9 :
        ax4.scatter(_, a3_error / 10, color = '#bf1722', linewidths=[1], label = "3 terminals")
        ax4.scatter(_, a4_error / 10, color='#17aebf', linewidths=[1], marker = "^", label = "4 terminals")
        ax4.scatter(_, a5_error / 10, color='#57ba06', linewidths=[1], marker = "s", label = "5 terminals")
        ax4.scatter(_, a6_error / 10, color='#8d17bf', linewidths=[1], marker = "x", label = "6 terminals")
        round_list.append(_)
        a3_error_list.append(a3_error / 10)
        a4_error_list.append(a4_error / 10)
        a5_error_list.append(a5_error / 10)
        a6_error_list.append(a6_error / 10)
    if _ == 29 :
        ax4.scatter(_, a3_error / 30, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 30, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 30, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 30, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 30)
        a4_error_list.append(a4_error / 30)
        a5_error_list.append(a5_error / 30)
        a6_error_list.append(a6_error / 30)
    if _ == 99 :
        ax4.scatter(_, a3_error / 100, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 100, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 100, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 100, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 100)
        a4_error_list.append(a4_error / 100)
        a5_error_list.append(a5_error / 100)
        a6_error_list.append(a6_error / 100)
    if _ == 199 :
        ax4.scatter(_, a3_error / 200, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 200, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 200, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 200, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 200)
        a4_error_list.append(a4_error / 200)
        a5_error_list.append(a5_error / 200)
        a6_error_list.append(a6_error / 200)
    if _ == 499 :
        ax4.scatter(_, a3_error / 500, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 500, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 500, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 500, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 500)
        a4_error_list.append(a4_error / 500)
        a5_error_list.append(a5_error / 500)
        a6_error_list.append(a6_error / 500)
    if _ == 999 :
        ax4.scatter(_, a3_error / 1000, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 1000, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 1000, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 1000, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 1000)
        a4_error_list.append(a4_error / 1000)
        a5_error_list.append(a5_error / 1000)
        a6_error_list.append(a6_error / 1000)
    if _ == 1799 :
        ax4.scatter(_, a3_error / 1800, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 1800, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 1800, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 1800, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 1800)
        a4_error_list.append(a4_error / 1800)
        a5_error_list.append(a5_error / 1800)
        a6_error_list.append(a6_error / 1800)
    if _ == 2999 :
        ax4.scatter(_, a3_error / 3000, color = '#bf1722', linewidths=[1])
        ax4.scatter(_, a4_error / 3000, color='#17aebf', linewidths=[1], marker = "^")
        ax4.scatter(_, a5_error / 3000, color='#57ba06', linewidths=[1], marker = "s")
        ax4.scatter(_, a6_error / 3000, color='#8d17bf', linewidths=[1], marker = "x")
        round_list.append(_)
        a3_error_list.append(a3_error / 3000)
        a4_error_list.append(a4_error / 3000)
        a5_error_list.append(a5_error / 3000)
        a6_error_list.append(a6_error / 3000)
ax4.xaxis.grid(True, which='major', linestyle=(0, (8, 8)))
ax4.yaxis.grid(True, which='major', linestyle=(0, (8, 8)))
ax4.plot(round_list, a3_error_list, c = '#bf1722', linewidth = 0.8)
ax4.plot(round_list, a4_error_list, c = '#17aebf', linewidth = 0.8)
ax4.plot(round_list, a5_error_list, c = '#57ba06', linewidth = 0.8)
ax4.plot(round_list, a6_error_list, c = '#8d17bf', linewidth = 0.8)
ax4.legend(fontsize = 10,framealpha=0)
print(a3_error_list[7], a4_error_list[7], a5_error_list[7], a6_error_list[7])
plt.show()


