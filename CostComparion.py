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
fig, (ax1) = plt.subplots(1,1, figsize=(5,3))
fig2, (ax2) = plt.subplots(1,1, figsize=(5,3))

ax1.set_xlabel("Round", fontsize=14, weight = 'normal')
ax1.set_ylabel("Cost", fontsize=14, weight = 'normal')


n = 3

cost_s = 1
cost_w = 0.5

m = np.linspace(1, 100, 100)
ax1.plot(m, n*(n-1)*cost_s+m*n*cost_w, marker = "." ,label="3 terminal", color='#bf1722', markevery=10 ,markersize=8)
n=4
ax1.plot(m, n*(n-1)*cost_s+m*n*cost_w, marker = "^", label="4 terminal", color='#17aebf', markevery=10,markersize=8)
n=5
ax1.plot(m, n*(n-1)*cost_s+m*n*cost_w, marker = "s", label="5 terminal", color='#57ba06', markevery=10,markersize=8)
n=6
ax1.plot(m, n*(n-1)*cost_s+m*n*cost_w, marker = "x", label="6 terminal", color='#8d17bf', markevery=10,markersize=8)
n = 3

ax2.set_xlabel("Terminal quantity", fontsize=14, weight = 'normal')
ax2.set_ylabel("Cost", fontsize=14, weight = 'normal')

m1 = 1
n1 = np.linspace(3, 10, 100)
ax2.plot(n1, n1*(n1-1)*cost_s+m*n1*cost_w, color='#bf1722',markersize=8)


ax1.legend(fontsize = 12,framealpha=0)
ax1.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
ax1.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))

ax2.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
ax2.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
plt.show()