import matplotlib.pyplot as plt # importing matplotlib
import numpy as np # importing numpy
import GenNorm
import Terminal
x = np.random.normal(0, 1, 30) # normal distribution
plt.figure(figsize=(14,7)) # Make it 14x7 inch
plt.style.use('seaborn-whitegrid') # nice and clean grid
a = Terminal.CartesianPoint(5, 5, "UWB", "T1")
b = Terminal.CartesianPoint(10, 25, "UWB", "T2")
c = Terminal.CartesianPoint(5, 5, "BlueTooth", "T2")
aa= GenNorm.gen_dis_set(20000, a, b)
bb= GenNorm.gen_dis_set(20000, c, b)
n, bins, patches = plt.hist(aa, bins=200, facecolor='#aaaaff', edgecolor='#e0e0e0', linewidth=0.2, alpha=0.4, label = "aa")
n1, bins1, patches1 = plt.hist(bb, bins=200, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.2, alpha=0.4, label = "bb")
print(aa)
print(bb)
plt.legend()
n = n.astype('int') # it MUST be integer
print(n)
# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
# Make one bin stand out
for i in range(len(patches1)):
    patches1[i].set_facecolor(plt.cm.plasma(n1[i]/max(n1)))

# Add annotation
#plt.annotate('Important Bar!', xy=(0.57, 175), xytext=(2, 130), fontsize=15, arrowprops={'width':0.4,'headwidth':7,'color':'#333333'})
# Add title and labels with custom font sizes
plt.title('Normal Distribution', fontsize=12)
plt.xlabel('Bins', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.show()
