import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns
from matplotlib.patches import Rectangle

def set_bgc(ax, c : "str" = "white", alpha : "float" = 1.0):
    ax.add_patch(Rectangle((0, 0), 100, 100, color=c, alpha=alpha))


def draw_rect(ax, c, alpha, start, width, inter, mode):
    count = int(100/inter)+1
    if mode == "h":
        for _ in range(count):
            ax.add_patch(Rectangle((0, start + _ * inter), 100, width, color=c, alpha=alpha))
            print(start, _, inter, width)
    if mode == "v":
        for _ in range(count):
            ax.add_patch(Rectangle((start + _ * inter, 0), width, 100, color=c, alpha=alpha))
            print(start, _, inter, width)


fig, ((ax1, ax2, ax3),(ax4,ax5, ax6)) = plt.subplots(2,3,figsize=(30,30))
ax1.set_xlim([0,100])
ax1.set_ylim([0,100])
set_bgc(ax1, "#cbb097", 0.25)
#d57e64
a1 = 0.8
a2 = 0.35

draw_rect(ax1,"#8c6d7b", a1, 10, 7, 45, "h")
draw_rect(ax1,"#8c6d7b", a1, 20, 7, 45, "h")
draw_rect(ax1,"#8377ac", a1, 8, 2, 45, "h")
draw_rect(ax1,"#bb9993", a1, 7, 1, 45, "h")
draw_rect(ax1,"#bb9993", a1, 5, 1, 45, "h")
draw_rect(ax1,"#8377ac", a1, 3, 2, 45, "h")
draw_rect(ax1,"#f7d798", a1, 27, 7, 45, "h")
draw_rect(ax1,"#c47c67", a1, 26.5, 2, 45, "h")
draw_rect(ax1,"#c47c67", a1, 31, 1, 45, "h")

draw_rect(ax1,"#8c6d7b", a2, 10, 7, 45, "v")
draw_rect(ax1,"#8c6d7b", a2, 20, 7, 45, "v")
draw_rect(ax1,"#8377ac", a2, 8, 2, 45, "v")
draw_rect(ax1,"#bb9993", a2, 7, 1, 45, "v")
draw_rect(ax1,"#bb9993", a2, 5, 1, 45, "v")
draw_rect(ax1,"#8377ac", a2, 3, 2, 45, "v")
draw_rect(ax1,"#f7d798", a2, 27, 7, 45, "v")
draw_rect(ax1,"#c47c67", a2, 26.5, 2, 45, "v")
draw_rect(ax1,"#c47c67", a2, 31, 1, 45, "v")

#e3d7b8
#fffefa
a3=0.6
ax2.set_xlim([0,100])
ax2.set_ylim([0,100])
set_bgc(ax2, "#e3d7b8", 0.45)
draw_rect(ax2,"#282827", a1, 10, 3, 45, "h")
draw_rect(ax2,"#fffefa", a1, 13, 3, 45, "h")
draw_rect(ax2,"#282827", a1, 16, 3, 45, "h")
draw_rect(ax2,"#fffefa", a1, 19, 3, 45, "h")
draw_rect(ax2,"#282827", a1, 22, 3, 45, "h")
draw_rect(ax2,"#c4665c", a1, 40, 1, 45, "h")

draw_rect(ax2,"#282827", a3, 10, 3, 45, "v")
draw_rect(ax2,"#fffefa", a3, 13, 3, 45, "v")
draw_rect(ax2,"#282827", a3, 16, 3, 45, "v")
draw_rect(ax2,"#fffefa", a3, 19, 3, 45, "v")
draw_rect(ax2,"#282827", a3, 22, 3, 45, "v")
draw_rect(ax2,"#c4665c", a3, 40, 1, 45, "v")

#8097bb
ax3.set_xlim([0,100])
ax3.set_ylim([0,100])
a1=0.35
a2=0.8
set_bgc(ax3, "#d4b99e", 0.35)

draw_rect(ax3,"#b37055", a2, 3.5, 0.5, 35, "v")
draw_rect(ax3,"#ecc196", a2, 4, 8, 35, "v")
draw_rect(ax3,"#b37055", a2, 12, 0.5, 35, "v")
draw_rect(ax3,"#d17d5b", a2, 6.5, 0.4, 35, "v")
draw_rect(ax3,"#d17d5b", a2, 9, 0.4, 35, "v")
draw_rect(ax3,"#b37055", a2, 17.5, 0.5, 35, "v")
draw_rect(ax3,"#536479", a2, 18, 5, 35, "v")
draw_rect(ax3,"#d17d5b", a2, 27, 0.5, 35, "v")
draw_rect(ax3,"#ecc196", a2, 23, 4, 35, "v")
draw_rect(ax3,"#536479", a2, 27.5, 5, 35, "v")
draw_rect(ax3,"#d17d5b", a2, 32.5, 0.5, 35, "v")
draw_rect(ax3,"#d17d5b", a2, 24, 0.5, 35, "v")
draw_rect(ax3,"#d17d5b", a2, 25.5, 0.5, 35, "v")

draw_rect(ax3,"#b37055", a1, 3.5, 0.5, 35, "h")
draw_rect(ax3,"#ecc196", a1, 4, 8, 35, "h")
draw_rect(ax3,"#b37055", a1, 12, 0.5, 35, "h")
draw_rect(ax3,"#d17d5b", a1, 6.5, 1, 35, "h")
draw_rect(ax3,"#d17d5b", a1, 8.5, 1, 35, "h")
draw_rect(ax3,"#b37055", a1, 17.5, 0.5, 35, "h")
draw_rect(ax3,"#536479", a1, 18, 5, 35, "h")
draw_rect(ax3,"#d17d5b", a1, 27, 0.5, 35, "h")
draw_rect(ax3,"#ecc196", a1, 23, 4, 35, "h")
draw_rect(ax3,"#536479", a1, 27.5, 5, 35, "h")
draw_rect(ax3,"#d17d5b", a1, 32.5, 0.5, 35, "h")
draw_rect(ax3,"#d17d5b", a1, 24, 0.5, 35, "h")
draw_rect(ax3,"#d17d5b", a1, 25.5, 0.5, 35, "h")


ax4.set_xlim([0,100])
ax4.set_ylim([0,100])
a1=0.35
a2=0.65
inter = 28
mode = "v"
set_bgc(ax4, "#dcd9d9", 0.35)
#507d96
draw_rect(ax4,"#507d96", a2, 3, 1, inter, mode)
draw_rect(ax4,"#507d96", a2, 5, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 6.3, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 7.6, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 8.9, 1, inter, mode)

draw_rect(ax4,"#507d96", a2, 15, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 16.5, 4, inter, mode)
draw_rect(ax4,"#507d96", a2, 18.5, 5, inter, mode)
draw_rect(ax4,"#507d96", a2, 24, 0.2, inter, mode)
draw_rect(ax4,"#507d96", a2, 24.8, 0.2, inter, mode)
a2=0.35
mode="h"
draw_rect(ax4,"#507d96", a2, 3, 1, inter, mode)
draw_rect(ax4,"#507d96", a2, 5, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 6.3, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 7.6, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 8.9, 1, inter, mode)

draw_rect(ax4,"#507d96", a2, 15, 0.3, inter, mode)
draw_rect(ax4,"#507d96", a2, 16.5, 4, inter, mode)
draw_rect(ax4,"#507d96", a2, 18.5, 5, inter, mode)
draw_rect(ax4,"#507d96", a2, 24, 0.2, inter, mode)
draw_rect(ax4,"#507d96", a2, 24.8, 0.2, inter, mode)


ax5.set_xlim([0,100])
ax5.set_ylim([0,100])
a1=0.35
a2=0.65
inter = 26
mode = "v"
set_bgc(ax5, "#e1dedd", 0.6)
#507d96
draw_rect(ax5,"#c8405e", 0.8, 3, 0.3, inter, mode)
draw_rect(ax5,"#c8405e", 0.8, 5.5, 0.3, inter, mode)
draw_rect(ax5,"#c8405e", a2, 8, 6, inter, mode)
draw_rect(ax5,"#c8405e", a2, 11, 6, inter, mode)
draw_rect(ax5,"#c8405e", 0.75, 14, 6, inter, mode)
draw_rect(ax5,"#c8405e", 1, 14.5, 2, inter, mode)
draw_rect(ax5,"#fafafa", 0.65, 15.5, 1, inter, mode)
draw_rect(ax5,"#c8405e", 0.3, 23, 0.2, inter, mode)
draw_rect(ax5,"#c8405e", 0.3, 25, 0.2, inter, mode)
mode = "h"
a2=0.2
draw_rect(ax5,"#c8405e", a2, 3, 0.3, inter, mode)
draw_rect(ax5,"#c8405e", a2, 5.5, 0.3, inter, mode)
draw_rect(ax5,"#c8405e", a2, 8, 6, inter, mode)
draw_rect(ax5,"#c8405e", a2, 11, 6, inter, mode)
draw_rect(ax5,"#c8405e", a2, 14, 6, inter, mode)
draw_rect(ax5,"#c8405e", a2, 14.5, 2, inter, mode)
draw_rect(ax5,"#fafafa", a2, 15.5, 1, inter, mode)
draw_rect(ax5,"#c8405e", 0.3, 23, 0.2, inter, mode)
draw_rect(ax5,"#c8405e", 0.3, 25, 0.2, inter, mode)


ax6.set_xlim([0,100])
ax6.set_ylim([0,100])
a1=0.35
a2=0.65
inter = 30
mode = "v"
set_bgc(ax6, "#87afec", 0.6)
##ce94df
draw_rect(ax6,"#efd4ec", 0.8, 2, 3, inter, mode)
draw_rect(ax6,"#624371", a2, 2, 0.3, inter, mode)
draw_rect(ax6,"#624371", a2, 3, 0.7, inter, mode)
draw_rect(ax6,"#8d5b87", a2, 6, 15, inter, mode)
draw_rect(ax6,"#8d5b87", a2, 9, 15, inter, mode)
draw_rect(ax6,"#efd4ec", 1, 10, 4, inter, mode)
draw_rect(ax6,"#be70dc", a2, 11, 2, inter, mode)
draw_rect(ax6,"#8d5b87", a2, 11.5, 12.5, inter, mode)
draw_rect(ax6,"#ce94df", a2, 20, 3, inter, mode)
draw_rect(ax6,"#efd4ec", 1, 21, 1, inter, mode)
a2=0.2
mode="h"
draw_rect(ax6,"#efd4ec", a2, 2, 3, inter, mode)
draw_rect(ax6,"#624371", a2, 2, 0.3, inter, mode)
draw_rect(ax6,"#624371", a2, 3, 0.7, inter, mode)
draw_rect(ax6,"#8d5b87", a2, 6, 15, inter, mode)
draw_rect(ax6,"#8d5b87", a2, 9, 15, inter, mode)
draw_rect(ax6,"#efd4ec", a2, 10, 4, inter, mode)
draw_rect(ax6,"#be70dc", a2, 11, 2, inter, mode)
draw_rect(ax6,"#8d5b87", a2, 11.5, 12.5, inter, mode)
plt.show()