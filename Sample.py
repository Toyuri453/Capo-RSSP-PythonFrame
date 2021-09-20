import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math



def get_sigma(distance, factor = 0) -> 'float' :
    #print("[PARM] Get Sigma ", np.sqrt((1.0 + (distance / 7) ** 2) * (1 + type_factor_sigma[terminal_type]) + 1 * type_factor_sigma[terminal_type]), "By ", terminal_type, " In ", distance)
    return np.sqrt((2.0 + (distance / 7) ** 2 + factor ))

def gen_normal_point(point_number : 'int', terminal_x, terminal_y, target_x, target_y, factor = 0):
    mu = 0
    sigma = get_sigma(math.sqrt( ((terminal_x-target_x)**2)+((terminal_y-target_y)**2) ), factor)
    normal_set = np.zeros((point_number,2))
    for x in range(normal_set.shape[0]):
        normal_set[x, 0] = target_x + np.random.normal(mu, math.sqrt(sigma))
        normal_set[x, 1] = target_y + np.random.normal(mu, math.sqrt(sigma))
    print("[INFO] Base Terminal: (%s, %s), Weak Terminal: (%s, %s)" %(terminal_x, terminal_y, target_x, target_y) )
    print("[DATA] %s PointSetGenerated As:" %point_number)
    print(normal_set)
    return normal_set

gen_normal_point(100,3,18,5,5)
