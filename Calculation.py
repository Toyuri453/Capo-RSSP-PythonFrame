import math
import Terminal
import numpy as np

type_factor_sigma = {'UWB' : 0, 'BlueTooth' : 0.3}

def get_distance(terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint'):
    #print(terminalA._x, terminalB._x, terminalA._y, terminalB._y)
    dis = math.sqrt( ((terminalA._x-terminalB._x)**2)+((terminalA._y-terminalB._y)**2) )
    print("[DATA] Get Distance :" + terminalA._terminal_name + ": " , terminalA._x , terminalA._y , terminalB._terminal_name + ": " , terminalB._x , terminalB._y , "Dis: ", dis)
    return dis

def get_distance_sq(terminalA : 'Terminal.CartesianPoint', terminalB : 'Terminal.CartesianPoint'):
    #print(terminalA._x, terminalB._x, terminalA._y, terminalB._y)
    print("[DATA] Get Squared Distance :" + terminalA._terminal_name + ": " , terminalA._x , terminalA._y , terminalB._terminal_name + ": " , terminalB._x , terminalB._y , "Dis: ", ((terminalA._x-terminalB._x)**2)+((terminalA._y-terminalB._y)**2))
    return ((terminalA._x-terminalB._x)**2)+((terminalA._y-terminalB._y)**2)

def get_distance_from_origin(terminal : 'Terminal.CartesianPoint'):
    return math.sqrt((terminal._x**2)+(terminal._y**2))

def get_distance_from_origin_by_coord(x : 'float', y : 'float'):
    return math.sqrt((x**2)+(y**2))

def get_distance_by_set(set_a : 'np.array', set_b : 'np.array') -> 'float':
    print("[PARM] Get Distance By ", set_a, set_b)
    return math.sqrt((set_a[0]-set_b[0])**2 + (set_a[1]-set_b[1])**2) #done

def get_sigma(distance, terminal_type = 'UWB') -> 'float' :
    print("[PARM] Get Sigma ", np.sqrt((1.0 + (distance / 7) ** 2) * (1 + type_factor_sigma[terminal_type]) + 1 * type_factor_sigma[terminal_type]), "By ", terminal_type, " In ", distance)
    return np.sqrt((1.0 + (distance / 7) ** 2) * (1 + type_factor_sigma[terminal_type]) + 1 * type_factor_sigma[terminal_type])

def get_distance_from_origin_by_set(set : 'np.array'):
    return math.sqrt((set[0]) ** 2 + (set[1]** 2))

def get_distance_from_weak_terminal_by_coord(x : 'float', y : 'float', weak_terminal :'Terminal.CartesianPoint') -> 'float':
    return math.sqrt(((x-weak_terminal._x)**2)+((y-weak_terminal._y)**2))

def get_modified_distance_by_set_and_type(normal_set : 'np.array', terminal_type : "str" = "UWB"):
    dis_set = np.zeros((len(normal_set), 1))
    mod_set = np.zeros((len(normal_set), 1))
    for x in range(normal_set.shape[0]):
        dis_set[x] = get_distance_from_origin_by_coord(normal_set[x, 0], normal_set[x, 1])
        #dis_set[x] = math.sqrt((normal_set[x, 0] - terminal._x)**2 + (normal_set[x, 1] - terminal._y)**2)
        mod_set[x] = round(float(dis_set[x]*(1-(get_sigma(dis_set[x], terminal_type)**3/15))), 2)
        print("[DATA] Get Mod Distance By Point {2}, {3} Distance: {0}, Mod Distance: {1}".format(dis_set[x], mod_set[x], normal_set[x, 0], normal_set[x, 1]))
    return mod_set, dis_set

def get_modified_coord_by_nor_set_and_terminal(distribute_set : 'np.array', terminal :'Terminal.CartesianPoint'):
    ratio_set = get_modified_distance_by_set_and_type(distribute_set)[0]
    mod_coord_set = np.zeros((len(distribute_set), 2))
    for x in range(distribute_set.shape[0]):
        distance = get_distance_from_origin_by_coord(distribute_set[x, 0], distribute_set[x, 1])
        ratio_set[x] /= distance
        mod_coord_set[x, 0] = ((distribute_set[x, 0] - terminal._x) * ratio_set[x]) + terminal._x
        mod_coord_set[x, 1] = ((distribute_set[x, 1] - terminal._y) * ratio_set[x]) + terminal._y
        print("[DATA] Raw Point:({0}, {1}), Distance: {2}, Mod Ratio: {3}, Mod Coordinate: ({4}, {5})".format(distribute_set[x, 0], distribute_set[x, 1], distance, ratio_set[x], mod_coord_set[x, 0], mod_coord_set[x, 1]))
    return mod_coord_set

def get_ideal_coord_by_set(set : 'np.array') -> 'np.array':
    ideal_set = np.zeros((2, 1))
    x_sum = 0
    y_sum = 0
    for x in range(set.shape[0]):
        x_sum += set[x, 0]
        y_sum += set[x, 1]
    ideal_set[0] = x_sum / len(set)
    ideal_set[1] = y_sum / len(set)
    print("[DATA] Ideal Point Set Generated: ({0}, {1})".format(ideal_set[0], ideal_set[1]))
    return ideal_set

def get_shift_coord_by_radius_and_degree(coord_set : 'np.array', radius : 'float' = 1, angle : 'float' = 0, origin : 'np.array' = None) -> 'np.array':
    # return ([round(math.cos(np.arctan(coord_set[1]/coord_set[0]) + math.radians(angle)), 4) * radius,
    #          round(math.cos(np.arctan(coord_set[0, 1]/coord_set[0, 0]) + math.radians(angle)), 4) * radius])
    origin = ([0, 0]) if origin is None else origin
    print("[PARM] Get Shift Point By Radius ", radius, " And Degree ", angle, " In Coordination ", coord_set, " From ", origin)
    return ([round(math.cos(np.arctan2(coord_set[1] - origin[1], coord_set[0] - origin[0]) + math.radians(angle)), 4) * radius + origin[0],
             round(math.sin(np.arctan2(coord_set[1] - origin[1], coord_set[0] - origin[0]) + math.radians(angle)), 4) * radius + origin[1]])

def get_vector_by_terminal(start : 'Terminal.CartesianPoint', end: 'Terminal.CartesianPoint') -> np.array:
    print("[PARM] Get Terminal Vector From Terminal ", start, " To Terminal ", end)
    return np.array([end._x - start._x, end._y - start._y])

def get_vector_by_set(start : 'np.array', end : 'np.array') -> np.array :
    print("[PARM] Get Point Vector By Set ", start, " To Set ", end)
    return np.array([end[0] - start[0], end[1] - start[1]])

def get_degree_by_vector(vector : 'np.array') -> 'float':
    degree = round(math.degrees(np.arctan2(vector[1], vector[0])), 4)
    #return (degree if degree >= 0 else 360 + degree)
    return degree

def get_degree_diff_by_vector(terminal_vector : 'np.array', ideal_vector : 'np.array') -> 'float':
    #print("###################", get_degree_by_vector(terminal_vector), get_degree_by_vector(ideal_vector))
    return get_degree_by_vector(terminal_vector) - get_degree_by_vector(ideal_vector)

def get_shift_degree(ratio : 'float', degree : 'float') -> 'float':
    print("[PARM] Get Shift Degree ", (ratio) * (10* degree / 2), "By Ratio ", ratio, " And Degree ", degree)
    return (ratio) * (10* degree / 2)

def get_mid_point_by_set(set_a : 'np.array', set_b : 'np.array') -> 'np.array':
    x = float((set_a[0] + set_b[0]))/2
    y = float((set_a[1] + set_b[1]))/2
    mid = np.zeros((2, 1))
    mid[0] = x
    mid[1] = y
    print("[PARM] Get Mid Point By Set ", set_a, " To Set ", set_b)
    return mid

def get_ll_intersection(l1p1 : 'np.array', l1p2 : 'np.array', l2p1 : 'np.array', l2p2 : 'np.array'):
    #print(l1p1, l1p2, l2p1, l2p2, get_shift_coord_by_radius_and_degree(([1, 0]), 1, 90, ([2, 0])))
    denominator = (l1p1[0] - l1p2[0]) * (l2p1[1]- l2p2[1]) - (l1p1[1] - l1p2[1]) * (l2p1[0] - l2p2[0])
    div_x = (l1p1[0] * l1p2[1] - l1p1[1] * l1p2[0]) * (l2p1[0] * l2p2[0]) - (l1p1[0] - l1p2[0]) * (l2p1[0] * l2p2[1] - l2p1[1] * l2p2[0])
    div_y = (l1p1[0] * l1p2[1] - l1p1[1] * l1p2[0]) * (l2p1[1] * l2p2[1]) - (l1p1[1] - l1p2[1]) * (l2p1[0] * l2p2[1] - l2p1[1] * l2p2[0])
    intersection = ([div_x/denominator, div_y/denominator])
    print("[PARM] Get Target Point", intersection, " By ", l1p1, " ", l1p2, " And ", l2p1, " ", l2p2)
    return intersection


def get_optimized_target(start_terminal : 'Terminal.CartesianPoint', end_terminal : 'Terminal.CartesianPoint', start_point : 'np.array', end_point : 'np.array'):
    '''
    :param start_terminal:
    :param end_terminal:
    :param start_point:
    :param end_point:
    :return: shift_start_point, shift_end_point, intersection
    '''
    start_terminal_point = np.array([start_terminal._x, start_terminal._y])
    end_terminal_point = np.array([end_terminal._x, end_terminal._y])
    terminal_distance = get_distance(start_terminal, end_terminal)
    point_distance = get_distance_by_set(start_point, end_point)
    terminal_vector = get_vector_by_terminal(start_terminal, end_terminal)
    point_vector = get_vector_by_set(start_point, end_point)
    mid_point = get_mid_point_by_set(start_point, end_point)
    shift_degree = get_shift_degree((point_distance/terminal_distance), get_degree_diff_by_vector(terminal_vector, point_vector))
    shift_end_point = get_shift_coord_by_radius_and_degree(end_point, point_distance/2, shift_degree, mid_point)
    shift_start_point = get_shift_coord_by_radius_and_degree(start_point, point_distance/2, shift_degree, mid_point)
    intersection = get_ll_intersection(start_terminal_point, shift_start_point, end_terminal_point, shift_end_point)
    return ([shift_start_point, shift_end_point, intersection, point_distance/terminal_distance, get_degree_diff_by_vector(terminal_vector, point_vector)])


