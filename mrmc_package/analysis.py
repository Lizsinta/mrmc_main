import os
from math import sqrt, pi, atan, cos, sin, acos

import numpy as np

from mrmc_package import cal_angle, scatter, cylinder, line, bar


'''def read_shell_data():
    distance = np.array([])
    rdf = []
    for index in range(1):
        with open(r'D:\Monte Carlo\cu\temperature-3\result\result%d.txt' % index, 'r') as f:
            while True:
                lines = f.readline()
                if not lines.find('Best') == -1:
                    break
            f.readline()
            while True:
                data = f.readline()
                if data.isspace():
                    break
                distance = np.append(distance, float(data.split()[5]))
    rdf = list(np.unique(np.trunc(distance * 10) / 10, return_counts=True))
    r1_aver = distance[:12].mean()
    c21 = distance[:12].var()
    c31 = ((distance[:12] - r1_aver) ** 3).mean()
    plt.bar(rdf[0], rdf[1], width=0.075, align='center')
    print(r1_aver, c21, c31)
    return r1_aver, c21, c31, rdf'''

def read_coordinate():
    group = np.array([])
    for index in range(50):
        with open(r'D:\Monte Carlo\cu\total-6 each-3\result\result%d.txt' % index, 'r') as f:
            coordinate = np.array([])
            while True:
                lines = f.readline()
                if not lines.find('final') == -1:
                    break
            while True:
                data = f.readline()
                if data.isspace():
                    break
                temp = data.split()
                coordinate = np.append(coordinate, [float(temp[0]), float(temp[1]), float(temp[2])])
            coordinate = coordinate.reshape((55, 3))
        '''if group.size == 0:
            group = coordinate.copy()
        else:
            group = np.concatenate((group, coordinate))'''
        if len(group) == 0:
            group = [coordinate]
        else:
            group.append(coordinate)
    return group


def collect_pov():
    coordinate = np.array([])
    for index in range(50):
        with open(r'image\image%d.pov' % index, 'r') as f:
            while True:
                lines = f.readline()
                if not lines.find('no cell vertices') == -1:
                    break
            while True:
                data = f.readline()
                if not data.find('cylinder') == -1:
                    break
                coordinate = np.append(coordinate, data)
    f = open(r'../image/image_origin.pov', 'r')
    w = open(r'../image/image_write.pov', 'w')
    while True:
        lines = f.readline()
        if not lines.find('Rbond') == -1:
            w.write(lines.split('=')[0] + '= 0.010;')
        else:
            w.write(lines)
        if not lines.find('no cell vertices') == -1:
            break
    while True:
        data = f.readline()
        if not data.find('cylinder') == -1:
            temp = data
            break
        alpha = data.split(', 0.0,')
        w.write(alpha[0] + ', 0.7,' + alpha[1])
    for a_line in coordinate:
        w.write(a_line)
    rgb = temp.split('color rgb <0.78, 0.50, 0.20>')
    w.write(rgb[0] + 'color rgb <0, 0, 0>' + rgb[1])
    while True:
        data = f.readline()
        if data.find('cylinder') == -1:
            w.write(data)
            break
        rgb = data.split('color rgb <0.78, 0.50, 0.20>')
        w.write(rgb[0] + 'color rgb <0, 0, 0>' + rgb[1])
    f.close()
    w.close()

'''def image_list():
    from mrmc_package.instance import ATOMS
    from ase.io import write
    from ase.io.pov import get_bondpairs
    co_group = read_coordinate()
    cell = ATOMS('Cu')
    bondpairs = get_bondpairs(cell.atoms, radius=1.1)
    r = [{'Cu': 0.5}[at.symbol] for at in cell.atoms]
    write(r'../image/image_origin.pov', cell.atoms, format='pov', radii=r, rotation='10z,-80x',
          povray_settings=dict(canvas_width=200, bondatoms=bondpairs, transparent=0.7))
    for i in range(len(co_group)):
        bondpairs = get_bondpairs(cell.atoms, radius=1.1)
        cell.atoms.set_positions(co_group[i])
        r = [{'Cu': 0.1}[at.symbol] for at in cell.atoms]
        write(r'image\image%d.pov' % i, cell.atoms, format='pov', radii=r, rotation='10z,-80x',
              povray_settings=dict(canvas_width=200, bondatoms=bondpairs, transparent=0))'''

def read_chi(f_rep, rep_size, pol_size):
    chi_x = []
    chi_y = []
    chi_z = []
    with open(f_rep + r'\log.txt', 'r') as f:
        for i in range(rep_size):
            f.readline()
            for j in range(pol_size):
                f.readline()
                chi_i = np.array([])
                while True:
                    data = f.readline()
                    if data.isspace() or not data:
                        break
                    temp = data.split()
                    chi_i = np.append(chi_i, float(temp[1]))
                if j == 0:
                    chi_x.append(chi_i)
                elif j == 1:
                    chi_y.append(chi_i)
                else:
                    chi_z.append(chi_i)
    return np.asarray(chi_x), np.asarray(chi_y), np.asarray(chi_z)


def read_rep_substrate(f_rep, material, filter=True):
    rep_size = len(os.listdir(f_rep + r'\result')) - 4
    dist_rep = []
    coor_filter = []
    ele_rep = np.array([])
    absorb = 1 if material =='CuAlO' else 2
    if rep_size <= 0:
        status = False
    else:
        status = True
        coor_rep = []
        for i in range(rep_size):
            with open(f_rep + r'\result\result%d.txt' % i, 'r') as f:
                coor = np.array([])
                ele = np.array([])
                while True:
                    lines = f.readline()
                    if not lines.find('final') == -1:
                        break
                while True:
                    data = f.readline()
                    if data.isspace() or not data:
                        break
                    temp = data.split()
                    coor = np.append(coor, np.array([float(temp[0]), float(temp[1]), float(temp[2])]))
                    ele = np.append(ele, temp[4][:-1])
                coor = coor.reshape((int(coor.size / 3), 3))
            if coor.shape[0] > 3:
                if i == 0:
                    coor_rep.append(coor)
                    ele_rep = ele.copy()
                else:
                    coor_rep.append(coor[-absorb:])
            else:
                coor_rep.append(coor[np.array([0, 2, 1])])
                ele_rep = np.array(['Cu', 'S', 'O'])
        if coor_rep[0].shape[0] > 3:
            index = tca_filter(coor_rep, ele_rep)
        if not filter or coor_rep[0].shape[0] <= 3:
            index = np.arange(len(coor_rep))
        if not index[0] == 0:
            coor_rep[index[0]] = np.vstack((coor_rep[0][:-absorb], coor_rep[index[0]]))
        for i in index:
            coor_filter.append(coor_rep[i])
            dist_rep.append(np.array([sqrt((_ ** 2).sum()) for _ in coor_rep[i]]))
    return coor_filter, dist_rep, ele_rep, status


def tca_filter(coor, ele):
    y1 = -1.93
    y2l = -6.45
    rx = 4.95
    rxl = 6.09
    elevation = 65 / 180 * pi
    center = np.array([-0.52, y1])
    filtered = np.array([], dtype=int)
    index = np.array([], dtype=int)
    coor_rot = np.zeros(2)
    substrate = coor[0][:-2]
    for i in range(len(coor)):
        coordinate = coor[i][-2].copy()
        ne = np.array([], dtype=int)
        for j in range(ele.size - 2):
            if ele[j] == 'O' and sqrt(((coor[i][-2] - substrate[j]) ** 2).sum()) < 2.4:
                coordinate = np.vstack((coordinate, substrate[j]))
                ne = np.append(ne, j)
        k = np.argmin(np.array([sqrt(((coordinate[_] - coor[i][-2]) ** 2).sum()) for _ in range(1, coordinate.shape[0])]))
        coor[i][-2] -= (substrate[ne[k]] - substrate[41])
        coor[i][-1] -= (substrate[ne[k]] - substrate[41])
    for i in range(ele.size):
        if ele[i] == 'Ti' and (coor[0][i][1] == -13.011 or abs(coor[0][i][1]) == 6.505 or coor[0][i][1] == 0):
            index = np.append(index, i)
    for i in range(len(coor)):
        temp = coor[i][-1:].copy()
        for j in index:
            #if sqrt(((coor[0][j] - temp[0]) ** 2).sum()) < 7 and sqrt((coor[0][j][1] - temp[0][1]) ** 2) < 3.253:
            temp = np.vstack((temp, coor[0][j]))
        temp = np.vstack((temp, np.array([[-5.9, 0, 0], [-5.9, 6.505, 0]])))
        temp -= temp[0]
        for j in range(1, temp.shape[0]):
            azi = atan(temp[j][1] / temp[j][0]) if not temp[j][0] == 0 else 0
            if temp[j][0] < 0:
                azi += pi
            coor_rot[0] = temp[j][0] * cos(azi) + temp[j][1] * sin(azi)
            coor_rot[1] = temp[j][2]
            if y2l < coor_rot[1] < y1:
                vect = coor_rot - center
                if rx < sqrt((vect ** 2).sum()) < rxl:
                    if 0 < -atan(vect[1] / vect[0]) < elevation:
                        filtered = np.append(filtered, i)
                        if coor[i][-2][0] < substrate[41][0] and coor[i][-2][1] < substrate[41][1]:
                            print(coor[i][-1:], temp[j] + coor[i][-1:])
                        break
    return filtered



def select_atom(substrate, coor, ele, nearest=True):
    coordinate = []
    dist = []
    index = np.array([], dtype=int)
    for i in range(ele.size):
        if ele[i] == 'O':
            index = np.append(index, i)
    for i in range(coor.shape[0]):
        temp = coor[i].copy()
        if nearest:
            temp = np.vstack((temp, substrate[index[np.argmin(
                np.array([sqrt(((substrate[_] - temp[0]) ** 2).sum()) for _ in index]))]]))
        else:
            for _ in index:
                if sqrt(((substrate[_] - temp[0]) ** 2).sum()) < 3:
                    temp = np.vstack((temp, substrate[_]))
        temp -= temp[0]
        coordinate.append(temp)
        dist.append(np.array([sqrt((temp[_] ** 2).sum()) for _ in range(temp.shape[0])]))
    return np.asarray(coordinate), np.asarray(dist)


def plot_bondanlge(coor, ele):
    angle = np.array([])
    for i in range(coor.shape[0]):
        if ele[i][1] == 'O':
            angle = np.append(angle, 180 - cal_angle(coor[i][1], np.zeros(3), coor[i][2]))
        else:
            for j in range(2, coor[i].shape[0]):
                angle = np.append(angle, 180 - cal_angle(coor[i][j], np.zeros(3), coor[i][1]))
    return bar(angle[0], angle[1], width=0.01)


def plot_TiO2(coor, ele, graph):
    for i in range(ele.size):
        if ele[i] == 'Ti' or ele[i] == 'Al':
            color = 'grey'
        else:
            if coor[i][2] > 0:
                color = 'purple'
            else:
                color = 'red'
        size = 0.4 if ele[i] == 'O' else 0.6
        graph.addItem(scatter(coor[i][0], coor[i][1], coor[i][2], c=color, scale=size))
    for i in range(ele.size):  # substrate bonds
        for j in range(i):
            if not (ele[i] == ele[j]) and sqrt(((coor[i] - coor[j]) ** 2).sum()) < 3.5:
                graph.addItem(cylinder([coor[i][0], coor[j][0]], [coor[i][1], coor[j][1]],
                                       [coor[i][2], coor[j][2]], c='black', width=0.1))
    graph.addItem(line([-5, 5], [0, 0], [0, 0], c='red', width=3))
    graph.addItem(line([0, 0], [-5, 5], [0, 0], c='green', width=3))
    graph.addItem(line([0, 0], [0, 0], [-5, 5], c='blue', width=3))


def plot_Al2O3(coor, ele, graph):
    for i in range(ele.size):
        color = 'red'
        if ele[i] == 'Al':
            if coor[i][2] < 0:
                color = 'dimgray'
            elif coor[i][2] == 0:
                color = 'darkgray'
            else:
                color = 'lightgray'
        size = 0.4 if ele[i] == 'O' else 0.6
        graph.addItem(scatter(coor[i][0], coor[i][1], coor[i][2], c=color, scale=size))
    '''for i in range(ele.size - 2):  # substrate bonds
        for j in range(i):
            if not (ele[i] == ele[j]) and sqrt(((coor[i] - coor[j]) ** 2).sum()) < 3.5:
                graph.addItem(cylinder([coor[i][0], coor[j][0]], [coor[i][1], coor[j][1]],
                                       [coor[i][2], coor[j][2]], c='black', width=0.1))'''
    graph.addItem(line([-5, 5], [0, 0], [0, 0], c='red', width=3))
    graph.addItem(line([0, 0], [-5, 5], [0, 0], c='green', width=3))
    graph.addItem(line([0, 0], [0, 0], [-5, 5], c='blue', width=3))


def plot_on_substrate(coor, substrate, ele, material, graph):
    item_scatter = np.array([])
    item_cylinder = []
    absorb = 1 if material == 'CuAlO' else 2
    for rep in coor:
        flag = False
        for i in range(ele.size - absorb):
            if sqrt(((rep[0] - substrate[i]) ** 2).sum()) < 2.4:
                if ele[i] == 'Al':
                    flag = True
                    break
        if not flag:
            continue
        for i in range(absorb):
            color = ['brown', 'yellow']
            size = [0.6, 0.3]
            item_scatter = np.append(item_scatter, scatter(rep[i][0], rep[i][1], rep[i][2], c=color[i], scale=size[i]))
            graph.addItem(item_scatter[-1])
        '''graph.addItem(cylinder([rep[0][0], rep[1][0]],[rep[0][1], rep[1][1]], [rep[0][2], rep[1][2]],
                                                                  c='red', alpha=1, width=0.2))'''
    for rep in coor:
        item_cylinder_rep = np.array([])
        flag = False
        for i in range(ele.size - absorb):
            if sqrt(((rep[0] - substrate[i]) ** 2).sum()) < 2.4:
                if ele[i] == 'Al':
                    flag = True
                    break
        if not flag:
            continue
        for i in range(ele.size - absorb):
            if sqrt(((rep[0] - substrate[i]) ** 2).sum()) < 3:
                color = 'green' if ele[i] == 'Ti' else 'blue'
                color = 'red' if sqrt(((rep[0] - substrate[i]) ** 2).sum()) > 2.4 else color
                alpha = 1 if color == 'blue' else 0
                width = 0.2 if not color == 'red' else 0.05
                item_cylinder_rep = np.append(item_cylinder_rep, cylinder([rep[0][0], substrate[i][0]],
                                                                          [rep[0][1], substrate[i][1]],
                                                                          [rep[0][2], substrate[i][2]],
                                                                          c=color, alpha=alpha, width=width))
                graph.addItem(item_cylinder_rep[-1])
        item_cylinder.append(item_cylinder_rep)
        # graph.addItem(cylinder([rep[0][0], rep[1][0]], [rep[0][1], rep[1][1]],
        # [rep[0][2], rep[1][2]], c='red', width=0.05))
    return item_scatter, item_cylinder


def rdf_polarization(coor, dist):
    dist_o = np.array([])
    dist_s = np.array([])
    azimuth_o = np.array([])
    elevation_o = np.array([])
    azimuth_s = np.array([])
    elevation_s = np.array([])
    bond_angle = np.array([])
    for i in range(coor.shape[0]):
        if not coor[i][1][0] == 0:
            azimuth_s = np.append(azimuth_s, round(atan(coor[i][1][1] / coor[i][1][0]) / pi * 180, 0))
            if coor[i][1][0] < 0:
                azimuth_s[-1] += 180
        else:
            azimuth_s = np.append(azimuth_s, 90)
            if coor[i][1][1] < 0:
                azimuth_s[-1] += 180
        elevation_s = np.append(elevation_s, round(acos(coor[i][1][2] / dist[i][1]) / pi * 180, 0))
        dist_s = np.append(dist_s, round(dist[i][1], 2))

        for j in range(2, coor[i].shape[0]):
            if not coor[i][j][0] == 0:
                azimuth_o = np.append(azimuth_o, round(atan(coor[i][j][1] / coor[i][j][0]) / pi * 180, 0))
                if coor[i][j][0] < 0:
                    azimuth_o[-1] += 180
            else:
                azimuth_o = np.append(azimuth_o, 90)
                if coor[i][j][1] < 0:
                    azimuth_o[-1] += 180
            elevation_o = np.append(elevation_o, round(180 - acos(coor[i][j][2] / dist[i][j]) / pi * 180, 0))
            dist_o = np.append(dist_o, round(dist[i][j], 2))
        azimuth_o[np.where(azimuth_o > 90)[0]] -= 180
        azimuth_o[np.where(azimuth_o < 0)[0]] *= -1
        azimuth_s[np.where(azimuth_s > 90)[0]] -= 180
        azimuth_s[np.where(azimuth_s < 0)[0]] *= -1
        bond_angle = np.append(bond_angle, 180 - cal_angle(coor[i][1],coor[i][0],coor[i][2]))

    # print('%.1f %.1f %.1f %.1f' % (elevation_o.mean(), sqrt(elevation_o.var()), elevation_s.mean(), sqrt(elevation_s.var())))

    return [dist_o, azimuth_o, elevation_o, dist_s, azimuth_s, elevation_s, bond_angle]


def plot_rotate(substrate, coor, ele, material, graph, nearest=True, color_assign=''):
    # color = ['blue', 'red', 'orange', 'yellow', 'green', 'cyan', 'purple']
    color = ['blue', 'yellow', 'red', 'red', 'red', 'red', 'red']
    graph.addItem(scatter(0, 0, 0, c='brown', scale=0.6))
    graph.addItem(line([-3, 3], [0, 0], [0, 0], c='black'))
    graph.addItem(line([0, 0], [-3, 3], [0, 0], c='black'))
    graph.addItem(line([0, 0], [0, 0], [-3, 3], c='black'))
    item_rotate = []
    absorb = 1 if material == 'CuAlO' else 2
    for i in range(coor.shape[0]):
        item_rotate_rep = np.array([])
        distance = np.array([sqrt(((substrate[_] - coor[i][0]) ** 2).sum())
                             for _ in range(substrate.shape[0])])
        coordinate = coor[i].copy()
        for j in range(ele.size - absorb):
            if ele[j] == 'O' and distance[j] < 3:
                coordinate = np.vstack((coordinate, substrate[j]))
        cti = np.array([])
        for j in range(ele.size - absorb):
            if (ele[j] == 'Ti' or ele[j] == 'Al') and distance[j] < 2.4:
                cti = np.append(cti, substrate[j])
        if cti.size > 0:
            cti = cti.reshape((int(cti.size / 3), 3))
            cti -= coordinate[0]
        coordinate -= coordinate[0]
        if coordinate.shape[0] > 2:  # select nearest
            k = np.argmin(np.array([sqrt((coordinate[_] ** 2).sum()) for _ in range(absorb, coordinate.shape[0])])) + absorb
        print(coordinate)
        if absorb > 1:
            if coordinate[1][0] > 0:
                coordinate[..., 0] *= -1
                if cti.size > 0:
                    cti[..., 0] *= -1
            if coordinate[1][1] > 0:
                coordinate[..., 1] *= -1
                if cti.size > 0:
                    cti[..., 1] *= -1
            item_rotate_rep = np.append(item_rotate_rep, scatter(coordinate[1][0], coordinate[1][1], coordinate[1][2],
                                                                 c='yellow', scale=0.3))
            graph.addItem(item_rotate_rep[-1])
        if cti.size > 0:
            for j in range(cti.shape[0]):
                item_rotate_rep = np.append(item_rotate_rep, scatter(cti[j][0], cti[j][1], cti[j][2],
                                                                     c='grey', scale=0.6))
                graph.addItem(item_rotate_rep[-1])
        for j in range(absorb, coordinate.shape[0]):
            if nearest and j == k or not nearest:
                colo = 'purple' if coordinate[j][2] + coor[i][0][2] > 0 else 'red'
                if not color_assign == '':
                    colo = color_assign
                item_rotate_rep = np.append(item_rotate_rep, scatter(coordinate[j][0], coordinate[j][1],
                                                                     coordinate[j][2], c=colo, scale=0.4))
                graph.addItem(item_rotate_rep[-1])
        item_rotate.append(item_rotate_rep)
        # ax.plot3D([coor[i][1][0], coor[i][2][0]], [coor[i][1][1], coor[i][2][1]], [coor[i][1][2], coor[i][2][2]], c='black')
    return item_rotate