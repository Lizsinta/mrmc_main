

from mrmc_package import k_range, deltaE_shift, back_k_space, intp1D
import numpy as np
from csv import writer
from math import sqrt, acos, pi


def used_file(file_name):
    with open(file_name + r'\chi.dat', 'r') as f_chi:
        file_index = np.array([], dtype=int)
        while True:
            line = f_chi.readline()
            if not line.find('file') == -1:
                break
        while True:
            temp = f_chi.readline()
            if not temp.find('paths used') == -1:
                break
            file_index = np.append(file_index, int(temp.split()[1]))
        return file_index


def read_path(file_name, index):
    with open(file_name + r'\paths.dat', 'r') as f_path:
        path_r = []
        path_angle = []
        f_path.readline()
        f_path.readline()
        i = 0
        while True:
            line = f_path.readline()
            if not line:
                break
            if not line.find('index') == -1:
                if index[i] == int(line.split()[0]):
                    f_path.readline()
                    r = ()
                    angle = ()
                    i += 1
                    for step in range(int(line.split()[1])):
                        data = f_path.readline().split()
                        r += (float(data[6]),)
                        angle += (round(float(data[7]), 0),)
                    path_r.append(r)
                    path_angle.append(angle)
                    del r
                    del angle
        return path_r, path_angle


def cal_angle(coor1, coor2, coor3):
    vect1 = coor2 - coor1
    vect2 = coor3 - coor2
    transvection = round(np.sum(vect1 * vect2), 6)
    module = round(sqrt(np.sum(vect1**2) * np.sum(vect2**2)), 6)
    return round(acos(transvection / module) / pi * 180, 2)

def create_dist_table(new_array):
    atom_count = int(new_array.size / 3)
    table = np.zeros([atom_count, atom_count])
    for row in range(atom_count):
        for column in range(row):
            if not row == column:
                table[row][column] = round(sqrt(np.sum((new_array[row] - new_array[column]) ** 2)), 4)
    return table + table.T


def cal_rpath_para(table_rpath):
    table = []
    for group in table_rpath:
        table.append(list(np.unique(group, return_counts=True)))
    return table


class TABLE:
    def __init__(self, coordinate, k, path_file, sig, energy, s02):
        self.k0 = k
        self.dw = np.exp(-2 * (k ** 2) * (sig ** 2))
        self.energy = energy
        self.s02 = s02
        self.k = np.array([])
        self.file = path_file
        self.file_index = used_file(path_file)
        # dist_table: table of all inter-atomic distance between any two atoms.
        self.distance_table = create_dist_table(coordinate)
        # path_table: according to distance of each steps in paths.dat, get steps index in dist_table (fixed).
        self.path_table = self.create_path_table(path_file, coordinate)
        # rpath_table: D/2 distance of each paths in path_table, when dist_table change, this should update.
        self.rpath_table = self.cal_rpath()
        # rpath_para_table: sum up path with same distance to be coordination number.
        self.rpath_para_table = cal_rpath_para(self.rpath_table)
        self.group_amount = len(self.path_table)
        self.feff = []
        self.read_feff(path_file.split('paths.dat')[0])
        self.chi_base_table = self.cal_oscillation_base()
        self.chi_table = self.cal_oscillation(self.rpath_para_table)
        self.chi = np.array([])

        self.distance_table_temp = np.zeros(self.distance_table[0].size)
        self.index_temp = 0
        self.rpath_table_new = [None] * self.group_amount
        self.para_temp = []

    def reset_table(self, path, coordinate):
        self.file_index = used_file(path)
        self.distance_table = create_dist_table(coordinate)
        with open(path + r'\distance_table.csv', 'w') as f:
            f_write = writer(f, lineterminator='\n')
            f_write.writerows(self.distance_table)
        self.path_table = self.create_path_table(path, coordinate)
        with open(self.file + r'\path_table.csv', 'w') as f:
            f_write = writer(f, lineterminator='\n')
            f_write.writerows(self.path_table)
        self.rpath_table = self.cal_rpath()
        with open(self.file + r'\rpath_table.csv', 'w') as f:
            f_write = writer(f, lineterminator='\n')
            f_write.writerows(self.rpath_table)
        self.rpath_para_table = cal_rpath_para(self.rpath_table)
        with open(self.file + r'\rpath_para_table.csv', 'w') as f:
            f_write = writer(f, lineterminator='\n')
            f_write.writerows(self.rpath_para_table)
        self.group_amount = len(self.path_table)
        self.feff = []
        self.read_feff(path)
        self.chi_base_table = self.cal_oscillation_base()
        self.chi_table = self.cal_oscillation(self.rpath_para_table)

        self.distance_table_temp = np.zeros(self.distance_table[0].size)
        self.rpath_table_new = [None] * self.group_amount

    def create_path_table(self, file_name, coordinate):
        dict_r, dict_angle = read_path(file_name, self.file_index)
        table = []
        for index in range(len(dict_r)):
            if len(dict_r[index]) == 2:
                first = np.where(dict_r[index][0] == self.distance_table[0])[0]
                path_group = np.stack((first, np.zeros(first.size, dtype=int))).T
            elif len(dict_r[index]) == 3:
                first = np.array([], dtype=int)
                second = np.array([], dtype=int)
                satisfy_1 = np.where(dict_r[index][0] == self.distance_table[0])[0]
                for i in satisfy_1:
                    satisfy_2 = np.where(dict_r[index][1] == self.distance_table[i])[0]  # second step index
                    for j in satisfy_2:
                        if dict_r[index][2] == self.distance_table[j][0]:     # distance filter
                            second = np.append(second, j)
                            first = np.append(first, i)
                # final step is all back to absorbed atom
                path_group = np.stack((first, second, np.zeros(second.size, dtype=int))).T
                # if points of step1 and step2 are not symmetric, path (0,1,2,0) and (0,2,1,0) count separately
                if not dict_angle[index][0] == dict_angle[index][1]:
                    path_group = np.concatenate((path_group, path_group.T[[1, 0, 2]].T))
            elif len(dict_r[index]) == 4:
                first = np.array([], dtype=int)
                second = np.array([], dtype=int)
                third = np.array([], dtype=int)
                satisfy_1 = np.where(dict_r[index][0] == self.distance_table[0])[0]
                for i in satisfy_1:
                    satisfy_2 = np.where(dict_r[index][1] == self.distance_table[i])[0]  # second step index
                    for j in satisfy_2:
                        temp_1 = cal_angle(coordinate[0], coordinate[i], coordinate[j])
                        if dict_angle[index][0] == temp_1:      # angle filter
                            satisfy_3 = np.where(dict_r[index][2] == self.distance_table[j])[0]  # third step index
                            for k in satisfy_3:
                                temp_2 = cal_angle(coordinate[i], coordinate[j], coordinate[k])
                                if dict_angle[index][1] == temp_2:      # angle filter
                                    third = np.append(third, k)
                                    second = np.append(second, j)
                                    first = np.append(first, i)
                path_group = np.stack((first, second, third, np.zeros(third.size, dtype=int))).T
            table.append(path_group)
            # path_table[index] use feff000(index).dat [fixed]
        return table

    def cal_rpath(self):
        table = []
        for group in self.path_table:  # group: use specific feff data
            rpath_group = np.zeros(int(group.size / group[0].size))
            for i in range(rpath_group.size):  # rpath_group: r of paths using same feff data
                start = 0
                for step in group[i]:  # group[i]: index sequence of one path
                    rpath_group[i] += self.distance_table[start][step]  # dist_table[step]: distance of a step
                    start = step
                rpath_group[i] = round(rpath_group[i] / 2, 4)  # half of total path length.
            table.append(rpath_group)  # rpath_table share same index with path_table
        return table

    def read_feff(self, folder):
        for group in self.file_index:
            if group < 10:
                self.feff.append(FEFF(folder + r'\feff000%d.dat' % group, self.k0))
            elif group < 100:
                self.feff.append(FEFF(folder + r'\feff00%d.dat' % group, self.k0))
            else:
                self.feff.append(FEFF(folder + r'\feff0%d.dat' % group, self.k0))

    def cal_oscillation_base(self):
        table = []
        for group in range(self.group_amount):
            table.append(np.multiply(np.multiply(self.feff[group].amp, self.dw), self.k0**2) * self.s02)
        return table

    def cal_oscillation(self, para):
        table = []
        for index in range(self.group_amount):
            group = []
            for distance in para[index][0]:
                chi0 = np.multiply(np.multiply(np.sin(2 * distance * self.k0 + self.feff[index].phase),
                                   np.exp(-2 * distance * self.feff[index].lamb)), self.chi_base_table[index]
                                   ) / (distance ** 2)
                group.append(deltaE_shift(self.k0, chi0, self.energy))
            table.append(group)
        return table

    def sum_up_chi(self, k_start, k_end, r_start, r_end, para):
        chi = np.zeros(self.k0.size)
        for group in range(self.group_amount):
            for index in range(len(self.chi_table[group])):
                chi += para[group][1][index] * self.chi_table[group][index]
        chi_ift = back_k_space(chi, self.k0.size, r_start, r_end)
        self.k, chi = k_range(self.k0, chi_ift, k_start, k_end, False)
        return chi

    def modify_dist_table(self, new_array, atom_i):
        for column in range(self.distance_table[0].size):
            self.distance_table_temp[column] = self.distance_table[atom_i][column]
        self.index_temp = atom_i
        for column in range(self.distance_table[0].size):
            if not column == atom_i:
                self.distance_table[atom_i][column] = round(
                    sqrt(np.sum((new_array[column] - new_array[atom_i]) ** 2)), 4)
                self.distance_table[column][atom_i] = self.distance_table[atom_i][column]

    def recover_dist_table(self):
        for column in range(self.distance_table[0].size):
            if not column == self.index_temp:
                self.distance_table[self.index_temp][column] = self.distance_table_temp[column]
                self.distance_table[column][self.index_temp] = self.distance_table[self.index_temp][column]

    def modify_rpath(self, atom_i):
        table = []
        for group in range(self.group_amount):
            rpath_group = np.zeros(self.rpath_table[group].size)
            for path in range(rpath_group.size):
                if not np.where(self.path_table[group][path] == atom_i)[0].size == 0:
                    start = 0
                    for step in self.path_table[group][path]:
                        rpath_group[path] += self.distance_table[start][step]
                        start = step
                    rpath_group[path] = round(rpath_group[path] / 2, 4)
                else:
                    rpath_group[path] = self.rpath_table[group][path]
            table.append(rpath_group)
        return table

    def accept(self):
        for group in range(self.group_amount):
            self.rpath_table[group] = self.rpath_table_new[group].copy()
            for _ in range(2):
                self.rpath_para_table[group][_] = self.para_temp[group][_].copy()

    def modify_oscillation(self, para_source, para_target):
        for group in range(self.group_amount):
            i, j, k = 0, 0, 0
            while True:
                distance_i = para_source[group][0][i]
                distance_j = para_target[group][0][j]
                if distance_i > distance_j:
                    chi0 = np.multiply(np.multiply(np.sin(2 * distance_j * self.k0 + self.feff[group].phase),
                                       np.exp(-2 * distance_j * self.feff[group].lamb)), self.chi_base_table[group]
                                       ) / (distance_j ** 2)
                    self.chi_table[group].insert(k, deltaE_shift(self.k0, chi0, self.energy))
                    j += 1
                    k += 1
                elif distance_i < distance_j:
                    self.chi_table[group].pop(k)
                    i += 1
                else:
                    i += 1
                    j += 1
                    k += 1
                if i == para_source[group][0].size:
                    for r in para_target[group][0][j:]:
                        chi0 = np.multiply(np.multiply(np.sin(2 * r * self.k0 + self.feff[group].phase),
                                                       np.exp(-2 * r * self.feff[group].lamb)),
                                           self.chi_base_table[group]
                                           ) / (r ** 2)
                        self.chi_table[group].insert(k, deltaE_shift(self.k0, chi0, self.energy))
                    break
                elif j == para_target[group][0].size:
                    for _ in range(para_source[group][0].size - i):
                        self.chi_table[group].pop()
                    break

    def walk(self, coordinate, index):
        self.modify_dist_table(coordinate, index)
        self.rpath_table_new = self.modify_rpath(index)
        self.para_temp = cal_rpath_para(self.rpath_table_new)
        self.modify_oscillation(self.rpath_para_table, self.para_temp)


class FEFF:
    def __init__(self, file_name='feff0001.dat', k=np.array([]), reduce=1.0, weight=3):
        temp = file_name.split('\\')[-1].split('.')[0]
        self.index = int(temp[-4]) if len(temp) == 8 else int(temp.split('feff')[1])
        #self.phase_c = np.array([])
        #self.phase_s = np.array([])
        self.phase = np.array([])
        self.amp = np.array([])
        self.lamb = np.array([])
        self.atom = np.array([])
        self.chi = np.array([])
        self.weight = weight
        self.distance = 0
        #self.realp = np.array([])
        self.k = k

        self.read(file_name, reduce)

    def read(self, file_name, reduce):
        with open(file_name, 'r') as target:
            k0 = np.array([])
            amp = np.array([])
            #phase_c = np.array([])
            #phase_s = np.array([])
            phase = np.array([])
            lamb = np.array([])
            #realp = np.array([])
            while True:
                lines = target.readline()
                if not lines or not lines.find('------') == -1:
                    break
            info = target.readline().split()
            leg = int(info[0]) - 1
            self.distance = float(info[2])
            target.readline()
            target.readline()
            for i in range(leg):
                temp = target.readline().split()
                self.atom = np.append(self.atom, np.array([float(temp[0]), float(temp[1]), float(temp[2])]))
            self.atom = self.atom.reshape(leg, 3)
            target.readline()
            while True:
                data = target.readline()
                if not data:
                    break
                temp = [i for i in data.split()]
                k0 = np.append(k0, float(temp[0]))
                amp = np.append(amp, float(temp[2]) * float(temp[4]))
                phase = np.append(phase, float(temp[1]) + float(temp[3]))
                lamb = np.append(lamb, float(temp[5]))
                #realp = np.append(realp, float(temp[6]))

            if self.k.size == 0:
                self.k = k0.copy()
            #self.phase_c = intp1D(k0, self.k, phase_c)
            #self.phase_s = intp1D(k0, self.k, phase_s)
            self.phase = intp1D(k0, self.k, phase)
            self.amp = intp1D(k0, self.k, amp) * (self.k ** (self.weight - 1)) * reduce
            self.lamb = np.reciprocal(intp1D(k0, self.k, lamb))
            self.chi = np.multiply(np.multiply(np.sin(2 * self.distance * self.k + self.phase),
                                               np.exp(-2 * self.distance * self.lamb)), self.amp) / (self.distance ** 2)
            #self.chi = np.multiply(np.multiply(np.sin(2 * self.distance * self.k + intp1D(k0, self.k, phase)),
            #                                   np.exp(-2 * self.distance * np.reciprocal(intp1D(k0, self.k, lamb)))),
            #                       np.multiply(intp1D(k0, self.k, amp), self.k ** 2) * reduce) / (self.distance ** 2)
            #self.realp = intp1D(k0, self.k, realp)
