from mrmc_package import k_range, deltaE_shift, back_k_space, cal_angle, FEFF
import numpy as np
from math import sqrt, pi
from scipy.special import comb


class TABLE_POL:
    def __init__(self, k_head, k_tail, r_head, r_tail, sig2, dE, s02, k0, coor, element,
                 folder, polarization=-1, ms_en=False, weight=3):
        self.k0 = k0
        self.k_head = k_head
        self.k_tail = k_tail
        self.r_head = r_head
        self.r_tail = r_tail
        self.element = element
        self.coordinate = coor
        self.distance = np.array([sqrt((_ ** 2).sum()) for _ in coor])
        self.sig2 = np.exp(2 * sig2 * k0 ** 2)
        self.dE = dE
        self.s02 = s02
        self.folder = folder
        self.pol = polarization
        self.ms_en = ms_en
        self.weight = weight

        self.atom = np.array([])
        self.amount = self.distance.size

        l_range, l_step = self.read_ini(self.folder)
        self.decimals = 1 if (l_step * 100) % 10 == 0 else 2

        self.length = np.round(np.arange(float(l_range[0]), float(l_range[1]) + l_step, l_step), self.decimals)
        self.angle = np.linspace(0, pi, 91)
        self.angle_degree = self.angle / pi * 180
        self.path_size = self.length.size ** 2 * self.angle.size

        self.chi_1st = np.zeros((self.amount, k0.size))
        self.log_1st = np.zeros(self.amount)
        self.chi_1st_temp = np.zeros((self.amount, k0.size))
        self.log_1st_temp = np.zeros(self.amount)
        if self.ms_en:
            self.chi_commute = np.zeros((self.amount, k0.size))
            self.log_commute = np.zeros(self.amount)
            self.chi_commute_temp = np.zeros((self.amount, k0.size))
            self.log_commute_temp = np.zeros(self.amount)
            self.chi_2nd = np.zeros((self.amount, self.amount, k0.size))
            self.log_2nd = np.zeros((self.amount, self.amount))
            self.chi_2nd_temp = np.zeros((self.amount, self.amount, k0.size))
            self.log_2nd_temp = np.zeros(self.amount)
            self.chi_3rd = np.zeros((self.amount, self.amount, k0.size))
            self.log_3rd = np.zeros((self.amount, self.amount))
            self.chi_3rd_temp = np.zeros((self.amount, self.amount, k0.size))
            self.log_3rd_temp = np.zeros((self.amount, self.amount))

        self.single = np.empty((self.atom.size, self.length.size), dtype=FEFF)
        if self.ms_en:
            self.commute = np.empty((self.atom.size, self.length.size), dtype=FEFF)
            self.double = np.empty((comb(self.atom.size, 2), self.path_size), dtype=FEFF)
            self.triple = np.empty((comb(self.atom.size, 2), self.path_size), dtype=FEFF)

        # start=timer()
        self.create_single()
        if ms_en:
            self.create_commute()
            self.create_multi()
        self.k, self.chi = self.sum_up_chi(True)
        # print('create', timer()-start)

    def read_ini(self, file):
        with open(file + r'\table.ini', 'r') as f:
            self.atom = np.array([f.readline().split()[-1]])
            for _ in range(3):
                temp = f.readline().split()[-1]
                if not temp == 'Null':
                    self.atom = np.append(self.atom, temp)
            lrange = f.readline().split()[-2:]
            step = float(f.readline().split()[-1])
        return lrange, step

    def single_scattering(self, target):
        symbol = np.where(self.atom == self.element[target])[0][0]
        file = self.folder + r'\table2_%d' % symbol
        table = self.single[symbol - 1]
        de = self.dE[symbol - 1]
        if self.distance[target] < self.length[-1]:
            try:
                index = np.where(self.length == round(self.distance[target], self.decimals))[0][0]
            except IndexError:
                print(self.distance)
                print('Atoms overlapped', self.distance[target])
                self.log_1st[target] = 0
                return
            if not type(table[index]) == FEFF:
                try:
                    table[index] = FEFF(file + r'\feff%d.dat' % index, self.k0, self.s02, self.weight)
                except FileNotFoundError:
                    print(self.coordinate, self.distance, target, self.distance[target])
                    self.log_1st[target] = 0
                    return
            self.log_1st[target] = 1
            if 0 <= self.pol < 3:
                polar = 3 * ((self.coordinate[target][self.pol] -
                              self.coordinate[0][self.pol]) / self.distance[target]) ** 2
            elif self.pol == 3:
                polar = 3 * (((self.coordinate[target][0] - self.coordinate[0][0]) ** 2 +
                              (self.coordinate[target][1] - self.coordinate[0][1]) ** 2) /
                             self.distance[target] ** 2)
            else:
                polar = 1
            chi0 = np.multiply(np.multiply(np.multiply(
                np.sin(2 * self.distance[target] * self.k0 + table[index].phase),
                np.exp(-2 * self.distance[target] * table[index].lamb)), self.sig2),
                table[index].amp) / (self.distance[target] ** 2) * polar
            self.chi_1st[target] = deltaE_shift(self.k0, chi0, de)
        else:
            self.log_1st[target] = 0

    def create_single(self):
        for i in range(1, self.amount):
            self.single_scattering(i)

    def commute_scattering(self, target):
        symbol = np.where(self.atom == self.element[target])[0][0]
        file = self.folder + r'\table2s_%d' % symbol
        table = self.single[symbol - 1]
        de = self.dE[symbol - 1]
        if self.distance[target] < self.length[-1] / 2:
            index = np.where(self.length == round(self.distance[target], self.decimals))[0][0]
            if not type(table[index]) == FEFF:
                try:
                    table[index] = FEFF(file + r'\feff%d.dat' % index, self.k0, self.s02, self.weight)
                except FileNotFoundError:
                    self.log_commute[target] = 0
                    return
            self.log_commute[target] = 1
            rpath = self.distance[target] * 2
            polar = 3 * ((self.coordinate[target][self.pol] -
                          self.coordinate[0][self.pol]) / self.distance[target]) ** 2 if self.pol >= 0 else 1
            chi0 = np.multiply(np.multiply(np.multiply(
                np.sin(2 * rpath * self.k0 + table[index].phase),
                np.exp(-2 * rpath * table[index].lamb)), self.sig2),
                table[index].amp) / (rpath ** 2) * polar
            self.chi_commute[target] = deltaE_shift(self.k0, chi0, de)
        else:
            self.log_commute[target] = 0

    def create_commute(self):
        for i in range(1, self.amount):
            self.commute_scattering(i)

    def read_feff(self, index, array, folder):
        if not type(array[index]) == FEFF:
            try:
                array[index] = FEFF(folder + r'\feff%d.dat' % index, self.k0, self.s02)
            except FileNotFoundError:
                return False
        return True

    def multi_scattering(self, step1, step2, debug=False):
        symbol = np.array([np.where(self.atom == self.element[step1])[0][0],
                           np.where(self.atom == self.element[step2])[0][0]])
        symbol = np.flipud(symbol) if symbol[1] < symbol[0] else symbol
        file2nd = self.folder + r'\table3_%d_%d' % (symbol[0], symbol[1])
        table2nd = self.double[int(symbol.sum())]
        file3rd = self.folder + r'\table4_%d_%d' % (symbol[0], symbol[1])
        table3rd = self.triple[int(symbol.sum()) - 2]
        de = self.dE[np.where(self.atom == self.element[step1])[0][0] - 1]
        d_vector = sqrt(((self.coordinate[step2] - self.coordinate[step1]) ** 2).sum())
        if round(self.distance[step1], self.decimals) > self.length[-1] or \
                round(d_vector, self.decimals) > self.length[-1]:
            self.log_2nd[step1][step2] = 0
            self.log_3rd[step1][step2] = 0
            return
        angle_scat = cal_angle(np.zeros(3), self.coordinate[step1], self.coordinate[step2])
        angle_inter = cal_angle(self.coordinate[step1], np.zeros(3), self.coordinate[step2])
        if angle_inter < 90 and angle_scat >= 160:
            rpath_tri = self.distance[step1] + self.distance[step2]
        elif angle_inter > 90:
            if angle_scat >= 160:
                rpath_tri = self.distance[step2] + d_vector
            elif angle_scat <= 20:
                rpath_tri = self.distance[step1] + d_vector
            else:
                self.log_2nd[step1][step2] = 0
                self.log_3rd[step1][step2] = 0
                return
        else:
            return
        rpath_dou = (self.distance[step1] + d_vector + self.distance[step2]) / 2
        polar_1 = 3 * ((self.coordinate[step1][self.pol] -
                        self.coordinate[0][self.pol]) / self.distance[step1]) ** 2 if self.pol >= 0 else 1
        polar_2 = 3 * ((self.coordinate[step2][self.pol] -
                        self.coordinate[0][self.pol]) / self.distance[step2]) ** 2 if self.pol >= 0 else 1
        index_i = np.where(self.length == round(self.distance[step1], self.decimals))[0][0]
        index_j = np.where(self.angle_degree >= round(angle_scat, 0))[0][0]
        index_k = np.where(self.length == round(d_vector, self.decimals))[0][0]
        index = (index_i * self.angle.size + index_j) * self.length.size + index_k
        if self.read_feff(index, table2nd, file2nd):
            self.log_2nd[step1][step2] = 1
            chi0 = np.multiply(np.multiply(np.multiply(
                np.sin(2 * rpath_dou * self.k0 + table2nd[index].phase),
                np.exp(-2 * rpath_dou * table2nd[index].lamb)), self.sig2),
                table2nd[index].amp) / (rpath_dou ** 2) * (polar_1 + polar_2)
            self.chi_2nd[step1][step2] = deltaE_shift(self.k0, chi0, de)
        else:
            self.log_2nd[step1][step2] = 0
        if self.read_feff(index, table3rd, file3rd):
            if angle_inter > 90:
                if self.distance[step1] > self.distance[step2]:
                    polar_1 = 0
                else:
                    polar_2 = 0
            self.log_3rd[step1][step2] = 1
            chi0 = np.multiply(np.multiply(np.multiply(
                np.sin(2 * rpath_tri * self.k0 + table3rd[index].phase),
                np.exp(-2 * rpath_tri * table3rd[index].lamb)), self.sig2),
                table3rd[index].amp) / (rpath_tri ** 2) * (polar_1 + polar_2)
            self.chi_3rd[step1][step2] = deltaE_shift(self.k0, chi0, de)
        else:
            self.log_3rd[step1][step2] = 0

    def create_multi(self):
        for i in range(1, self.amount):
            for j in range(1, self.amount):
                if i < j:
                    self.multi_scattering(i, j)

    def modify_chi(self, atom):
        self.single_scattering(atom)
        if self.ms_en:
            self.commute_scattering(atom)
            for i in range(1, self.amount):
                if atom < i:
                    self.multi_scattering(atom, i)
                elif atom > i:
                    self.multi_scattering(i, atom)

    def sum_up_chi(self, get_k=False, debug=False):
        chi0 = np.zeros(self.k0.size)
        if debug:
            print(self.chi_1st.shape)
        for i in np.where(self.log_1st == 1)[0]:
            chi0 += self.chi_1st[i]
        if self.ms_en:
            for i in np.where(self.log_commute == 1)[0]:
                chi0 += self.chi_commute[i]
            used = np.where(self.log_2nd == 1)
            for i in range(used[0].size):
                chi0 += self.chi_2nd[used[0][i]][used[1][i]]
            used = np.where(self.log_3rd == 1)
            for i in range(used[0].size):
                chi0 += self.chi_3rd[used[0][i]][used[1][i]]
        chi_ift = back_k_space(chi0, self.k0.size, self.r_head, self.r_tail)
        return k_range(self.k0, chi_ift, self.k_head, self.k_tail, False, get_k=get_k)

    def moving(self, target, coor, debug=False):
        # start = timer()
        self.coordinate[target] = coor
        self.distance[target] = sqrt((coor ** 2).sum())

        if debug:
            print('moving_s(chi_temp, chi_1st):', self.chi_1st.shape, self.chi_1st_temp.size)
        self.chi_1st_temp = self.chi_1st[target].copy()
        self.log_1st_temp[0] = self.log_1st[target]
        if self.ms_en:
            self.chi_commute_temp = self.chi_commute[target].copy()
            self.log_commute_temp[0] = self.log_commute[target]
            self.chi_2nd[target, 1:target] = self.chi_2nd[1:target, target]
            self.chi_3rd[target, 1:target] = self.chi_3rd[1:target, target]
            self.chi_2nd[target:, target] = self.chi_2nd[target, target:]
            self.chi_3rd[target:, target] = self.chi_3rd[target, target:]
            self.log_2nd_temp = self.log_2nd.copy()
            self.log_3rd_temp = self.log_3rd.copy()

        self.modify_chi(target)
        self.chi = self.sum_up_chi()
        # print('modify', timer() - start)

    def recover(self, target, coor, debug=False):
        self.coordinate[target] = coor
        self.distance[target] = sqrt((coor ** 2).sum())

        if debug:
            print('recover_s(chi_temp, chi_1st):', self.chi_1st.shape, self.chi_1st_temp.size)
        self.chi_1st[target] = self.chi_1st_temp.copy()
        self.log_1st[target] = self.log_1st_temp[0]
        if self.ms_en:
            self.chi_commute[target] = self.chi_commute_temp.copy()
            self.log_commute[target] = self.log_commute_temp[0]
            self.chi_2nd[1:target, target] = self.chi_2nd[target, 1:target]
            self.chi_3rd[1:target, target] = self.chi_3rd[target, 1:target]
            self.chi_2nd[target, target:] = self.chi_2nd[target:, target]
            self.chi_3rd[target, target:] = self.chi_3rd[target:, target]
            self.log_2nd = self.log_2nd_temp.copy()
            self.log_3rd = self.log_3rd_temp.copy()

        self.chi = self.sum_up_chi()

    def moving_group(self, coor, dist, elem, debug=False):
        self.coordinate = coor
        self.distance = dist
        if debug:
            # print('moving1', dist.size, elem.size, self.amount, self.log_1st.size, self.log_1st_temp.size)
            print('moving_g(ele_i, ele_o, dist, coor, chi):', elem, self.element, self.amount, coor.shape[0],
                  self.chi_1st.shape)
        self.chi_1st_temp = self.chi_1st.copy()
        self.log_1st_temp = self.log_1st.copy()
        if self.ms_en:
            self.chi_commute_temp = self.chi_commute.copy()
            self.log_commute_temp = self.log_commute.copy()
            self.chi_2nd_temp = self.chi_2nd.copy()
            self.log_2nd_temp = self.log_2nd.copy()
            self.chi_3rd_temp = self.chi_3rd.copy()
            self.log_3rd_temp = self.log_3rd.copy()
        if not np.unique(elem, return_counts=True)[0].size == np.unique(self.element, return_counts=True)[0].size:
            change = True
        else:
            change = np.unique(elem, return_counts=True)[1] == np.unique(self.element, return_counts=True)[1]
            change = not (change if type(change) == bool else change.all())
        if change:
            self.element = elem
            self.amount = dist.size
            self.chi_1st = np.zeros((self.amount, self.k0.size))
            self.log_1st = np.zeros(self.amount)
            if self.ms_en:
                self.chi_2nd = np.zeros((self.amount, self.amount, self.k0.size))
                self.chi_3rd = np.zeros((self.amount, self.amount, self.k0.size))
                self.log_2nd = np.zeros((self.amount, self.amount))
                self.log_3rd = np.zeros((self.amount, self.amount))
        if debug:
            # print('moving2', dist.size, elem.size, self.amount, self.log_1st.size, self.log_1st_temp.size)
            print('chi_1st:', self.chi_1st.shape)
        self.create_single()
        if self.ms_en:
            self.create_commute()
            self.create_multi()
        self.chi = self.sum_up_chi(False)

    def recover_group(self, coor, dist, elem, debug=False):
        self.coordinate = coor
        self.distance = dist
        if debug:
            # print('recover1', dist.size, elem.size, self.amount, self.log_1st.size, self.log_1st_temp.size)
            print('recover_g(ele, dist, coor, chi):', elem.size, self.amount, coor.shape[0], self.chi_1st.shape)
        if not np.unique(elem, return_counts=True)[0].size == np.unique(self.element, return_counts=True)[0].size:
            change = True
        else:
            change = np.unique(elem, return_counts=True)[1] == np.unique(self.element, return_counts=True)[1]
            change = not (change if type(change) == bool else change.all())
        if change:
            self.element = elem
            self.amount = dist.size
        self.chi_1st = self.chi_1st_temp.copy()
        self.log_1st = self.log_1st_temp.copy()
        if self.ms_en:
            self.chi_commute = self.chi_commute_temp.copy()
            self.log_commute = self.log_commute_temp.copy()
            self.chi_2nd = self.chi_2nd_temp.copy()
            self.log_2nd = self.log_2nd_temp.copy()
            self.chi_3rd = self.chi_3rd_temp.copy()
            self.log_3rd = self.log_3rd_temp.copy()
        if debug:
            # print('recover2', dist.size, elem.size, self.amount, self.log_1st.size)
            print('chi_1st:', self.chi_1st.shape)
        self.chi = self.sum_up_chi(False, debug)
