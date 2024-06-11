from math import sqrt
from random import randrange
import os

from math import exp, pi, acos, atan, sin, cos

import numpy as np
from matplotlib import pyplot as plt

from mrmc_package import k_range, deltaE_shift, back_k_space, norm_fft



def get_distance(coor):
    return np.array([sqrt((_ ** 2).sum()) for _ in coor])

def sort(xyz, compound):
    distance = get_distance(xyz)
    index = distance.argsort()
    distance.sort()
    coordinate = xyz.take([index], 0)[0]
    element = compound.take([index], 0)[0]
    return coordinate, distance, element

def metropolis(r_0, r, tau=10e-5):
    if r_0 > r:
        return True
    else:
        if tau == 0:
            return False
        met = exp(-(r - r_0) / tau)
        judge = randrange(0, 100)
        if (judge / 100) < met:
            return True
        else:
            return False


class EXP:
    def __init__(self, file_name, k_start, k_end, r_start, r_end, weight, larch):
        self.name = file_name
        self.k_start = k_start
        self.k_end = k_end
        self.r_start = r_start
        self.r_end = r_end
        self.k0 = np.array([])
        self.k = np.array([])
        self.chi = np.array([])
        self.ft = np.array([])
        self.cross = np.array([])
        self.r = np.arange(0, 6 + pi / 102.4, pi / 102.4)
        self.r_range = np.where((r_start < self.r) & (self.r < r_end))[0]
        self.r_cut = np.array([])
        self.ft_cut = np.array([])

        self.init(weight, larch)
        self.max = [np.argmax(np.abs(self.chi)), np.max(np.abs(self.chi))]

    def init(self, kw, larch=True):
        #dtype = self.name.split('.')[1]
        if larch:
            self.read_bk(self.name, kw)
        else:
            self.read_exp(self.name)
        self.chi_bottom = np.sum(self.chi ** 2)
        self.ft_bottom = np.sum(self.ft_cut ** 2)
        self.cross_bottom = np.sum(self.cross ** 2)

    def read_exp(self, filename='703K_diff_H2-dry.rex'):
        # reading oscillation data
        if filename.split('.')[1] == 'rex':
            with open(filename, 'r') as f_exp:
                k = np.array([])
                xi = np.array([])
                while True:
                    lines = f_exp.readline(11)
                    if not lines or not lines.find('[ED_BEGIN]') == -1:
                        break
                while True:
                    data = f_exp.readline()
                    if data.isspace():
                        break
                    temp = [i for i in data.split()]
                    k = np.append(k, float(temp[0]))
                    xi = np.append(xi, float(temp[1]))
                self.k0 = k
                self.process(xi)
        else:
            with open(filename, 'r') as f:
                while True:
                    line = f.readline()
                    if not line.find('#  k') == -1:
                        break
                k = np.array([])
                xi = np.array([])
                while True:
                    line = f.readline()
                    if not line or line.isspace():
                        break
                    temp = line.split()
                    k = np.append(k, float(temp[0]))
                    xi = np.append(xi, float(temp[2]))
            self.k0 = k
            self.process(xi)

    def read_bk(self, filename='Tcu.ex3', k_weight=3):
        # reading oscillation data
        dtype = self.name.split('.')[1]
        from larch.xafs import autobk, xftf, xftr
        from larch.io import read_ascii
        if dtype == 'ex3' or dtype == 'rex':
            f_exp = open(filename, 'r')
            f_bk = open('temp_bk.dat', 'w')
            f_bk.write('#   energy        xmu\n')
            while True:
                lines = f_exp.readline(11)
                if not lines or not lines.find('[EX_BEGIN]') == -1:
                    break
            while True:
                data = f_exp.readline()
                if data.isspace() or not data.find('[EX_END]') == -1:
                    break
                temp = data.split()
                f_bk.write('%.9s %.9s\n' % (temp[0], temp[1]))
            f_exp.close()
            f_bk.close()
        x_data = read_ascii('temp_bk.dat' if (dtype == 'ex3' or dtype == 'rex') else filename, labels='energy xmu')
        autobk(x_data.energy, x_data.xmu, group=x_data)
        xftf(x_data.k, x_data.chi, group=x_data, kmin=3, kmax=9, kweight=3)
        xftr(x_data.r, x_data.chir, group=x_data, rmin=1, rmax=2.7)
        #plt.plot(x_data.q, x_data.chiq/2, c='k')
        self.k0 = x_data.k
        self.process(x_data.chi * self.k0**k_weight)

    def process(self, source):
        self.k, chi = k_range(self.k0, source, self.k_start, self.k_end, False)
        self.chi, self.ft = back_k_space(chi, self.r, self.k.size, self.r_start, self.r_end)
        self.r_cut = self.r[self.r_range]
        self.ft_cut = self.ft[self.r_range]
        self.cross = self.chi * np.transpose([self.ft_cut])

    def r_factor_chi(self, target):
        return np.sum(np.subtract(self.chi, target)**2) / self.chi_bottom

    def r_factor_ft(self, target):
        return np.sum(np.subtract(self.ft_cut, target[self.r_range])**2) / self.ft_bottom

    def r_factor_cross(self, target):
        return np.sum(np.subtract(self.cross, target) ** 2) / self.cross_bottom

    def amp_ratio(self, target):
        amp = (self.max[1] / np.abs(target)[self.max[0]]) - 1
        return amp if amp > 0 else 0


class CHI:
    def __init__(self, file_name='cu/chi.dat', k_start=3.0, k_end=12.0, r_start=0.0, r_end=6.0,
                 dE=0.0, s02=1.0, k_size=0):
        self.name = file_name
        self.k_start = k_start
        self.k_end = k_end
        self.dE = dE
        self.s02 = s02
        self.r_start = r_start
        self.r_end = r_end
        self.size = k_size
        self.k0 = np.array([])
        self.k = np.array([])
        self.chi = np.array([])
        self.chi0 = np.array([])
        self.r = np.arange(0, 6 + pi / 102.4, pi / 102.4)

        self.read_file()
        self.ft = np.abs(norm_fft(self.chi, self.r.size))

    def read_file(self):
        with open(self.name, 'r') as target:
            if self.k0.size == 0:
                k = np.array([])
            chi = np.array([])
            if self.name.split('.')[1] == 'dat':
                while True:
                    lines = target.readline()
                    if not lines or not lines.find('phase @#') == -1:
                        break
            if self.k0.size == 0:
                while True:
                    data = target.readline()
                    if not data:
                        break
                    temp = [i for i in data.split()]
                    k = np.append(k, float(temp[0]))
                    chi = np.append(chi, float(temp[1]))
                self.size = k.size if self.size == 0 else self.size
                self.k0 = k.copy()[:self.size]
            else:
                while True:
                    data = target.readline()
                    if not data:
                        break
                    chi = np.append(chi, float(data.split()[1]))
            self.chi0 = np.multiply(chi.copy()[:self.size], self.k0 ** 3)
            self.process(self.s02 * self.chi0)

    def process(self, source):
        chi_shift = deltaE_shift(self.k0, source, dE=self.dE)
        chi_ift = back_k_space(chi_shift, self.size, self.r_start, self.r_end)
        self.k, self.chi = k_range(self.k0, chi_ift, self.k_start, self.k_end, False)

    def r_factor(self, target):
        return np.sum(np.power(np.subtract(self.chi, target), 2)) / np.sum(self.chi ** 2)


'''class TetraFactory(CenteredTetragonalFactory):
    bravais_basis = [[0, 0, 0], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]
    element_basis = (0, 1, 1, 1, 1)


class OctaFactory(SimpleCubicFactory):
    atomic_basis = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.], [0., 0., .5]])
    element_basis = [0, 1, 1, 1]


class CrossFactory(SimpleCubicFactory):
    atomic_basis = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.]])
    element_basis = [0, 1, 1]'''


class ATOMS:
    def __init__(
            self, database='', file=' ', pos=np.array([]), element=np.array([]), spherical=True, random=True,
            local_range=np.array([]), surface='', surface_path='', step=np.array([]), step_range=np.array([]),
            crate_flag=True, surface_range=np.array([]), trial=50):
        self.file = file
        self.surface = surface
        self.coordinate_whole = np.array([])
        self.element_whole = np.array([])
        self.cw_temp = np.array([])
        self.ew_temp = np.array([])
        self.distance = np.array([])
        self.coordinate = np.array([])
        self.element = np.array([])
        self.c_temp = np.array([])
        self.e_temp = np.array([])
        self.symbol = np.array([])
        self.step = step
        self.step_range = np.array([int(step_range[_]/step[_]) for _ in range(step.size)])
        self.limit = surface_range
        self.trial = trial
        self.surface_symbol = np.array([])
        self.surface_path = surface_path

        if not self.surface == '' and self.limit.size == 0:
            if self.surface == 'TiO2':
                self.limit = np.array([-4.7, 6.2, -13.3, 12, 1.5, 5])
            elif self.surface == 'Al2O3':
                self.limit = np.array([-7, 7, -7, 7, 0, 3])

        with open(database + r'\table.ini', 'r') as f:
            for i in range(4):
                temp = f.readline().split()[-1]
                if temp == 'Null':
                    continue
                self.symbol = np.append(self.symbol, temp)
            self.min_distance = float(f.readline().split()[1])

        self.local_size = element.size
        self.local_range = local_range
        self.center_e = element[0]
        self.satellite_e = element[1:]
        if spherical:
            self.center_c = np.array([pos[0][0] * sin((pos[0][2]) / 180 * pi) * cos((pos[0][1]) / 180 * pi),
                                      pos[0][0] * sin((pos[0][2]) / 180 * pi) * sin((pos[0][1]) / 180 * pi),
                                      pos[0][0] * cos((pos[0][2]) / 180 * pi)])
            self.satellite_c = np.array([])
            for i in range(1, element.size):
                if pos[i][1] > 360 or pos[i][1] < 0:
                    pos[i][1] = randrange(360)
                self.satellite_c = np.append(self.satellite_c,
                                    np.array([pos[i][0] * sin((pos[i][2]) / 180 * pi) * cos((pos[i][1]) / 180 * pi),
                                              pos[i][0] * sin((pos[i][2]) / 180 * pi) * sin((pos[i][1]) / 180 * pi),
                                              pos[i][0] * cos((pos[i][2]) / 180 * pi)]))
            self.satellite_c = np.reshape(self.satellite_c, (self.satellite_e.size, 3))
        else:
            self.center_c = pos[0]
            self.satellite_c = pos[1:]
        self.surface_c = np.array([])
        self.surface_e = np.array([])

        if crate_flag:
            if self.surface == 'TiO2':
                root = self.create_TiO2(random, self.surface_path)
                self.deposition(root)
                self.c_best = self.coordinate_whole.copy()
                self.e_best = self.element_whole.copy()
            elif self.surface == 'Al2O3':
                root = self.create_Al2O3(random, self.surface_path)
                self.deposition(root)
                self.c_best = self.coordinate_whole.copy()
                self.e_best = self.element_whole.copy()
            else:
                self.coordinate = np.vstack((self.center_c, self.satellite_c))
                self.element = np.append(self.center_e, self.satellite_e)
                self.distance = get_distance(self.coordinate)
                self.c_best = self.coordinate.copy()
                self.e_best = self.element.copy()
            self.cw_temp = self.coordinate_whole.copy()
            self.ew_temp = self.element_whole.copy()
            self.c_temp = self.coordinate.copy()
            self.e_temp = self.element.copy()

    def create_TiO2(self, ran, file):
        ele = np.array([])
        coor = np.array([])
        if len(file) == 0:
            file = os.getcwd() + r'\TiO2.xyz'
        with open(file, 'r') as f:
            f.readline()
            f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                temp = line.split()
                ele = np.append(ele, temp[0])
                coor = np.append(coor, np.array([float(temp[1]), float(temp[2]), float(temp[3])]))
        self.coordinate_whole = np.round(np.reshape(coor, (-1, 3)), 3)
        self.element_whole = ele.copy()
        self.surface_c = self.coordinate_whole.copy()
        self.surface_e = self.element_whole.copy()
        self.surface_symbol = np.array(['O', 'Ti'])
        adsorb = np.array([], dtype=int)
        for i in range(ele.size):
            if self.surface_c[i][0] == 0 and self.surface_c[i][1] == 0 and self.surface_c[i][2] == 0:
                center = i
            if ele[i] == 'O':
                if (self.limit[0] + 1.5) < self.surface_c[i][0] < (self.limit[1] - 1.5) \
                        and (self.limit[2] + 1.5) < self.surface_c[i][1] < (self.limit[3] - 1.5):
                    adsorb = np.append(adsorb, i)

        self.y1 = -1.93
        self.y2l = -6.45
        self.rx = 4.95
        self.rxl = 6.09
        self.elevation = 65 / 180 * pi
        self.eccenter = np.array([-0.52, self.y1])
        self.dangling = np.array([], dtype=int)
        for i in range(ele.size):
            if ele[i] == 'Ti' and (self.surface_c[i][1] == -13.011 or
                                   abs(self.surface_c[i][1]) == 6.505 or
                                   self.surface_c[i][1] == 0):
                self.dangling = np.append(self.dangling, self.surface_c[i])
        self.dangling = np.reshape(self.dangling, (int(self.dangling.size / 3), 3))


        top = np.unique(self.surface_c[adsorb][..., 2])[-2:]
        top_layer = np.array([], dtype=int)
        for i in adsorb:
            if self.surface_c[i][2] == top[0] or self.surface_c[i][2] == top[1]:
                top_layer = np.append(top_layer, i)
        return top_layer[randrange(top_layer.size)] if ran else center

    def create_Al2O3(self, ran, file):
        ele = np.array([])
        coor = np.array([])
        if len(file) == 0:
            file = os.getcwd() + r'\Al2O3.xyz'
        with open(file, 'r') as f:
            f.readline()
            f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                temp = line.split()
                ele = np.append(ele, temp[0])
                coor = np.append(coor, np.array([float(temp[1]), float(temp[2]), float(temp[3])]))
        self.coordinate_whole = np.round(np.reshape(coor, (-1, 3)), 3)
        self.element_whole = ele.copy()
        self.surface_c = self.coordinate_whole.copy()
        self.surface_e = self.element_whole.copy()
        self.surface_symbol = np.array(['O', 'Al'])
        adsorb = np.array([], dtype=int)
        for i in range(ele.size):
            if self.surface_c[i][0] == 0 and self.surface_c[i][1] == 0 and self.surface_c[i][2] == 0:
                center = i
            if ele[i] == 'O':
                if (self.limit[0] + 1.5) < self.surface_c[i][0] < (self.limit[1] - 1.5) \
                        and (self.limit[2] + 1.5) < self.surface_c[i][1] < (self.limit[3] - 1.5):
                    adsorb = np.append(adsorb, i)
        top = np.max(self.surface_c[adsorb][..., 2])
        top_layer = np.array([], dtype=int)
        for i in adsorb:
            if self.surface_c[i][2] == top:
                top_layer = np.append(top_layer, i)
        return top_layer[randrange(top_layer.size)] if ran else center

    def deposition(self, root):
        self.center_c += self.coordinate_whole[root]
        self.coordinate_whole = np.vstack((self.coordinate_whole, self.center_c))
        self.element_whole = np.append(self.element_whole, self.center_e)
        for i in range(self.satellite_e.size):
            self.satellite_c[i] += self.center_c
            self.coordinate_whole = np.vstack((self.coordinate_whole, self.satellite_c[i]))
            self.element_whole = np.append(self.element_whole, self.satellite_e[i])
        distance = get_distance(self.coordinate_whole[:-self.local_size] - self.coordinate_whole[-self.local_size])
        self.coordinate = self.coordinate_whole[-self.local_size:].copy()
        self.element = self.element_whole[-self.local_size:].copy()
        self.surface_symbol = np.unique(self.surface_e)
        for j in range(self.surface_symbol.size):
            select = np.where((self.surface_e == self.surface_symbol[j]) &
                              (distance < self.local_range[self.surface_symbol[j]]))[0]
            self.coordinate = np.vstack((self.coordinate, self.surface_c[select]))
            self.element = np.append(self.element, self.surface_e[select])
        self.coordinate -= self.coordinate[0]
        self.distance = get_distance(self.coordinate)

    def write(self, coor, ele):
        distance = np.array([sqrt((_ ** 2).sum()) for _ in coor])
        with open(self.file + r'\feff.inp', 'r+') as file_feff:
            file_feff.seek(0)
            while True:
                lines = file_feff.readline(6)
                if not lines or not lines.find('ATOMS') == -1:
                    break
            file_feff.seek(file_feff.tell())
            file_feff.write('\n')
            file_feff.seek(file_feff.tell())
            file_feff.write('   %.5f     %.5f     %.5f    %d  %s1              %.5f\n'
                            % (coor[0][0], coor[0][1], coor[0][2], 0, ele[0], distance[0]))
            for i in range(1, coor.shape[0]):
                file_feff.seek(file_feff.tell())
                file_feff.write('   %.5f     %.5f     %.5f    %d  %s1              %.5f\n'
                                % (coor[i][0], coor[i][1],
                                   coor[i][2], np.where(self.symbol == ele[i])[0][0]+1, ele[i], distance[i]))
            file_feff.seek(file_feff.tell())
            file_feff.write('END\n')
            file_feff.truncate()

    def read(self, index):
        os.popen('copy "%s" "%s"' % (self.file + r'\result\result%d.txt' % index,
                                     self.file + r'\backup\result\result%d.txt' % index))
        with open(self.file + r'\result\result%d.txt' % index, 'r') as f:
            coordinate_best = np.array([])
            element_best = np.array([])
            distance_best = np.array([])
            coordinate = np.array([])
            element = np.array([])
            distance = np.array([])
            while True:
                lines = f.readline()
                if not lines.find('Best') == -1:
                    break
            while True:
                data = f.readline()
                if data.isspace() or not data:
                    break
                temp = data.split()
                coordinate_best = np.append(coordinate_best, np.array([float(temp[0]), float(temp[1]), float(temp[2])]))
                element_best = np.append(element_best, temp[4][:-1])
                distance_best = np.append(distance_best, float(temp[5]))
            while True:
                lines = f.readline()
                if not lines.find('final') == -1:
                    break
            while True:
                data = f.readline()
                if data.isspace() or not data:
                    break
                temp = data.split()
                coordinate = np.append(coordinate, np.array([float(temp[0]), float(temp[1]), float(temp[2])]))
                element = np.append(element, temp[4][:-1])
                distance = np.append(distance, float(temp[5]))
        print('data read')
        if not self.surface == '':
            self.coordinate_whole = coordinate.reshape(element.size, 3)
            self.cw_temp = self.coordinate_whole.copy()
            self.element_whole = element.copy()
            self.surface_c = self.coordinate_whole[:-self.local_size]
            self.surface_e = self.element_whole[:-self.local_size]
            distance = get_distance(self.coordinate_whole[:-self.local_size] - self.coordinate_whole[-self.local_size])
            self.coordinate = self.coordinate_whole[-self.local_size:].copy()
            self.element = self.element_whole[-self.local_size:].copy()
            self.surface_symbol = np.unique(self.surface_e)
            for j in range(self.surface_symbol.size):
                select = np.where((self.surface_e == self.surface_symbol[j]) &
                                  (distance < self.local_range[self.surface_symbol[j]]))[0]
                self.coordinate = np.vstack((self.coordinate, self.surface_c[select]))
                self.element = np.append(self.element, self.surface_e[select])
            self.coordinate -= self.coordinate[0]
            self.distance = get_distance(self.coordinate)
            self.cw_temp = self.coordinate_whole.copy()
            self.ew_temp = self.element_whole.copy()
            self.center_c = self.coordinate_whole[-self.local_size].copy()
            self.center_e = self.element_whole[-self.local_size].copy()
            self.satellite_c = self.coordinate_whole[-self.local_size - 1:].copy()
            self.satellite_e = self.element_whole[-self.local_size - 1:].copy()
            if self.surface == 'TiO2':
                self.y1 = -1.93
                self.y2l = -6.45
                self.rx = 4.95
                self.rxl = 6.09
                self.elevation = 65 / 180 * pi
                self.eccenter = np.array([-0.52, self.y1])
                self.dangling = np.array([], dtype=int)
                for i in range(self.surface_e.size):
                    if self.surface_e[i] == 'Ti' and (self.surface_c[i][1] == -13.011 or
                                           abs(self.surface_c[i][1]) == 6.505 or
                                           self.surface_c[i][1] == 0):
                        self.dangling = np.append(self.dangling, self.surface_c[i])
                self.dangling = np.reshape(self.dangling, (int(self.dangling.size / 3), 3))
        else:
            self.coordinate = coordinate.reshape(distance.size, 3)
            self.distance = distance.copy()
            self.element = element.copy()
            self.center_c = self.coordinate[0].copy()
            self.center_e = self.element[0].copy()
            self.satellite_c = self.coordinate[1:].copy()
            self.satellite_e = self.element[1:].copy()
        self.c_best = coordinate_best.reshape(element_best.size, 3)
        self.e_best = element_best.copy()
        self.c_temp = self.coordinate.copy()
        self.e_temp = self.element.copy()
        print('parameter set up')

    def moving(self, target_i=None):
        trials = self.trial
        while trials > 0:
            self.c_temp = self.coordinate.copy()
            target = randrange(1, self.distance.size) if target_i is None else target_i
            self.c_temp[target][randrange(3)] += round(randrange(-self.step_range[1], self.step_range[1] + 1)
                                                       * self.step[1], 3)
            distance = np.delete(get_distance(self.c_temp - self.c_temp[target]), target)
            if np.min(distance) < self.min_distance:
                trials -= 1
                continue
            if not self.surface == '' and np.min(distance) > self.local_range[self.element[target]]:
                trials -= 1
                continue
            if self.c_temp.shape[0] > 3 and self.element[3] == 'O':
                if not (2.80 < sqrt(((self.c_temp[2] - self.c_temp[3]) ** 2).sum()) < 3):
                    trials -= 1
                    continue
            self.distance[target] = sqrt((self.c_temp[target] ** 2).sum())
            break
        flag = True if trials == 0 else False
        return target, flag

    def moving_spherical(self, target_i=None):
        trials = self.trial
        while trials > 0:
            self.c_temp = self.coordinate.copy()
            target = randrange(1, self.local_size) if target_i is None else target_i
            ri = sqrt(((self.c_temp[target]) ** 2).sum())
            if not self.c_temp[target][0] == 0:
                azimuth = atan(self.c_temp[target][1] / self.c_temp[target][0])
                if self.c_temp[target][0] < 0:
                    azimuth += pi
            else:
                azimuth = pi / 2
                if self.c_temp[target][1] < 0:
                    azimuth += pi
            elevation = acos(self.c_temp[target][2] / ri)
            axis = randrange(3)
            if axis == 0:
                ri += round(randrange(-self.step_range[1], self.step_range[1] + 1) * self.step[1], 3)
            elif axis == 1:
                azimuth += randrange(-50, 51) / 1800 * pi
            else:
                elevation += randrange(-50, 51) / 1800 * pi
                if elevation > pi:
                    elevation = 2 * pi - elevation
                    azimuth += pi
                elif elevation < 0:
                    elevation = -elevation
                    azimuth += pi
            if azimuth > 2 * pi:
                azimuth -= 2 * pi
            elif azimuth < 0:
                azimuth += 2 * pi
            self.c_temp[target][0] = round(ri * sin(elevation) * cos(azimuth), 3)
            self.c_temp[target][1] = round(ri * sin(elevation) * sin(azimuth), 3)
            self.c_temp[target][2] = round(ri * cos(elevation), 3)
            distance = np.delete(get_distance(self.c_temp - self.c_temp[target]), target)
            if np.min(distance) < self.min_distance:
                trials -= 1
                continue
            if not self.surface == '' and np.min(distance) > self.local_range[self.element[target]]:
                trials -= 1
                continue
            '''if self.c_temp.shape[0] > 3 and self.element[3] == 'O':
                if not (2.80 < sqrt(((self.c_temp[2] - self.c_temp[3]) ** 2).sum()) < 3):
                    trials -= 1
                    continue'''
            self.distance[target] = sqrt((self.c_temp[target] ** 2).sum())
            break
        flag = True if trials == 0 else False
        return target, flag

    def moving_center(self):
        trials = self.trial
        while trials > 0:
            self.cw_temp = self.coordinate_whole.copy()
            step = round(randrange(-self.step_range[0], self.step_range[0] + 1) * self.step[0], 3)
            axis = randrange(3)
            for i in range(self.local_size):
                self.cw_temp[self.surface_e.size+i][axis] += step
            if not (self.limit[0] < self.cw_temp[-self.local_size][0] < self.limit[1]) or \
                    not (self.limit[2] < self.cw_temp[-self.local_size][1] < self.limit[3]) or \
                    not (self.limit[4] < self.cw_temp[-self.local_size][2] < self.limit[5]):
                trials -= 1
                continue

            '''if self.surface == 'TiO2':
                if not self.tca_filter(self.cw_temp[-1]):
                    trials -= 1
                    continue'''

            distance = get_distance(self.cw_temp[:-self.local_size] - self.cw_temp[-self.local_size])
            self.c_temp = self.cw_temp[-self.local_size:].copy()
            self.e_temp = self.ew_temp[-self.local_size:].copy()
            for j in range(self.surface_symbol.size):
                select = np.where((self.surface_e == self.surface_symbol[j]) &
                                (distance < self.local_range[self.surface_symbol[j]]))[0]
                self.c_temp = np.vstack((self.c_temp, self.surface_c[select]))
                self.e_temp = np.append(self.e_temp, self.surface_e[select])
            self.c_temp -= self.c_temp[0]
            distance = get_distance(self.c_temp)[self.local_size:]
            if distance.size == 0 or np.min(distance) < self.min_distance:
                trials -= 1
                continue
            self.distance = get_distance(self.c_temp)
            break
        return True if trials == 0 else False

    def tca_filter(self, coor_s):
        coor_rot = np.zeros(2)
        temp = self.dangling - coor_s
        qualify = False
        for j in range(self.dangling.shape[0]):
            azi = atan(temp[j][1] / temp[j][0]) if not temp[j][0] == 0 else 0
            if temp[j][0] < 0:
                azi += pi
            coor_rot[0] = temp[j][0] * cos(azi) + temp[j][1] * sin(azi)
            coor_rot[1] = temp[j][2]
            if self.y2l < coor_rot[1] < self.y1:
                vect = coor_rot - self.eccenter
                if self.rx < sqrt((vect ** 2).sum()) < self.rxl:
                    if 0 < -atan(vect[1] / vect[0]) < self.elevation:
                        qualify = True
                        break
        return qualify

    def add_atom(self, rx, ry, rz):
        #center = np.array([self.coordinate[:, _].sum() for _ in range(3)]) / self.distance.size
        i_vector = np.array([rx, ry, rz])
        surface = np.array([], dtype=int)
        for i in range(self.coordinate.shape[0]):
            if np.where(np.array(([sqrt(((self.coordinate[i] - self.coordinate[_]) ** 2).sum()
                                        ) for _ in range(self.coordinate.shape[0])])) < 3)[0].size < 12:
                surface = np.append(surface, i)
        off_center = self.coordinate[surface] #- center[0]
        d_off_center = np.array([sqrt((off_center[_] ** 2).sum()) for _ in range(off_center.shape[0])])
        vect_angle = np.array([acos(((np.abs(off_center[_]) * i_vector).sum()) / sqrt((i_vector ** 2).sum())
                                    / d_off_center[_]) for _ in range(d_off_center.size)])
        d_off_center = d_off_center ** 2
        print('distance_index:', np.argsort(d_off_center))
        print('distance:', np.sort(d_off_center))
        print('angle_index:', np.argsort(vect_angle))
        print('angle:', np.sort(vect_angle))
        vect_final = vect_angle * d_off_center
        ad_index = np.argmin(vect_final)#randrange(surface.size)
        print('final_index:', np.argsort(vect_final))
        print('final:', np.sort(vect_final))

        #neighbor = np.where(np.array(([sqrt(((self.coordinate[surface[ad_index]] - self.coordinate[_]) ** 2).sum()
        #                                    ) for _ in range(self.coordinate.shape[0])])) < 3)[0]
        possible_pos = self.coordinate0[1:13]

        # single random add
        '''while True:
            overlap = False
            ad_pos = self.coordinate[surface[ad_index]] + possible_pos[randrange(1, 13)]
            for i in range(self.distance.size):
                if not i == surface[ad_index]:
                    if sqrt(((ad_pos - self.coordinate[i]) ** 2).sum()) < self.min_distance[1]:
                        overlap = True
            if overlap:
                continue
            break
        self.coordinate = np.vstack((self.coordinate, ad_pos))
        self.distance = np.append(self.distance, sqrt((self.coordinate[-1] ** 2).sum()))
        self.element = np.append(self.element, self.element[-1])'''
        # cluster add
        '''for i in range(possible_pos.shape[0]):
            overlap = False
            ad_pos = self.coordinate[surface[ad_index]] + possible_pos[i]
            for j in range(self.distance.size):
                if not j == surface[ad_index]:
                    if sqrt(((ad_pos - self.coordinate[j]) ** 2).sum()) < self.min_distance[1]:
                        overlap = True
            if not overlap:
                self.coordinate = np.vstack((self.coordinate, ad_pos))
                self.distance = np.append(self.distance, sqrt((self.coordinate[-1] ** 2).sum()))
                self.element = np.append(self.element, self.element[-1])'''

        # single vector add
        off_center = possible_pos #- center[0]
        d_off_center = np.array([sqrt((off_center[_] ** 2).sum()) for _ in range(off_center.shape[0])])
        vect_angle = np.array([acos(((np.abs(off_center[_]) * i_vector).sum()) / sqrt((i_vector ** 2).sum())
                                    / d_off_center[_]) for _ in range(d_off_center.size)])
        for i in np.argsort(vect_angle):
            overlap = False
            ad_pos = self.coordinate[surface[ad_index]] + possible_pos[i]
            for i in range(self.distance.size):
                if not i == surface[ad_index]:
                    if sqrt(((ad_pos - self.coordinate[i]) ** 2).sum()) < self.min_distance[1]:
                        overlap = True
            if not overlap:
                self.coordinate = np.vstack((self.coordinate, ad_pos))
                self.distance = np.append(self.distance, sqrt((self.coordinate[-1] ** 2).sum()))
                self.element = np.append(self.element, self.element[-1])
                break

        '''distance_sum = np.zeros(self.coordinate0.shape[0])
        for i in range(1, self.coordinate0.shape[0]):
            ad_pos = self.coordinate[surface[ad_index]] + self.coordinate0[i]
            neighbor = np.where(np.array(([sqrt(((self.coordinate[surface[ad_index]] - self.coordinate[_]) ** 2).sum()
                                                ) for _ in range(self.coordinate.shape[0])])) < 3)[0]
            for j in neighbor:
                distance_sum[i] += sqrt(((ad_pos - self.coordinate[j]) ** 2).sum())'''

if __name__ == '__main__':
    exp_larch = EXP(r'D:\simulation\Cu207_sum.rex', 3, 9, 1, 2.7, 3, True)
    exp_rex = EXP(r'D:\simulation\Cu207_sum.rex', 3, 9, 1, 2.7, 3, False)
    exp_athena = EXP(r'D:\simulation\marked.chik3', 3, 9, 1, 2.7, 3, False)
    '''with open(r'D:\simulation\marked.chiq_re', 'r') as f:
        while True:
            line = f.readline()
            if not line.find('#  q') == -1:
                break
        k = np.array([])
        chi = np.array([])
        while True:
            line = f.readline()
            if not line or line.isspace():
                break
            temp = line.split()
            k = np.append(k, float(temp[0]))
            chi = np.append(chi, float(temp[1]))
    #plt.plot(k, chi, c='b')'''
    #exp_larch = EXP(r'J:\Monte Carlo\Tcu.rex', 3, 16, 0, 6, 3, True)
    #exp_rex = EXP(r'J:\Monte Carlo\Tcu.rex', 3, 16, 0, 6, 3, False)
    plt.plot(exp_larch.k, exp_larch.chi/2, c='k')
    plt.plot(exp_rex.k, exp_rex.chi, c='r')
    plt.plot(exp_athena.k, exp_athena.chi, c='b')
    plt.show()

