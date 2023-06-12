import time

from mrmc_package import EXP, RMC4, metropolis, Ui_MainWindow_Pol, folder_create, \
    scatter, line, init_3dplot, plot_Al2O3, plot_TiO2, fft_cut

import os
from time import perf_counter as timer
from time import sleep
from random import randrange

import numpy as np
from math import sqrt

from PyQt5.QtGui import QIcon, QFont, QCursor
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMenu, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from matplotlib import use
from sys import argv, exit
import icon_rc
use('Qt5Agg')



def chi_average(input_array=np.array([]), size=3):
    chi_sum = np.zeros((size, input_array[0].exp[0].k.size))
    for pol in range(size):
        for index in range(input_array.size):
            chi_sum[pol] += input_array[index].table[pol].chi
    return chi_sum / input_array.size


class Worker(QThread):
    sig_plotinfo = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    sig_plot3d = pyqtSignal(int, int, np.ndarray)
    sig_init3d = pyqtSignal(str, int)
    sig_folder = pyqtSignal(str)
    sig_tau = pyqtSignal(float, float)
    sig_backup = pyqtSignal(bool)

    sig_step = pyqtSignal(int)
    sig_time = pyqtSignal(float)
    sig_best = pyqtSignal(float)
    sig_current = pyqtSignal(float)
    sig_statistic = pyqtSignal(np.ndarray)
    sig_statusbar = pyqtSignal(str, int)
    sig_warning = pyqtSignal(str)
    sig_close = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.folder = ''
        self.t_base = 0.0
        self.t_end = 0.0
        self.r_now = 0
        self.r_lowest = 0
        self.step_count = 0
        self.flag = False
        self.t_start = 0
        self.rep = np.array([], dtype=RMC4)
        self.file_inp = ''
        self.backup_count = 2000
        self.fit_space = 'k'

        self.chi_sum = np.array([])
        self.chi_sum_best = np.array([])
        self.ft_sum = np.array([])
        self.rep_size = 0
        self.tau_t = 1e-3
        self.tau_i = 1e-2
        self.tau_ratio = self.tau_i / self.tau_t


    def init(self):
        self.rep = np.empty(self.rep_size, dtype=RMC4)
        self.folder = folder_create(self.simulation_name)
        os.makedirs(self.folder + r'\result')
        os.popen('copy "%s" "%s"' % (self.material_folder + r'\feff.inp', self.folder + r'\feff.inp'))
        os.popen('copy "%s" "%s"' % (self.file_inp, self.folder + r'\mrmc.inp'))
        # os.popen('copy "%s" "%s"' % (self.material_folder + r'\feff8.exe', self.folder + r'\feff8.exe'))
        sleep(0.5)
        for index in range(self.rep_size):
            self.rep[index] = RMC4(index, self.exp, self.sig2, self.E0, self.S0, data_base=self.material_folder,
                                   path=self.folder, init_pos=self.init_pos.copy(), init_element=self.init_element,
                                   spherical=self.spherical, random=self.random_init, local_range=self.local_range,
                                   surface=self.surface, surface_range=self.surface_range, r2chi=self.fit_space,
                                   step=self.step_min, step_range=self.step_max, ini_flag=True,
                                   ms=self.multiscattering_en, weight=self.weight, trial=self.trial)
            self.rep[index].table_init()
        print('replica created')
        with open(self.folder + r'/model.dat', 'w') as f:
            if not self.surface == '':
                f.write('[initial surface model]\n')
                for i in range(self.rep[0].cell.surface_e.size):
                    f.write('%s %.6f %.6f %.6f\n' % (self.rep[0].cell.surface_e[i],
                                                     self.rep[0].cell.surface_c[i][0],
                                                     self.rep[0].cell.surface_c[i][1],
                                                     self.rep[0].cell.surface_c[i][2]))
                for replica in self.rep:
                    f.write('%s %.6f %.6f %.6f\n' % (replica.cell.center_e,
                                                     replica.cell.center_c[0],
                                                     replica.cell.center_c[1],
                                                     replica.cell.center_c[2]))
                for replica in self.rep:
                    for i in range(replica.cell.satellite_e.size):
                        f.write('%s %.6f %.6f %.6f\n' % (replica.cell.satellite_e[i],
                                                         replica.cell.satellite_c[i][0],
                                                         replica.cell.satellite_c[i][1],
                                                         replica.cell.satellite_c[i][2]))
                f.write('\n')
            f.write('[initial local model]\n')
            f.write('%s %.6f %.6f %.6f\n' % (self.rep[0].cell.center_e, 0, 0, 0))
            for replica in self.rep:
                for i in range(1, replica.cell.element.size):
                    f.write('%s %.6f %.6f %.6f\n' % (replica.cell.element[i],
                                                     replica.cell.coordinate[i][0],
                                                     replica.cell.coordinate[i][1],
                                                     replica.cell.coordinate[i][2]))
            f.write('\n')
        print('initial model saved')
        self.chi_sum = chi_average(self.rep, self.exp.size)
        self.ft_sum = np.array([])
        if self.fit_space == 'k':
            r_new_pol = np.array([self.exp[_].r_factor_chi(self.chi_sum[_]) for _ in range(self.exp.size)])
        elif self.fit_space == 'r':
            self.ft_sum = np.array([fft_cut(self.chi_sum[_], self.exp[_].r, self.exp[_].r_start,
                                            self.exp[_].r_end) for _ in range(self.exp.size)])
            r_new_pol = np.array([self.exp[_].r_factor_ft(self.ft_sum[_]) for _ in range(self.exp.size)])
        else:
            self.ft_sum = np.array([fft_cut(self.chi_sum[_], self.exp[_].r, self.exp[_].r_start,
                                            self.exp[_].r_end) for _ in range(self.exp.size)])
            r_new_pol = np.array([self.exp[_].r_factor_cross(
                self.chi_sum[_] * np.transpose([self.ft_sum[_]])) for _ in range(self.exp.size)])
        self.sig_plotinfo.emit(self.chi_sum if not self.fit_space == 'r' else self.ft_sum, r_new_pol, self.ft_sum)
        self.sig_init3d.emit(self.surface, self.rep_size)
        self.r_now = r_new_pol.sum()
        self.r_lowest = self.r_now
        self.chi_sum_best = self.chi_sum.copy()
        self.step_count = 1
        self.sig_step.emit(self.step_count)

        print('information displayed')

    def get_folder(self):
        try:
            self.rep = np.empty(len(os.listdir(self.folder + r'\result')), dtype=RMC4)
        except ValueError:
            return False
        if not os.path.exists(self.folder + r'\backup'):
            os.makedirs(self.folder + r'\backup')
        if not os.path.exists(self.folder + r'\backup\result'):
            os.makedirs(self.folder + r'\backup\result')
        for i in range(self.exp.size):
            os.popen('copy "%s" "%s"' % (self.folder + r'\chi_sum%d.txt' % (i + 1),
                                         self.folder + r'\backup\chi_sum%d.txt' % (i + 1)))
        os.popen('copy "%s" "%s"' % (self.folder + r'\observe.png', self.folder + r'\backup\observe.png'))
        os.popen('copy "%s" "%s"' % (self.folder + r'\info.txt', self.folder + r'\backup\info.txt'))
        os.popen('copy "%s" "%s"' % (self.folder + r'\log.txt', self.folder + r'\backup\log.txt'))
        self.rep_size = self.rep.size
        print('folder:', self.folder)

        with open(self.folder + r'\info.txt', 'r') as info:
            '''self.init_element = np.array([info.readline().split()[1]])
            self.init_element = np.append(self.init_element, np.asarray(info.readline().split()[1:]))
            self.init_pos = np.zeros((self.init_element.size, 3))
            temp = info.readline().split()
            self.surface = temp[1] if len(temp) > 1 else ''
            self.E0 = [float(_) for _ in info.readline().split()[1:]]'''
            for i in range(6):
                info.readline()
            self.step_count = int(info.readline().split()[1])
            self.t_base = float(info.readline().split()[1])
            self.tau_t = float(info.readline().split()[1])
            self.tau_i = float(info.readline().split()[1])
            temp = info.readline().split(': ')[1]
            self.r_lowest = float(temp.split(' (')[0])
            temp_pol = np.asarray(temp.split(' (')[1].split(')')[0].split())
            self.r_lowest_pol = np.array([float(_) for _ in temp_pol])
        print('information read')

        for index in range(self.rep_size):
            self.rep[index] = RMC4(index, self.exp, self.sig2, self.E0, self.S0, data_base=self.material_folder,
                                   path=self.folder, init_pos=self.init_pos.copy(), init_element=self.init_element,
                                   spherical=self.spherical, random=self.random_init, local_range=self.local_range,
                                   surface=self.surface, surface_range=self.surface_range, r2chi=self.fit_space,
                                   step=self.step_min, step_range=self.step_max, ini_flag=False,
                                   ms=self.multiscattering_en, weight=self.weight, trial=self.trial)
            print('replica %d created' % index)
            self.rep[index].read_result()

        self.chi_sum = chi_average(self.rep, self.exp.size)
        self.ft_sum = np.array([])
        if self.fit_space == 'k':
            r_new_pol = np.array([self.exp[_].r_factor_chi(self.chi_sum[_]) for _ in range(self.exp.size)])
        elif self.fit_space == 'r':
            self.ft_sum = np.array([fft_cut(self.chi_sum[_], self.exp[_].r, self.exp[_].r_start,
                                            self.exp[_].r_end) for _ in range(self.exp.size)])
            r_new_pol = np.array([self.exp[_].r_factor_ft(self.ft_sum[_]) for _ in range(self.exp.size)])
        else:
            self.ft_sum = np.array([fft_cut(self.chi_sum[_], self.exp[_].r, self.exp[_].r_start,
                                            self.exp[_].r_end) for _ in range(self.exp.size)])
            self.cross_sum = np.array([self.chi_sum[_] * np.transpose([self.ft_sum[_]]) for _ in range(self.exp.size)])
            r_new_pol = np.array([self.exp[_].r_factor_cross(self.cross_sum[_]) for _ in range(self.exp.size)])
        self.r_now = r_new_pol.sum()
        self.chi_sum_best = self.chi_sum.copy()

        self.sig_step.emit(self.step_count)
        self.sig_time.emit(self.t_base)
        self.sig_best.emit(self.r_lowest)
        self.sig_current.emit(self.r_now)

        self.sig_tau.emit(self.tau_t, self.tau_i)
        self.tau_ratio = self.tau_i / self.tau_t

        self.sig_plotinfo.emit(self.chi_sum if not self.fit_space == 'r' else self.ft_sum, r_new_pol, self.ft_sum)
        self.sig_init3d.emit(self.surface, self.rep_size)
        print('information displayed')
        return True

    def read_inp(self, file):
        with open(file, 'r') as f:
            while True:
                if not f.readline().find('experimental') == -1:
                    break
            exp_path = np.array([])
            for i in range(3):
                temp = f.readline()
                if not temp or temp.find('exp') == -1:
                    self.sig_warning.emit('exp formal error')
                    return False
                temp = temp.split(temp.split(':')[0] + ':')[1].strip()
                if not temp or len(temp.split()) == 0:
                    break
                exp_path = np.append(exp_path, temp.split('\n')[0])
            if exp_path.size == 0:
                self.sig_warning.emit('no experimental data')
                return False

            weight = f.readline().split(':')
            if weight[0].find('weight') == -1:
                self.sig_warning.emit('weight formal error')
                return False
            self.weight = int(weight[1])

            while True:
                if not f.readline().find('model') == -1:
                    break

            rep_size = f.readline().split(':')
            if rep_size[0].find('replicas_size') == -1:
                self.sig_warning.emit('rep_size formal error')
                return False
            self.rep_size = int(rep_size[1])

            temp = f.readline().split(':')
            if temp[0].find('surface') == -1:
                self.sig_warning.emit('surface formal error')
                return False
            surface = temp[1].strip()
            if len(surface) == 0 or (not surface == 'TiO2' and not surface == 'Al2O3'):
                self.surface = ''
            else:
                self.surface = surface

            if f.readline().find('center_atom') == -1:
                self.sig_warning.emit('center_atom line missing')
                return False

            self.init_pos = np.array([])
            self.init_element = np.array([])
            temp = f.readline().split()
            self.init_element = np.append(self.init_element, temp[0])
            try:
                self.init_pos = np.append(self.init_pos, np.array([float(temp[1]), float(temp[2]), float(temp[3])]))
            except ValueError or IndexError:
                self.sig_warning.emit('center atom format error')
                return False

            if f.readline().find('satellite_atoms') == -1:
                self.sig_warning.emit('satellite_atoms line missing')
                return False
            while True:
                lines = f.readline()
                if not lines.split():
                    continue
                if not lines.find('coordinate') == -1:
                    self.sig_warning.emit('END indicator missing')
                    return False
                if not lines.find('END') == -1:
                    break
                sat = lines.split()
                self.init_element = np.append(self.init_element, sat[0])
                try:
                    self.init_pos = np.append(self.init_pos, np.array([float(sat[1]), float(sat[2]), float(sat[3])]))
                except ValueError or IndexError:
                    self.sig_warning.emit('satellite atoms format error')
                    return False
            self.init_pos = np.reshape(self.init_pos, (self.init_element.size, 3))

            spherical = f.readline().split(':')
            if spherical[0].find('coordinate_system') == -1:
                self.sig_warning.emit('coordinate system line missing')
                return False
            temp = spherical[1].strip()
            if temp == 'True' or temp == 'true' or temp == '1' or temp == 'spherical':
                self.spherical = True
            elif temp == 'False' or temp == 'false' or temp == '0' or temp == 'cartesian':
                self.spherical = False
            else:
                self.sig_warning.emit('coordinate system parameter error')
                return False

            random = f.readline().split(':')
            if random[0].find('random_deposition') == -1:
                self.sig_warning.emit('random_deposition line missing')
                return False
            temp = random[1].strip()
            if temp == 'True' or temp == 'true' or temp == '1':
                self.random_init = True
            elif temp == 'False' or temp == 'false' or temp == '0':
                self.random_init = False
            else:
                self.sig_warning.emit('random deposition parameter error')
                return False

            pattern = f.readline().split(':')
            if pattern[0].find('move_pattern') == -1:
                self.sig_warning.emit('move pattern missing')
                return False
            temp = pattern[1].strip()
            if temp == 'True' or temp == 'true' or temp == '1':
                self.move_pattern = True
            elif temp == 'False' or temp == 'false' or temp == '0':
                self.move_pattern = False
                self.backup_count = 2000 * self.rep_size
            else:
                self.sig_warning.emit('move pattern parameter error')
                return False

            step_min = f.readline().split(':')
            if step_min[0].find('trial_count') == -1:
                self.sig_warning.emit('trial count missing')
                return False
            temp = step_min[1].split()
            if len(temp) == 0:
                self.trial = np.array([50, 50])
            elif len(temp) == 2:
                self.trial = np.array([float(_) for _ in temp])
            else:
                self.sig_warning.emit('trial count parameter error')
                return False

            step_min = f.readline().split(':')
            if step_min[0].find('step_min') == -1:
                self.sig_warning.emit('minimum step missing')
                return False
            temp = step_min[1].split()
            if len(temp) == 0:
                self.step_min = np.array([0.01, 0.01])
            elif len(temp) == 2:
                self.step_min = np.array([float(_) for _ in temp])
            else:
                self.sig_warning.emit('minimum step format error')
                return False

            step_max = f.readline().split(':')
            if step_max[0].find('step_max') == -1:
                self.sig_warning.emit('maximum step missing')
                return False
            temp = step_max[1].split()
            if len(temp) == 0:
                self.step_max = np.array([2, 0.1])
            elif len(temp) == 2:
                self.step_max = np.array([float(_) for _ in temp])
            else:
                self.sig_warning.emit('maximum step format error')
                return False

            surface_range = f.readline().split(':')
            if surface_range[0].find('surface_range') == -1:
                self.sig_warning.emit('surface range missing')
                return False
            temp = surface_range[1].split()
            if len(temp) == 6:
                self.surface_range = np.array([float(_) for _ in temp])
            elif len(temp) == 0:
                self.surface_range = np.array([])
            else:
                self.sig_warning.emit('surface range format error')
                return False

            while True:
                if not f.readline().find('spectrum') == -1:
                    break

            s0 = f.readline().split(':')
            if s0[0].find('S0') == -1:
                self.sig_warning.emit('S0 not found')
                return False
            if len(s0) == 2:
                try:
                    self.S0 = float(s0[1])
                except ValueError:
                    self.sig_warning.emit('S0 format error')
                    return False
            else:
                self.S0 = 1

            sig2 = f.readline().split(':')
            if sig2[0].find('SIG2') == -1:
                self.sig_warning.emit('sig2 not found')
                return False
            if len(sig2) == 2:
                try:
                    self.sig2 = float(sig2[1])
                except ValueError:
                    self.sig_warning.emit('sig2 format error')
                    return False
            else:
                self.sig2 = 0

            temp = f.readline().split(':')
            if temp[0].find('k_range') == -1:
                self.sig_warning.emit('k range not found')
                return False
            temp = temp[1].split()
            try:
                k_range = np.array([float(temp[0]), float(temp[1])])
            except ValueError or IndexError:
                self.sig_warning.emit('k range format error')
                return False

            temp = f.readline().split(':')
            if temp[0].find('r_range') == -1:
                self.sig_warning.emit('r range not found')
                return False
            temp = temp[1].split()
            try:
                r_range = np.array([float(temp[0]), float(temp[1])])
            except ValueError or IndexError:
                self.sig_warning.emit('r range format error')
                return False
            if k_range[0] >= k_range[1] or r_range[0] >= r_range[1]:
                self.sig_warning.emit('r or k range value error')
                return False

            self.exp = np.array([EXP(_, k_range[0], k_range[1], r_range[0], r_range[1]) for _ in exp_path])
            for pol in self.exp:
                if pol.chi.size < (20 * (k_range[1] - k_range[0])) - 1:
                    self.sig_warning.emit('k range exceeds experimental data')
                    return False

            e0 = f.readline().split(':')
            if e0[0].find('delta_E') == -1:
                self.sig_warning.emit('E0 not found')
                return False
            temp = e0[1].split()
            amount = self.init_element.size - 1
            if not self.surface == '':
                amount += 2
            if not len(temp) == amount:
                self.sig_warning.emit('Please set E0 for %d atoms' % amount)
                return False
            self.E0 = np.array([float(_) for _ in temp])

            rpath = f.readline().split(':')
            if rpath[0].find('rpath') == -1:
                self.sig_warning.emit('rpath not found')
                return False
            try:
                rpath = rpath[1].split()
                self.local_range = np.array([float(rpath[_]) for _ in range(len(rpath))])
            except ValueError:
                self.sig_warning.emit('rpath format error')
                return False

            mts = f.readline().split(':')
            if mts[0].find('multiscattering') == -1:
                self.sig_warning.emit('multi-scattering line missing')
                return False
            temp = mts[1].strip()
            if temp == 'True' or temp == 'true' or temp == 1:
                self.multiscattering_en = True
            elif temp == 'False' or temp == 'false' or temp == 0:
                self.multiscattering_en = False
            else:
                self.sig_warning.emit('multi-scattering parameter error')
                return False

            posi = f.tell()
            fspace = f.readline().split(':')
            if fspace[0].find('fitting_space') == -1:
                f.seek(posi)
                self.fit_space = 'k'
            else:
                temp = fspace[1].strip()
                if temp == 'K' or temp == 'k':
                    self.fit_space = 'k'
                elif temp == 'R' or temp == 'r':
                    self.fit_space = 'r'
                elif temp == 'X' or temp == 'x':
                    self.fit_space = 'x'
                else:
                    self.sig_warning.emit('fitting space parameter error')
                    return False

            while True:
                if not f.readline().find('path') == -1:
                    break

            temp = f.readline()
            if temp.split()[0].find('material_folder') == -1:
                self.sig_warning.emit('material folder missing')
                return False
            self.material_folder = temp.split(temp.split(':')[0] + ':')[1].split('\n')[0].strip()
            if not os.path.exists(self.material_folder + r'\table.ini'):
                self.sig_warning.emit('table.ini not found in selected folder')
                return False
            temp = f.readline()
            if temp.split()[0].find('simulation_folder') == -1:
                self.sig_warning.emit('simulation folder missing')
                return False
            self.simulation_name = temp.split(temp.split(':')[0] + ':')[1].split('\n')[0].strip()
            if self.simulation_name[-1] == '\\' or self.simulation_name[-1] == '/':
                self.simulation_name = self.simulation_name[:-1]

        # parameters region
        print(exp_path)
        print('weight:%d\nreplica size:%d\nsurface:%s\nmoving atoms:' % (self.weight, self.rep_size, self.surface))
        print(self.init_pos)
        print(self.init_element)
        print('spherical coordinate:', self.spherical, '\nrandom deposition:', self.random_init,
              '\nmove pattern:', self.move_pattern)
        print('step_min:%.3f %.3f\nstep_max:%.3f %.3f\nS0:%f\nsig2:%f' %
              (self.step_min[0], self.step_min[1], self.step_max[0], self.step_max[1], self.S0, self.sig2))
        print('k range:%f %f\nr range:%f %f' % (k_range[0], k_range[1], r_range[0], r_range[1]))
        print('E0:', self.E0)
        print('rpath:', self.local_range)
        print('multi scattering:', self.multiscattering_en)
        print('material folder:%s\nsimulation name:%s' % (self.material_folder, self.simulation_name))

        self.r_now_pol = np.zeros(self.exp.size)
        self.r_lowest_pol = np.zeros(self.exp.size)
        self.chi_sum = np.zeros((self.exp.size, self.exp[0].k.size))
        self.chi_sum_best = np.zeros((self.exp.size, self.exp[0].k.size))

        self.tau_t = 1e-3
        self.tau_ratio = round(sqrt(self.rep_size), 1)
        self.tau_i = self.tau_t * self.tau_ratio
        self.tca_en = False
        return True

    def run(self):
        self.sig_statusbar.emit('Running', 0)
        trials = np.zeros(self.rep_size)
        stamp0 = timer()
        while True:
            if self.step_count % 100 == 0:
                stamp1 = timer()
                if stamp1 - stamp0 < 1:
                    sleep(1)
                stamp0 = stamp1
            if self.step_count % self.backup_count == 0:
                self.sig_backup.emit(True)
                self.flag = False
            '''rep_rf = np.array([replica.r_factor_t for replica in self.rep])
            deviate = np.where(rep_rf > rep_rf.mean() + rep_rf.var() * 2)[0]
            if deviate.size > 0:
                move_array = deviate if self.move_pattern else deviate[randrange(0, deviate.size)]
            else:'''
            move_array = np.arange(self.rep_size) if self.move_pattern else np.array([randrange(0, self.rep_size)])
            trials -= trials
            for replica in move_array:
                trials[replica] = self.rep[replica].walk(self.tau_i)
            if trials.sum() > 0:
                self.step_count += 1
                self.sig_step.emit(self.step_count)
                self.chi_sum = chi_average(self.rep, self.exp.size)
                if self.fit_space == 'k':
                    r_new_pol = np.array([self.exp[_].r_factor_chi(self.chi_sum[_]) for _ in range(self.exp.size)])
                elif self.fit_space == 'r':
                    self.ft_sum = np.array([fft_cut(self.chi_sum[_], self.exp[_].r, self.exp[_].r_start,
                                                    self.exp[_].r_end) for _ in range(self.exp.size)])
                    r_new_pol = np.array([self.exp[_].r_factor_ft(self.ft_sum[_]) for _ in range(self.exp.size)])
                else:
                    self.ft_sum = np.array([fft_cut(self.chi_sum[_], self.exp[_].r, self.exp[_].r_start,
                                                    self.exp[_].r_end) for _ in range(self.exp.size)])
                    self.cross_sum = np.array(
                        [self.chi_sum[_] * np.transpose([self.ft_sum[_]]) for _ in range(self.exp.size)])
                    r_new_pol = np.array([self.exp[_].r_factor_cross(self.cross_sum[_]) for _ in range(self.exp.size)])
                r_new = r_new_pol.sum()
                if metropolis(self.r_now, r_new, self.tau_t):
                    '''if deviate.size > 0:
                        print(rep_rf.mean(), rep_rf.var(), rep_rf[deviate])'''
                    if r_new < self.r_lowest:
                        self.sig_best.emit(self.r_lowest)
                        self.r_lowest = r_new
                        self.r_lowest_pol = r_new_pol.copy()
                        self.chi_sum_best = self.chi_sum.copy()
                        for replica in move_array:
                            if not self.surface == '':
                                self.rep[replica].cell.c_best = self.rep[replica].cell.cw_temp.copy()
                            else:
                                self.rep[replica].cell.c_best = self.rep[replica].cell.c_temp.copy()

                    for i in np.where(trials > 0)[0]:
                        if not self.surface == '':
                            base = self.rep[i].cell.surface_e.size
                            if self.rep[i].moved_atom == 0:
                                for j in range(self.rep[i].cell.local_size):
                                    self.sig_plot3d.emit(i, j, self.rep[i].cell.cw_temp[base + j]
                                                         - self.rep[i].cell.coordinate_whole[base + j])
                            else:
                                self.sig_plot3d.emit(i, self.rep[i].moved_atom,
                                                     self.rep[i].cell.cw_temp[base + self.rep[i].moved_atom] -
                                                     self.rep[i].cell.coordinate_whole[base + self.rep[i].moved_atom])
                        else:
                            self.sig_plot3d.emit(i, self.rep[i].moved_atom,
                                                 self.rep[i].cell.c_temp[self.rep[i].moved_atom] -
                                                 self.rep[i].cell.coordinate[self.rep[i].moved_atom])
                    for replica in move_array:
                        if not trials[replica] == 0:
                            self.rep[replica].accept()
                    self.sig_plotinfo.emit(self.chi_sum if not self.fit_space == 'r' else self.ft_sum,
                                           r_new_pol, self.ft_sum)
                    self.r_now = r_new
                    self.r_now_pol = r_new_pol.copy()
                    self.sig_current.emit(self.r_now)
                else:
                    for replica in move_array:
                        if not trials[replica] == 0:
                            self.rep[replica].reject()
                self.sig_time.emit(timer() - self.t_start + self.t_base)
            if not self.flag:
                break


class MainWindow(QMainWindow, Ui_MainWindow_Pol):
    def __init__(self):
        # from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
        super(MainWindow, self).__init__()
        print('*******************************************************************************************************')
        print('                                       Micro Reverse Monte Carlo                                       ')
        print('*******************************************************************************************************')
        self.setupUi(self)
        self.setWindowTitle('mRMC')
        self.setWindowIcon(QIcon(':\\mRMC.ico'))
        self.setMinimumSize(0, 0)

        self.thread = Worker()
        self.thread.sig_warning.connect(self.warning_window)
        self.thread.sig_close.connect(self.close_signal)

        self.polx = pg.PlotWidget(background=(255, 255, 255, 255))
        self.poly = pg.PlotWidget(background=(255, 255, 255, 255))
        self.polz = pg.PlotWidget(background=(255, 255, 255, 255))
        self.polx2 = pg.PlotWidget(background=(255, 255, 255, 255))
        self.poly2 = pg.PlotWidget(background=(255, 255, 255, 255))
        self.polz2 = pg.PlotWidget(background=(255, 255, 255, 255))
        self.polxyz = {0: self.polx, 1: self.poly, 2: self.polz,
                       3: self.polx2, 4: self.poly2, 5: self.polz2}
        self.layoutxyz = {0: self.g2dxLayout, 1: self.g2dyLayout, 2: self.g2dzLayout}
        self.plot = {}
        self.plotex = {}

        self.model = gl.GLViewWidget()
        self.model.setObjectName('model')
        self.scatter = np.array([])
        self.bond = np.array([])
        self.model.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model.customContextMenuRequested.connect(self.save_3d_menu)

        self.tau_t_value_box.valueChanged.connect(self.tau_t_change)
        self.tau_t_degree_box.valueChanged.connect(self.tau_t_change)
        self.tau_i_value_box.valueChanged.connect(self.tau_i_change)
        self.tau_i_degree_box.valueChanged.connect(self.tau_i_change)
        self.ratioCheckBox.clicked.connect(self.tau_ratio_lock)

        self.startButton.clicked.connect(self.cal_start)
        self.endButton.clicked.connect(self.cal_end)
        self.continueButton.clicked.connect(self.restart)
        self.actioninp.triggered.connect(self.read_inp)
        self.action_new.triggered.connect(self.new_event)
        self.action_open.triggered.connect(self.folder_read)
        self.action_log.triggered.connect(self.force_log)
        self.action_read_ini.triggered.connect(self.read_init_pos)
        self.startButton.setEnabled(False)
        self.action_new.setEnabled(False)
        self.continueButton.setEnabled(False)

        self.thread.sig_plotinfo.connect(self.pol_info)
        self.thread.sig_init3d.connect(self.plot_init)
        self.thread.sig_plot3d.connect(self.plot_3D_event)
        self.thread.sig_step.connect(self.stepLCD_change)
        self.thread.sig_time.connect(self.time_display)
        self.thread.sig_best.connect(self.best_r_display)
        self.thread.sig_current.connect(self.current_r_display)
        self.thread.sig_statistic.connect(self.statistic_display)
        self.thread.sig_tau.connect(self.tau_init)
        self.thread.sig_backup.connect(self.force_log)

    def read_inp(self):
        file_name = QFileDialog.getOpenFileName(self, 'select inp file...', os.getcwd(), '*.inp')
        if file_name[0] == '':
            return
        self.statusbar.showMessage('Reading', 0)
        self.thread.file_inp = file_name[0].replace('/', '\\')
        if self.thread.read_inp(self.thread.file_inp):
            self.window_init()

    def folder_read(self):
        self.thread.folder = QFileDialog.getExistingDirectory(self, 'select folder...', os.getcwd())
        if self.thread.folder == '':
            return
        self.startButton.setEnabled(False)
        self.statusbar.showMessage('Reading', 0)
        if self.thread.read_inp(self.thread.folder + r'\mrmc.inp'):
            self.window_init()
        if self.thread.get_folder():
            self.statusbar.showMessage('Done!', 3000)
            sleep(3)
            self.startButton.setEnabled(True)
            self.actioninp.setEnabled(False)
        else:
            self.statusbar.showMessage('Error!', 3000)

    def window_init(self):
        self.startButton.setEnabled(True)
        self.action_new.setEnabled(True)

        if not len(self.plot) == 0:
            for i in range(len(self.plot)):
                self.plot[i].removeItem(self.line_exp[i])
            if self.thread.fit_space == 'x':
                for i in range(len(self.plotex)):
                    self.plotex[i].removeItem(self.line_exp[i+len(self.plotex)])
            if len(self.plotex) > 0 and not self.thread.fit_space == 'x':
                for i in range(3):
                    self.layoutxyz[i].removeWidget(self.plotex[i])

        if self.thread.fit_space == 'x':
            for i in range(3):
                self.layoutxyz[i].addWidget(self.polxyz[i+3])
        if self.thread.exp.size == 3:
            title = ['Polarization [001]', 'Polarization [1-10]', 'Polarization [110]']
        elif self.thread.exp.size == 2:
            title = ['Polarization S', '', 'Polarization P']
        else:
            title = ['Spectrum']
        for i in range(3):
            self.polxyz[i].setTitle(title[i], color='#000000', size='18pt')
            if self.thread.fit_space == 'x':
                self.polxyz[i+3].setTitle(title[i], color='#000000', size='18pt')
        if self.thread.exp.size == 3:
            self.plot = {0: self.polx, 1: self.poly, 2: self.polz}
            if self.thread.fit_space == 'x':
                self.plotex = {0: self.polx2, 1: self.poly2, 2: self.polz2}
            self.current_label_p1.setText('[001]')
            self.current_label_p2.setText('[1-10]')
            self.current_label_p3.setText('[110]')
        elif self.thread.exp.size == 2:
            self.plot = {0: self.polx, 1: self.polz}
            if self.thread.fit_space == 'x':
                self.plotex = {0: self.polx2, 1: self.polz2}
            self.current_label_p1.setText('S-pol')
            self.current_label_p2.setText('P-pol')
            self.current_label_p3.setText('NULL')
        else:
            self.plot = {0: self.polx}
            if self.thread.fit_space == 'x':
                self.plotex = {0: self.polx2}
            self.current_label_p1.setText('spectrum')
            self.current_label_p2.setText('NULL')
            self.current_label_p3.setText('NULL')

        fn = QFont()
        fn.setPointSize(15)
        name = ['χ', 'k'] if not self.thread.fit_space == 'r' else ['FT', 'R']
        for i in range(3):
            self.polxyz[i].setLabel('left', name[0])
            self.polxyz[i].setLabel('bottom', name[1])
            self.polxyz[i].getAxis('left').setTickFont(fn)
            self.polxyz[i].getAxis('left').setTextPen('black')
            self.polxyz[i].getAxis('bottom').setTickFont(fn)
            self.polxyz[i].getAxis('bottom').setTextPen('black')
            self.layoutxyz[i].addWidget(self.polxyz[i])
            if self.thread.fit_space == 'x':
                self.polxyz[i+3].setLabel('left', 'FT')
                self.polxyz[i+3].setLabel('bottom', 'R')
                self.polxyz[i+3].getAxis('left').setTickFont(fn)
                self.polxyz[i+3].getAxis('left').setTextPen('black')
                self.polxyz[i+3].getAxis('bottom').setTickFont(fn)
                self.polxyz[i+3].getAxis('bottom').setTextPen('black')

        if not self.thread.fit_space == 'r':
            self.line_exp = np.array([line(self.thread.exp[_].k, self.thread.exp[_].chi, c='black', width=3)
                                      for _ in range(self.thread.exp.size)])
            if self.thread.fit_space == 'x':
                self.line_exp = np.append(self.line_exp, np.array([
                    line(self.thread.exp[_].r_cut, self.thread.exp[_].ft_cut, c='black', width=3)
                    for _ in range(self.thread.exp.size)]))
        else:
            self.line_exp = np.array([line(self.thread.exp[_].r_cut, self.thread.exp[_].ft_cut, c='black', width=3)
                                      for _ in range(self.thread.exp.size)])
        self.line_chi = np.array([])
        for i in range(len(self.plot)):
            self.plot[i].addItem(self.line_exp[i])
            if self.thread.fit_space == 'x':
                self.plotex[i].addItem(self.line_exp[i + len(self.plot)])

        self.r_aver = np.zeros(self.thread.init_element.size - 1)
        self.c2 = np.zeros(self.thread.init_element.size - 1)
        self.c3 = np.zeros(self.thread.init_element.size - 1)
        if self.r_aver.size == 0:
            self.CuO_Box.setTitle('Null')
            self.CuS_Box.setTitle('Null')
        if self.r_aver.size > 0:
            self.CuO_Box.setTitle('%s-%s bond' % (self.thread.init_element[0], self.thread.init_element[1]))
        else:
            self.CuO_Box.setTitle('Null')
        if self.r_aver.size > 1:
            self.CuS_Box.setTitle('%s-%s bond' % (self.thread.init_element[0], self.thread.init_element[2]))
        else:
            self.CuS_Box.setTitle('Null')
        self.tau_t_value_box.setValue(float(('%.1e' % self.thread.tau_t).split('e-')[0]))
        self.tau_t_degree_box.setValue(int(('%.1e' % self.thread.tau_t).split('e-')[1]))
        self.tau_i_value_box.setValue(float(('%.1e' % self.thread.tau_i).split('e-')[0]))
        self.tau_i_degree_box.setValue(int(('%.1e' % self.thread.tau_i).split('e-')[1]))

    def cal_start(self):
        self.thread.flag = True
        if self.thread.folder == '':
            self.thread.init()
            self.actioninp.setEnabled(False)
        self.setWindowTitle('mRMC (%s)' % self.thread.folder)
        self.thread.t_start = timer()
        print('iteration started')
        self.continueButton.setEnabled(False)
        self.startButton.setEnabled(False)
        self.action_new.setEnabled(False)
        self.action_open.setEnabled(False)
        self.action_log.setEnabled(True)
        self.thread.start()

    def restart(self):
        self.thread.flag = True
        self.thread.t_start = timer()
        self.thread.t_base = self.thread.t_end
        self.continueButton.setEnabled(False)
        self.thread.start()

    def cal_end(self):
        self.statusbar.showMessage('Saving...', 0)
        self.thread.t_end = timer() - self.thread.t_start + self.thread.t_base
        sleep(0.1)
        if self.thread.flag:
            self.thread.flag = False
            if self.thread.isRunning():
                self.thread.wait()
            print('iteration ended')
            for replica in self.thread.rep:
                replica.write_result(self.thread.r_lowest, self.thread.r_now, self.thread.folder + r'\result')
            print('replica result wrote')
            with open(self.thread.folder + r'\log.txt', 'w') as f:
                for replica in self.thread.rep:
                    f.write('replica %d:\n' % replica.index)
                    for pol in range(self.thread.exp.size):
                        for dat in replica.table[pol].log_1st:
                            f.write('%d ' % dat)
                        f.write('\n')
                        for xi in range(replica.table[pol].k.size):
                            f.write('%f %f\n' % (replica.table[pol].k[xi], replica.table[pol].chi[xi]))
                        f.write('\n')
            print('log data wrote')
            self.write_model()
            print('model wrote')
            for pol in range(self.thread.exp.size):
                with open(self.thread.folder + r'\chi_sum%d.txt' % (pol + 1), 'w') as result:
                    result.write('[calculated spectrum]\n')
                    for _ in range(self.thread.chi_sum[pol].size):
                        result.write('   %f     %f\n' % (self.thread.exp[pol].k[_], self.thread.chi_sum[pol][_]))
                    result.write('\n[experimental spectrum]\n')
                    for _ in range(self.thread.chi_sum[pol].size):
                        result.write('   %f     %f\n' % (self.thread.exp[pol].k[_], self.thread.exp[pol].chi[_]))
                    if not self.thread.fit_space == 'k':
                        result.write('\n[calculated FT]\n')
                        for _ in range(self.thread.ft_sum[pol].size):
                            result.write('   %f     %f\n' % (self.thread.exp[pol].k[_], self.thread.chi_sum[pol][_]))
                        result.write('\n[experimental FT]\n')
                        for _ in range(self.thread.ft_sum[pol].size):
                            result.write('   %f     %f\n' % (self.thread.exp[pol].r[_], self.thread.exp[pol].ft[_]))
                    if self.thread.fit_space == 'x':
                        print(self.thread.cross_sum.shape)
                        result.write('[calculated cross]\n')
                        for i in range(self.thread.exp[pol].cross.shape[0]):
                            for j in range(self.thread.exp[pol].cross.shape[1]):
                                result.write('%.3f ' % self.thread.cross_sum[pol][i][j])
                            result.write('\n')
                        result.write('\n[experimental cross]\n')
                        for i in range(self.thread.exp[pol].cross.shape[0]):
                            for j in range(self.thread.exp[pol].cross.shape[1]):
                                result.write('%.3f ' % self.thread.exp[pol].cross[i][j])
                            result.write('\n')
            print('chi wrote')
            self.write_info(self.thread.folder)
            print('information wrote')
            fig = self.graphWidget.grab()
            fig.save(self.thread.folder + r'\observe.png', 'PNG')
            image = self.model.renderToArray((1000, 1000))
            pg.makeQImage(image).save(self.thread.folder + r'\model.png')
            self.statusbar.showMessage('Done!', 3000)
            self.continueButton.setEnabled(True)

    def force_log(self):
        self.statusbar.showMessage('backup...', 0)
        if not os.path.exists(self.thread.folder + r'\log'):
            os.makedirs(self.thread.folder + r'\log')
            os.makedirs(self.thread.folder + r'\log\result')
        for replica in self.thread.rep:
            replica.write_result(self.thread.r_lowest, self.thread.r_now, self.thread.folder + r'\log\result')
        self.thread.t_end = timer() - self.thread.t_start + self.thread.t_base
        self.write_info(self.thread.folder + r'\log')
        self.statusbar.showMessage('backup finished!', 3000)
        print('backup finished.', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        self.restart()

    def write_info(self, folder):
        with open(folder + r'\info.txt', 'w') as info:
            info.write('center_atom: %s\n' % self.thread.init_element[0])
            info.write('satellite_atoms: ')
            for i in range(self.thread.init_element.size - 1):
                info.write('%s ' % self.thread.init_element[i + 1])
            info.write('\n')
            info.write('surface: %s\n' % self.thread.surface)
            info.write('dE: ')
            for i in range(len(self.thread.E0)):
                info.write('%f ' % self.thread.E0[i])
            info.write('\n')
            info.write('k_range: %f %f\n' % (self.thread.exp[0].k_start, self.thread.exp[0].k_end))
            info.write('r_range: %f %f\n' % (self.thread.exp[0].r_start, self.thread.exp[0].r_end))
            info.write('step: %d\n' % self.thread.step_count)
            info.write('time: %f\n' % self.thread.t_end)
            info.write('tau_t: %.1e\n' % self.thread.tau_t)
            info.write('tau_i: %.1e\n' % self.thread.tau_i)
            if self.thread.exp.size == 2:
                info.write('best_R_factor(s, p): %f (%f %f)\n' % (self.thread.r_lowest,
                                                                  self.thread.r_lowest_pol[0],
                                                                  self.thread.r_lowest_pol[1]))
            elif self.thread.exp.size == 3:
                info.write('best_R_factor([001], [1-10], [110]): %f (%f %f %f)\n' % (self.thread.r_lowest,
                                                                                     self.thread.r_lowest_pol[0],
                                                                                     self.thread.r_lowest_pol[1],
                                                                                     self.thread.r_lowest_pol[2]))
            for i in range(self.thread.init_element.size - 1):
                info.write('%s-%s bond(r,c2,c3): %f    %f    %f\n' % (self.thread.init_element[0],
                                                                      self.thread.init_element[i + 1],
                                                                      self.r_aver[i], self.c2[i], self.c3[i]))

    def write_model(self):
        with open(self.thread.folder + r'/model.dat', 'r+') as f:
            while True:
                lines = f.readline()
                if not lines.find('initial local model') == -1:
                    break
            while True:
                lines = f.readline()
                if lines.isspace():
                    break
            f.truncate()

            if not self.thread.surface == '':
                f.seek(f.tell())
                f.write('[final surface model]\n')
                for i in range(self.thread.rep[0].cell.surface_e.size):
                    f.seek(f.tell())
                    f.write('%s %.6f %.6f %.6f\n' % (self.thread.rep[0].cell.surface_e[i],
                                                     self.thread.rep[0].cell.surface_c[i][0],
                                                     self.thread.rep[0].cell.surface_c[i][1],
                                                     self.thread.rep[0].cell.surface_c[i][2]))
                base = self.thread.rep[0].cell.surface_e.size
                for replica in self.thread.rep:
                    f.seek(f.tell())
                    f.write('%s %.6f %.6f %.6f\n' % (replica.cell.element_whole[base],
                                                     replica.cell.coordinate_whole[base][0],
                                                     replica.cell.coordinate_whole[base][1],
                                                     replica.cell.coordinate_whole[base][2]))
                for replica in self.thread.rep:
                    for i in range(1, replica.cell.local_size):
                        f.seek(f.tell())
                        f.write('%s %.6f %.6f %.6f\n' % (replica.cell.element_whole[base + i],
                                                         replica.cell.coordinate_whole[base + i][0],
                                                         replica.cell.coordinate_whole[base + i][1],
                                                         replica.cell.coordinate_whole[base + i][2]))
                f.seek(f.tell())
                f.write('\n')
            f.seek(f.tell())
            f.write('[final local model]\n')
            f.seek(f.tell())
            f.write('%s %.6f %.6f %.6f\n' % (self.thread.rep[0].cell.center_e, 0, 0, 0))
            for replica in self.thread.rep:
                for i in range(1, replica.cell.element.size):
                    f.seek(f.tell())
                    f.write('%s %.6f %.6f %.6f\n' % (replica.cell.element[i],
                                                     replica.cell.coordinate[i][0],
                                                     replica.cell.coordinate[i][1],
                                                     replica.cell.coordinate[i][2]))
            f.seek(f.tell())
            f.write('\n')
            f.seek(f.tell())
            if not self.thread.surface == '':
                f.write('[best surface model]\n')
                for i in range(self.thread.rep[0].cell.surface_e.size):
                    f.seek(f.tell())
                    f.write('%s %.6f %.6f %.6f\n' % (self.thread.rep[0].cell.surface_e[i],
                                                     self.thread.rep[0].cell.surface_c[i][0],
                                                     self.thread.rep[0].cell.surface_c[i][1],
                                                     self.thread.rep[0].cell.surface_c[i][2]))
                base = self.thread.rep[0].cell.surface_e.size
                for replica in self.thread.rep:
                    f.seek(f.tell())
                    f.write('%s %.6f %.6f %.6f\n' % (replica.cell.e_best[base],
                                                     replica.cell.c_best[base][0],
                                                     replica.cell.c_best[base][1],
                                                     replica.cell.c_best[base][2]))
                for replica in self.thread.rep:
                    for i in range(1, replica.cell.local_size):
                        f.seek(f.tell())
                        f.write('%s %.6f %.6f %.6f\n' % (replica.cell.e_best[base + i],
                                                         replica.cell.c_best[base + i][0],
                                                         replica.cell.c_best[base + i][1],
                                                         replica.cell.c_best[base + i][2]))
                f.seek(f.tell())
                f.write('\n')
                f.seek(f.tell())
                f.write('[Best local model]\n')
                f.seek(f.tell())
                f.write('%s %.6f %.6f %.6f\n' % (replica.cell.center_e, 0, 0, 0))
                for replica in self.thread.rep:
                    for i in range(1, replica.cell.local_size):
                        temp = replica.cell.c_best[base + i] - replica.cell.c_best[base]
                        f.seek(f.tell())
                        f.write('%s %.6f %.6f %.6f\n' % (replica.cell.e_best[base + i], temp[0], temp[1], temp[2]))
                    for i in range(replica.cell.surface_e.size):
                        temp = replica.cell.c_best[i] - replica.cell.c_best[base]
                        if sqrt((temp ** 2).sum()) < self.thread.local_range:
                            f.seek(f.tell())
                            f.write('%s %.6f %.6f %.6f\n' % (replica.cell.surface_e[i], temp[0], temp[1], temp[2]))
            else:
                f.seek(f.tell())
                f.write('[Best local model]\n')
                f.seek(f.tell())
                f.write('%s %.6f %.6f %.6f\n' % (replica.cell.center_e, 0, 0, 0))
                for replica in self.thread.rep:
                    for i in range(1, replica.cell.element.size):
                        f.seek(f.tell())
                        f.write('%s %.6f %.6f %.6f\n' % (replica.cell.e_best[i],
                                                         replica.cell.c_best[i][0],
                                                         replica.cell.c_best[i][1],
                                                         replica.cell.c_best[i][2]))
            f.truncate()

    def closeEvent(self, event):
        if self.thread.flag:
            self.cal_end()
        event.accept()

    def pol_info(self, chik, factor, chiex=np.array([])):
        if len(self.polx.plotItem.items) == 1:
            if not self.thread.fit_space == 'r':
                self.line_chi = np.array([line(self.thread.exp[_].k, chik[_], c='red', width=3
                                               ) for _ in range(self.thread.exp.size)])
                if self.thread.fit_space == 'x':
                    self.line_chi = np.append(self.line_chi, np.array([
                        line(self.thread.exp[_].r_cut, chiex[_], c='red', width=3)
                        for _ in range(self.thread.exp.size)]))
            else:
                self.line_chi = np.array([line(self.thread.exp[_].r_cut, chik[_], c='red', width=3
                                               ) for _ in range(self.thread.exp.size)])
            for i in range(self.thread.exp.size):
                self.plot[i].addItem(self.line_chi[i])
                if self.thread.fit_space == 'x':
                    self.plotex[i].addItem(self.line_chi[i + self.thread.exp.size])
        else:
            if not self.thread.fit_space == 'r':
                for i in range(self.thread.exp.size):
                    self.line_chi[i].setData(x=self.thread.exp[i].k, y=chik[i])
                if self.thread.fit_space == 'x':
                    for i in range(self.thread.exp.size):
                        self.line_chi[i + self.thread.exp.size].setData(x=self.thread.exp[i].r_cut, y=chiex[i])
            else:
                for i in range(self.thread.exp.size):
                    self.line_chi[i].setData(x=self.thread.exp[i].r_cut, y=chik[i])

        for i in range(self.r_aver.size):
            dist = np.array([replica.cell.distance[i + 1] for replica in self.thread.rep])
            self.r_aver[i] = dist.mean()
            self.c2[i] = dist.var()
            self.c3[i] = ((dist - self.r_aver[i]) ** 3).mean()
        self.statistic_display()

        self.current_lcd_p1.display(factor[0])
        self.current_lcd_p2.display(factor[1])
        if self.thread.exp.size > 2:
            self.current_lcd_p3.display(factor[2])

    def tau_t_change(self):
        signal = self.sender()
        if signal.hasFocus():
            if signal.objectName() == 'tau_t_value_box' or signal.objectName() == 'tau_t_degree_box':
                self.thread.tau_t = round(self.tau_t_value_box.value() * (10 ** (-self.tau_t_degree_box.value())),
                                          self.tau_t_degree_box.value() + 1)
                if not self.ratioCheckBox.checkState() == 0:
                    self.thread.tau_i = self.thread.tau_t * self.thread.tau_ratio
                    self.thread.tau_i = round(self.thread.tau_i, int(('%.1e' % self.thread.tau_i).split('e-')[1]) + 1)

                    self.tau_i_value_box.setValue(float(('%.1e' % self.thread.tau_i).split('e-')[0]))
                    self.tau_i_degree_box.setValue(int(('%.1e' % self.thread.tau_i).split('e-')[1]))
                print(self.thread.tau_t)

    def tau_i_change(self):
        signal = self.sender()
        if signal.hasFocus():
            if signal.objectName() == 'tau_i_value_box' or signal.objectName() == 'tau_i_degree_box':
                self.thread.tau_i = round(self.tau_i_value_box.value() * (10 ** (-self.tau_i_degree_box.value())),
                                          self.tau_i_degree_box.value() + 1)
                if not self.ratioCheckBox.checkState() == 0:
                    self.thread.tau_t = self.thread.tau_i / self.thread.tau_ratio
                    self.thread.tau_t = round(self.thread.tau_t, int(('%.1e' % self.thread.tau_t).split('e-')[1]) + 1)

                    self.tau_t_value_box.setValue(float(('%.1e' % self.thread.tau_t).split('e-')[0]))
                    self.tau_t_degree_box.setValue(int(('%.1e' % self.thread.tau_t).split('e-')[1]))
                print(self.thread.tau_i)

    def tau_ratio_lock(self):
        if not self.ratioCheckBox.checkState() == 0:
            self.thread.tau_ratio = self.thread.tau_i / self.thread.tau_t
            print('ratio: ', self.thread.tau_ratio)

    def tau_init(self, tau_t, tau_i):
        self.tau_t_value_box.setValue(float(('%.1e' % tau_t).split('e-')[0]))
        try:
            self.tau_t_degree_box.setValue(int(('%.1e' % tau_t).split('e-')[1]))
        except IndexError:
            self.tau_t_degree_box.setValue(0)
        self.tau_i_value_box.setValue(float(('%.1e' % tau_i).split('e-')[0]))
        try:
            self.tau_i_degree_box.setValue(int(('%.1e' % tau_i).split('e-')[1]))
        except IndexError:
            self.tau_i_degree_box.setValue(0)

    def tca_event(self):
        if self.tcaCheckBox.checkState() == 0:
            for replica in self.thread.rep:
                replica.cell.adsorption = False
        else:
            for replica in self.thread.rep:
                replica.cell.adsorption = True

    def plot_init(self):
        color = ['blue', 'yellow', 'red', 'green']
        if not self.thread.surface == '':
            init_3dplot(self.model, grid=False, view=40, title='model')  # set initial view distance by "view"
            if self.thread.surface == 'TiO2':
                plot_TiO2(self.thread.rep[0].cell.surface_c, self.thread.rep[0].cell.surface_e, self.model)
            elif self.thread.surface == 'Al2O3':
                plot_Al2O3(self.thread.rep[0].cell.surface_c, self.thread.rep[0].cell.surface_e, self.model)
            surface = self.thread.rep[0].cell.surface_e.size
            for replica in self.thread.rep:
                for i in range(replica.cell.local_size):
                    self.scatter = np.append(self.scatter, scatter(replica.cell.coordinate_whole[surface + i][0],
                                                                   replica.cell.coordinate_whole[surface + i][1],
                                                                   replica.cell.coordinate_whole[surface + i][2],
                                                                   c=color[i], scale=0.3))
                    self.model.addItem(self.scatter[-1])
            self.scatter = self.scatter.reshape((self.thread.rep_size, self.thread.init_element.size))
        else:
            init_3dplot(self.model, grid=True, view=15, angle=[10, 30], title='model')
            # set initial view distance by "view", angle=[azimuth, elevation]
            self.model.addItem(scatter(0, 0, 0, c=color[0], scale=0.6))
            self.model.addItem(line([-5, 5], [0, 0], [0, 0], c='red', width=3))
            self.model.addItem(line([0, 0], [-5, 5], [0, 0], c='green', width=3))
            self.model.addItem(line([0, 0], [0, 0], [-5, 5], c='blue', width=3))

            for replica in self.thread.rep:
                for i in range(1, replica.cell.local_size):
                    self.scatter = np.append(self.scatter, scatter(replica.cell.coordinate[i][0],
                                                                   replica.cell.coordinate[i][1],
                                                                   replica.cell.coordinate[i][2],
                                                                   c=color[i], scale=0.3))
                    self.model.addItem(self.scatter[-1])
            self.scatter = self.scatter.reshape((self.thread.rep_size, self.thread.init_element.size - 1))
        self.g3dLayout.addWidget(self.model)

    def plot_3D_substrate(self, where, target, position):
        self.scatter[where][target].translate(position[0], position[1], position[2])

    def plot_3D_center(self, where, target, position):
        self.scatter[where][target-1].translate(position[0], position[1], position[2])

    def plot_3D_event(self, where, target, position):
        if not self.thread.surface == '':
            self.plot_3D_substrate(where, target, position)
        else:
            self.plot_3D_center(where, target, position)

    def read_init_pos(self):
        file_name = QFileDialog.getOpenFileName(self, 'select ini file...', self.thread.simulation_name)
        if file_name[0] == '':
            return
        self.statusbar.showMessage('Reading', 0)
        with open(file_name[0], 'r') as finit:
            temp = finit.readline().split(",")
            try:
                self.thread.init_pos[0] = [float(temp[-3].split(':')[1]), int(temp[-2]), int(temp[-1])]
            except IndexError or ValueError:
                self.statusbar.showMessage('failed to read', 3000)
                return
            temp = finit.readline().split(",")
            try:
                self.thread.init_pos[1] = [float(temp[-3].split(':')[1]), int(temp[-2]), int(temp[-1])]
            except IndexError or ValueError:
                self.statusbar.showMessage('failed to read', 3000)
                return
        self.statusbar.showMessage('successfully read', 3000)

    def save_3d_menu(self, pos):
        target = self.sender()
        menu = QMenu()
        action = menu.addAction('save')
        action.triggered.connect(lambda: self.save_3d_action(target))
        menu.exec_(QCursor.pos())

    def save_3d_action(self, target):
        address = self.thread.folder if not self.thread.folder == '' else os.getcwd()
        name = '/model.jpg'
        size = (1920, 1080) if self.thread.material == 'CuAlO' else (1080, 1080)
        name = QFileDialog.getSaveFileName(self, 'select path...', address + name, 'jpg(*.jpg)')
        if name[0] == '':
            return
        image = target.renderToArray(size)
        pg.makeQImage(image.transpose(1, 0, 2)).save(name[0])

    def stepLCD_change(self, value):
        self.step_lcd.display(value)

    def statistic_display(self):
        if self.r_aver.size > 0:
            self.r_O_lcd.display(self.r_aver[0])
            self.c2_O_lcd.display(self.c2[0])
            self.c3_O_lcd.display(self.c3[0])
            if self.r_aver.size > 1:
                self.r_S_lcd.display(self.r_aver[1])
                self.c2_S_lcd.display(self.c2[1])
                self.c3_S_lcd.display(self.c3[1])

    def new_event(self):
        self.thread.init()
        self.actioninp.setEnabled(False)

    def time_display(self, time):
        self.time_lcd.display(time)

    def best_r_display(self, rb):
        self.best_lcd.display(rb)

    def current_r_display(self, rn):
        self.current_lcd.display(rn)

    def warning_window(self, massage):
        QMessageBox.critical(self, 'Error', massage)

    def close_signal(self, a):
        self.close()

if __name__ == '__main__':
    app = QApplication(argv)
    main = MainWindow()
    main.show()
    exit(app.exec_())

