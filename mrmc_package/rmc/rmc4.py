from mrmc_package import ATOMS, metropolis, TABLE_POL, get_distance

from math import sqrt
from random import randrange
import os
import numpy as np

class RMC4:
    def __init__(self, index, exp, sig2, energy, s02, data_base, path, init_pos=np.array([]), init_element=np.array([]),
                 spherical=True, random=True, local_range=3.0, surface='', surface_range=np.array([]),
                 step=np.array([]), step_range=np.array([]), ini_flag=True, ms=False, weight=3, trial=np.array([])):
        self.index = index
        self.data_base = data_base
        self.path = path
        self.sig2 = sig2
        self.s02 = s02
        self.energy = energy
        self.exp = exp
        self.moved_atom = 0
        self.debug = False #if not index == 0 else True
        self.ms = ms
        self.weight = weight
        self.trial = trial[0]

        self.cell = ATOMS(database=data_base, file=path, pos=init_pos, element=init_element, spherical=spherical,
                          random=random, local_range=local_range, surface=surface, step=step, step_range=step_range,
                          crate_flag=ini_flag, surface_range=surface_range, trial=trial[1])
        self.table = np.zeros(self.exp.size, dtype=TABLE_POL)
        self.r_factor_i = np.zeros(self.exp.size)
        self.r_factor_t = 0

    def table_init(self):
        pol = np.arange(3) if self.exp.size == 3 else (np.arange(2) + 2)
        self.table = np.array([TABLE_POL(self.exp[i].k_start, self.exp[i].k_end, self.exp[i].r_start,
                                         self.exp[i].r_end, self.sig2, self.energy, self.s02, self.exp[i].k0,
                                         self.cell.coordinate.copy(), self.cell.element.copy(),
                                         self.data_base, i, ms_en=self.ms, weight=self.weight) for i in pol])
        self.r_factor_i = np.array([self.exp[_].r_factor(self.table[_].chi) for _ in range(self.exp.size)])
        self.r_factor_t = self.r_factor_i.sum()

    def walk(self, tau):
        trials = 50
        while trials > 0:
            if not self.cell.surface == '':
                self.moved_atom = randrange(self.cell.local_size)
            if not self.cell.surface == '' and self.moved_atom == 0:
                failure = self.cell.moving_center()
                if failure:
                    self.cell.cw_temp = self.cell.coordinate_whole.copy()
                    self.cell.c_temp = self.cell.coordinate.copy()
                    self.cell.e_temp = self.cell.element.copy()
                    self.cell.distance = get_distance(self.cell.coordinate)
                    trials = 0
                    continue
                for pol in self.table:
                    pol.moving_group(self.cell.c_temp.copy(), self.cell.distance.copy(),
                                     self.cell.e_temp.copy(), self.debug)
            else:
                if self.debug:
                    print('start move')
                target, failure = self.cell.moving_spherical(None)
                if self.debug:
                    print('end move', target, failure, self.cell.distance[target])
                if failure:
                    self.cell.c_temp = self.cell.coordinate.copy()
                    self.cell.distance[target] = sqrt((self.cell.c_temp[target] ** 2).sum())
                    trials = 0
                    continue
                for pol in self.table:
                    pol.moving(target, self.cell.c_temp[target], self.debug)
                if not self.cell.surface == '':
                    self.cell.cw_temp[target - self.cell.local_size] += (self.cell.c_temp[target]
                                                                                 - self.cell.coordinate[target])
                else:
                    self.moved_atom = target
            if not tau == 1:
                r_factor_i = np.array([self.exp[_].r_factor(self.table[_].chi) for _ in range(self.exp.size)])
                r_factor_new = r_factor_i.sum()
                if np.array([metropolis(self.r_factor_i[_], r_factor_i[_], tau) for _ in range(self.exp.size)]).all():
                #if metropolis(self.r_factor_t, r_factor_new, tau):
                    if self.debug:
                        print('inner accept', self.moved_atom)
                    self.r_factor_i = r_factor_i.copy()
                    self.r_factor_t = r_factor_new
                    break
                else:
                    if self.debug:
                        print('inner reject', self.moved_atom)
                    self.reject()
                    trials -= 1
                    continue
            else:
                break
        return trials

    def accept(self):
        if self.debug:
            print('accept', self.moved_atom)
        if not self.cell.surface == '':
            self.cell.coordinate_whole = self.cell.cw_temp.copy()
            self.cell.element = self.cell.e_temp.copy()
        self.cell.coordinate = self.cell.c_temp.copy()

    def reject(self):
        if self.debug:
            print('reject', self.moved_atom)
        if not self.cell.surface == '':
            self.cell.cw_temp = self.cell.coordinate_whole.copy()
            self.cell.e_temp = self.cell.element.copy()
        self.cell.c_temp = self.cell.coordinate.copy()
        if self.moved_atom == 0:
            self.cell.distance = get_distance(self.cell.coordinate)
            for pol in self.table:
                pol.recover_group(self.cell.coordinate.copy(), self.cell.distance.copy(),
                                  self.cell.element.copy(), self.debug)
        else:
            self.cell.distance[self.moved_atom] = sqrt((self.cell.c_temp[self.moved_atom] ** 2).sum())
            for pol in self.table:
                pol.recover(self.moved_atom, self.cell.coordinate[self.moved_atom], self.debug)

    def write_result(self, r1, r2, path=''):
        if len(path) == 0:
            path = self.path + r'\result'
        with open(path + r'\result%d.txt' % self.index, 'w') as best:
            coor = self.cell.coordinate_whole if not self.cell.surface == '' else self.cell.coordinate
            ele = self.cell.element_whole if not self.cell.surface == '' else self.cell.element
            center = self.cell.surface_e.size if not self.cell.surface == '' else 0
            if not r1 == 0:
                distance = get_distance(self.cell.c_best - self.cell.c_best[center])
                best.write('Best R-factor: %f\n' % r1)
                for i in range(distance.size):
                    best.write('   %.5f     %.5f     %.5f    %d  %s1              %.5f\n'
                               % (self.cell.c_best[i][0], self.cell.c_best[i][1], self.cell.c_best[i][2],
                                  np.where(self.cell.symbol == ele[i])[0][0], ele[i], distance[i]))
            distance = get_distance(coor - coor[center])
            best.write('\nfinal R-factor: %f\n' % r2)
            for i in range(distance.size):
                best.write('   %.5f     %.5f     %.5f    %d  %s1              %.5f\n'
                           % (coor[i][0], coor[i][1], coor[i][2], np.where(self.cell.symbol == ele[i])[0][0],
                              ele[i], distance[i]))

    def folder(self):
        name = self.path + r'\result\replica%d' % self.index
        if os.path.exists(name):
            return name
        os.makedirs(name)
        return name

    def read_result(self):
        self.cell.read(self.index)
        pol = np.arange(3) if self.exp.size == 3 else (np.arange(2) + 2)
        self.table = np.array([TABLE_POL(self.exp[i].k_start, self.exp[i].k_end, self.exp[i].r_start,
                                         self.exp[i].r_end, self.sig2, self.energy, self.s02, self.exp[i].k0,
                                         self.cell.coordinate.copy(), self.cell.element.copy(),
                                         self.data_base, i, ms_en=self.ms, weight=self.weight) for i in pol])
        print('table set up')
        self.r_factor_i = np.array([self.exp[_].r_factor(self.table[_].chi) for _ in range(self.exp.size)])
        self.r_factor_t = self.r_factor_i.sum()
