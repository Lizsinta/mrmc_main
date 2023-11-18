import os.path
from math import sqrt, pi, comb

import numpy as np
from larch.xafs import feffrunner
from time import perf_counter as timer
import matplotlib.pyplot as plt
from mrmc_package import k_range, back_k_space, get_distance

eletable = np.append(np.array(['None']), np.loadtxt(os.getcwd() + r'\element.dat', dtype='str', usecols=1))


class TABLE_LARCH:
    def __init__(self, k_head, k_tail, r_head, r_tail, sig2, dE, s02, k0, coor, element,
                 folder, polarization=-1, ms_en=False, weight=3):
        self.k_head = k_head
        self.k_tail = k_tail
        self.r_head = r_head
        self.r_tail = r_tail
        self.weight = weight
        self.k0 = np.array([])
        self.k = np.array([])
        self.chi = np.array([])
        self.chi_backup = self.chi.copy()
        self.r = np.arange(0, 6 + pi / 102.4, pi / 102.4)
        self.ft = np.array([])
        self.element = np.array([])
        self.atom = np.array([])
        self.atom_temp = np.array([])
        self.amount = 0
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.inp = folder + r'\feff.inp'
        self.dat = folder + r'\chi.dat'
        self.feffrunner = feffrunner(feffinp=self.inp, verbose=False)

        self.inp_form0 = np.array(['TITLE   absorb',
                                   'EDGE     K',
                                   'S02       s2',
                                   '*         pot    xsph  fms   paths genfmt ff2chi',
                                   'CONTROL   1    1    1     1      1      1',
                                   'PRINT     0      0      0      0      0     0',
                                   'RPATH   rpa',
                                   'NLEG   nleg',
                                   'CRITERIA     4.00    10',
                                   'SIG2 sig',
                                   'CORRECTION de 0.00',
                                   'POLARIZATION x y z',
                                   'POTENTIALS',
                                   '*   ipot   z [ label   l_scmt  l_fms  xnatph]'], dtype=str)
        self.inp_form = self.inp_form0.copy()
        self.change_flag = False
        pol = np.zeros(3)
        if polarization >= 0:
            pol[polarization] += 1
        self.header(coor, element, s02, dE, sig2, pol, nleg=(4 if ms_en else 1))

    def header(self, coor, ele, s02, dE, sig2, pol, rpath=6, nleg=4):
        self.element = ele.copy()
        self.atom = np.unique(ele)
        self.atom_temp = self.atom.copy()
        self.amount = ele.size
        dist = get_distance(coor)
        self.inp_form0[0] = self.inp_form0[0].replace('absorb', ele[0])
        self.inp_form0[2] = self.inp_form0[2].replace('s2', str(s02))
        self.inp_form0[6] = self.inp_form0[6].replace('rpa', str(rpath))
        self.inp_form0[7] = self.inp_form0[7].replace('nleg', str(nleg))
        self.inp_form0[9] = self.inp_form0[9].replace('sig', str(sig2))
        self.inp_form0[10] = self.inp_form0[10].replace('de', str(dE))
        self.inp_form0[11] = self.inp_form0[11].replace('x y z', '%d %d %d' % (pol[0], pol[1], pol[2]))
        for i in range(self.atom.size):
            self.inp_form0 = np.append(self.inp_form0, '%d %d %s -1 -1 0' % (
                i, np.where(eletable == self.atom[i])[0][0], self.atom[i]))
        self.inp_form0 = np.append(self.inp_form0, 'ATOMS')

        self.inp_form = self.inp_form0.copy()
        for i in range(self.amount):
            self.inp_form = np.append(self.inp_form, '%.5f %.5f %.5f %d %s1 %.5f' % (
                coor[i][0], coor[i][1], coor[i][2], np.where(self.atom == ele[i])[0][0], ele[i], dist[i]))
        self.inp_form = np.append(self.inp_form, 'END')
        self.run_feff()
        temp = np.loadtxt(self.dat, usecols=(0, 1)).T
        self.k0 = temp[0]
        self.k, chi_cut = k_range(self.k0, temp[1] * self.k0 ** self.weight, self.k_head, self.k_tail, padding=False)
        self.chi, self.ft = back_k_space(chi_cut, self.r, self.k.size, self.r_head, self.r_tail)

    def run_feff(self, pot=0):
        np.savetxt(self.inp, self.inp_form, fmt='%s', delimiter='\n')
        self.feffrunner.run()

    def moving(self, target, coor, debug=False):
        # start = timer()
        self.inp_form[self.inp_form0.size + target] = ('%.5f %.5f %.5f' % ( coor[0], coor[1], coor[2])
                                                       + self.inp_form[self.inp_form0.size + target][23:])
        self.run_feff()
        self.chi_backup = self.chi.copy()
        chi_cut = k_range(self.k0, np.loadtxt(self.dat, usecols=1) * self.k0 ** self.weight, self.k_head, self.k_tail,
                          padding=False, get_k=False)
        self.chi, self.ft = back_k_space(chi_cut, self.r, self.k.size, self.r_head, self.r_tail)
        # print('modify', timer() - start)

    def recover(self, target, coor, debug=False):
        self.inp_form[self.inp_form0.size + target] = ('%.5f %.5f %.5f' % (coor[0], coor[1], coor[2])
                                                       + self.inp_form[self.inp_form0.size + target][23:])
        self.chi = self.chi_backup.copy()

    def moving_group(self, coor, ele, debug=False):
        dist= get_distance(coor)
        if not np.unique(ele, return_counts=True)[0].size == np.unique(self.element, return_counts=True)[0].size:
            self.change_flag = True
        else:
            change = np.unique(ele, return_counts=True)[1] == np.unique(self.element, return_counts=True)[1]
            self.change_flag = not (change if type(change) == bool else change.all())
        if self.change_flag:
            self.element = ele.copy()
            self.amount = ele.size
            self.inp_coor_backup = self.inp_form[self.inp_form0.size:]
            self.atom_temp = np.unique(self.element)
            if not self.atom_temp.size == self.atom.size:
                self.potential_backup = self.inp_form0[15:]
                self.inp_form0 = self.inp_form0[:15]
                for i in range(1, self.atom_temp.size):
                    self.inp_form0 = np.append(self.inp_form0, '%d %d %s -1 -1 0' % (
                        i, np.where(eletable == self.atom_temp[i])[0][0], self.atom_temp[i]))
                self.inp_form0 = np.append(self.inp_form0, 'ATOMS')
        self.inp_form = self.inp_form0.copy()
        for i in range(self.amount):
            self.inp_form = np.append(self.inp_form, '%.5f %.5f %.5f %d %s1 %.5f' % (coor[i][0], coor[i][1], coor[i][2],
                np.where(self.atom_temp == self.element[i])[0][0], self.element[i], dist[i]))
        self.inp_form = np.append(self.inp_form, 'END')
        self.run_feff()
        self.chi_backup = self.chi.copy()
        chi_cut = k_range(self.k0, np.loadtxt(self.dat, usecols=1) * self.k0 ** self.weight, self.k_head, self.k_tail,
                          padding=False, get_k=False)
        self.chi, self.ft = back_k_space(chi_cut, self.r, self.k.size, self.r_head, self.r_tail)

    def recover_group(self, ele, debug=False):
        if self.change_flag:
            self.element = ele.copy()
            self.amount = ele.size
            if not self.atom_temp.size == np.unique(self.element).size:
                self.inp_form0 = np.concatenate((self.inp_form0[:15], self.potential_backup))
        self.inp_form = self.inp_form0.copy()
        self.inp_form = np.concatenate((self.inp_form, self.inp_coor_backup))
        self.chi = self.chi_backup.copy()

if __name__ == '__main__':
    from mrmc_package import get_distance
    cr = np.array([[   0.00000,     0.00000,     0.00000],
                     [1.84000,     0.00000,    0.00000],
                     [0.00000,   2.15000,    0.00000,],
                     [0.00000,    0.00000,   2.50000]])
    ele = np.array(['Cu', 'O', 'S', 'Ti'])
    table = TABLE_LARCH(3, 9, 1, 3, np.arange(3), inp=r'J:\Monte Carlo\larch\feff.inp')
    table.header(cr, ele, 1, 3, 0, 3, 4, np.arange(3))
    table.run_feff()
    plt.plot(table.k, table.chi)
    cr2 = np.array([[   0.00000,     0.00000,     0.00000],
                     [1.84000,     0.00000,    0.00000],
                     [0.00000,    0.00000,   2.50000]])
    table.moving_group(cr2,  np.array(['Cu', 'O', 'Ti']))
    #plt.plot(table.k, table.chi)
    table.recover_group(cr, ele)
    plt.plot(table.k, table.chi, linestyle=':')
    plt.show()


