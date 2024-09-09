import os

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from pyqtgraph.Qt import QtCore
import numpy as np
from math import sqrt, acos, pi
import matplotlib
from sys import flags
from sys import argv, exit

from mrmc_package import init_3dplot, scatter, line, plot_TiO2, cylinder

file = r'D:\simulation\tca3x\model.dat'

# framework: windows->large widget->layout->all kinds of graph widgets-> item
app = QApplication([])
mw = QMainWindow()
mw.resize(1280, 720)
mw.setWindowTitle('3D Viewer')
mw.show()
cw = QWidget()
mw.setCentralWidget(cw)
l = QHBoxLayout()
cw.setLayout(l)

# 3d graph
g3d = gl.GLViewWidget()
init_3dplot(g3d, background=[255., 255., 255.], alpha=1.0, view=40, title='3D viewer', grid=False, ortho=True)
l.addWidget(g3d)



color = ['blue', 'red', 'yellow', 'green', 'orange', 'purple', 'cyan']
coor = []
ele = []
with open(file, 'r') as f:
    while True:
        posi = f.tell()
        lines = f.readline()
        if lines.find('[initial surface model]') == -1:
            continue
        break
    while True:
        lines = f.readline()
        if not lines or lines.isspace():
            break
        temp = lines.split()
        if temp[0].isalpha():
            coor.append(np.array([float(temp[1]), float(temp[2]), float(temp[3])]))
            ele.append(temp[0])
        else:
            coor.append(np.array([float(temp[0]), float(temp[1]), float(temp[2])]))
            ele.append(temp[3])
coor = np.asarray(coor)
ele = np.asarray(ele)
surface_i = np.where((ele == 'Ti') | (ele == 'O'))[0]
surface_c = coor[surface_i]
surface_e = ele[surface_i]
cu = coor[np.where(ele == 'Cu')[0]]
s = coor[np.where(ele == 'S')[0]]

plot_TiO2(surface_c, surface_e, g3d)

g3d.addItem(line([-5, 5], [0, 0], [0, 0], c='red', width=3))
g3d.addItem(line([0, 0], [-5, 5], [0, 0], c='green', width=3))
g3d.addItem(line([0, 0], [0, 0], [-5, 5], c='blue', width=3))
for i in range(cu.shape[0]):
    g3d.addItem(scatter(cu[i][0], cu[i][1], cu[i][2], c='blue', scale=0.4))
    g3d.addItem(scatter(s[i][0], s[i][1], s[i][2], c='yellow', scale=0.4))
    g3d.addItem(cylinder([cu[i][0], s[i][0]], [cu[i][1], s[i][1]], [cu[i][2], s[i][2]], c='k', width=0.1))
    dist = np.array([sqrt(((surface_c[j] - cu[i])**2).sum()) for j in range(surface_e.size)])
    adsorb = np.argmin(dist)
    g3d.addItem(cylinder([cu[i][0], surface_c[adsorb][0]], [cu[i][1], surface_c[adsorb][1]], [cu[i][2], surface_c[adsorb][2]], c='k', width=0.1))



if (flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QApplication.instance().exec_()
