import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from pyqtgraph.Qt import QtCore
import numpy as np
from math import sqrt, acos, pi
import matplotlib
from sys import flags
from sys import argv, exit

from mrmc_package import init_3dplot, scatter, line, cylinder, plot_TiO2

file = r'C:\Users\lizsi\Desktop\paper\jpc\20230605\right.dat'

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


color = {'Cu': 'blue', 'O': 'red', 'Ti': 'purple', 'S': 'yellow', 'C': 'pink', 'H': 'gray'}
size = {'Cu': 0.6, 'O': 0.6, 'Ti': 0.4, 'S': 0.6, 'C': 0.4, 'H': 0.3}
endix = {'Cu': 0, 'O': 1, 'Ti': 2, 'S': 3, 'C': 4, 'H': 5}
bond = np.array([[0, 2.3, 2.7, 2.7, 2.3, 2],
                 [2.3, 0, 2.5, 2.5, 2, 2],
                 [2.7, 2.5, 0, 2.5, 2.5, 2],
                 [2.7, 2.5, 2.5, 0, 2.5, 2],
                 [2.3, 2, 2, 2, 2, 1.5],
                 [2, 2, 2, 2, 1.5, 0]])
'''coor = []
ele = []
with open(file, 'r') as f:
    while True:
        posi = f.tell()
        lines = f.readline()
        if not lines.find('#') == -1 or len(lines.split()) < 4:
            continue
        break
    f.seek(posi)
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
coor -= coor[50]
#coor[-13:, 0] += 2.95
i = np.where((-5 < coor[:, 1]) & (coor[:, 1] < 8) & (-7 < coor[:, 0]) & (coor[:, 0] < 5) & (-0.5 < coor[:, 2]))[0]
ele = np.asarray(ele)[i]
coor = coor[i]'''
coor, ele = np.array([]), np.array([])
with open(r'D:\PycharmProjects\mrmc_main\TiO2.xyz', 'r') as f:
    f.readline()
    f.readline()
    while True:
        lines = f.readline()
        if not lines:
            break
        temp = lines.split()
        ele = np.append(ele, temp[0])
        coor = np.append(coor, np.array([float(temp[1]), float(temp[2]), float(temp[3])]))
coor = coor.reshape(-1, 3)

g3d.addItem(line([-5, 5], [0, 0], [0, 0], c='red', width=3))
g3d.addItem(line([0, 0], [-5, 5], [0, 0], c='green', width=3))
g3d.addItem(line([0, 0], [0, 0], [-5, 5], c='blue', width=3))
#g3d.addItem(line([coor[-13][0]-2, coor[-13][0]+2], [coor[-13][1], coor[-13][1]], [coor[-13][2], coor[-13][2]], c='red', width=3))
#g3d.addItem(line([coor[-13][0], coor[-13][0]], [coor[-13][1]-2, coor[-13][1]+2], [coor[-13][2], coor[-13][2]], c='green', width=3))
#g3d.addItem(line([coor[-13][0], coor[-13][0]], [coor[-13][1], coor[-13][1]], [coor[-13][2]-2, coor[-13][2]+2], c='blue', width=3))
for i in range(ele.size):
    c = color[ele[i]]
    if ele[i] == 'O' and coor[i][2] < 0.5:
        c = 'orange'
    g3d.addItem(scatter(coor[i][0], coor[i][1], coor[i][2], c=c, scale=size[ele[i]]))
    for j in range(i, ele.size):
        if not i == j and sqrt(((coor[i] - coor[j])**2).sum()) < bond[endix[ele[i]]][endix[ele[j]]]:
            midp = (coor[i] + coor[j])/2
            cj = color[ele[j]]
            if ele[j] == 'O' and coor[j][2] < 0.5:
                cj = 'orange'
            g3d.addItem(
                cylinder([coor[i][0], midp[0]], [coor[i][1], midp[1]], [coor[i][2], midp[2]],
                         c=c, alpha=1, width=0.2))
            g3d.addItem(
                cylinder([midp[0], coor[j][0]], [midp[1], coor[j][1]], [midp[2], coor[j][2]],
                         c=cj, alpha=1, width=0.2))

if (flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QApplication.instance().exec_()
