import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from pyqtgraph.Qt import QtCore
import numpy as np
from math import sqrt, acos, pi
import matplotlib
from sys import flags
from sys import argv, exit

from mrmc_package import init_3dplot, scatter, line

file = r'J:\dft2\origin\right.txt'

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
init_3dplot(g3d, background=[255., 255., 255.], alpha=1.0, view=40, title='3D viewer')
l.addWidget(g3d)

color = ['blue', 'red', 'yellow', 'green', 'orange', 'purple', 'cyan']
coor = []
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
ele = np.asarray(ele)
symbol = np.unique(ele, return_inverse=True)[1]

g3d.addItem(line([-5, 5], [0, 0], [0, 0], c='red', width=3))
g3d.addItem(line([0, 0], [-5, 5], [0, 0], c='green', width=3))
g3d.addItem(line([0, 0], [0, 0], [-5, 5], c='blue', width=3))
for i in range(ele.size):
    g3d.addItem(scatter(coor[i][0], coor[i][1], coor[i][2], c=color[symbol[i]], scale=0.5))
if (flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QApplication.instance().exec_()
