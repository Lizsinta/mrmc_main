from math import atan, pi, acos, sqrt
import os

import numpy as np
from mrmc_package import get_distance

fin = r'C:\Users\lizsi\Desktop\paper\dft2\origin\right_tca_reverse2.txt'
name = fin.split('\\')[-1].split('.')[0]
farray = np.loadtxt(fin, dtype=str).T

ele = farray[0]
coor = farray[1:].astype(float).T

sfi = np.where(coor[:-13, 2] > 11.5)[0]
sfe = ele[sfi]
sfc = coor[sfi]

tcac = coor[-11:]
tcae = ele[-11:]
c = np.argmin(get_distance(tcac - coor[-13]))

loi = np.where(get_distance(sfc - coor[-13]) < 3)[0]
loc = np.vstack((coor[-13:-11], sfc[loi]))
loe = np.append(ele[-13:-11], sfe[loi])


center = sfc[np.argmin(get_distance(sfc - np.mean(sfc, axis=0)))]
sfc = sfc - center

folder =r'C:\Users\lizsi\Desktop\paper\dft2'
os.makedirs(folder) if not os.path.exists(folder) else None
os.makedirs(folder + r'\surface') if not os.path.exists(folder + r'\surface') else None
os.makedirs(folder + r'\local') if not os.path.exists(folder + r'\local') else None
with open(folder + r'\surface\%s.xyz'%name, 'w') as f:
    f.write('TiO2 of %s\n'%name)
    f.write('%d\n'%sfe.size)
    for i in range(sfe.size):
        f.write('%s %.9f %.9f %.9f\n' % (sfe[i], sfc[i][0], sfc[i][1], sfc[i][2]))

with open(folder + r'\surface\%s.txt' % name, 'w') as f:
    f.write('%s %.9f %.9f %.9f\n' % (ele[-13], coor[-13][0] - center[0], coor[-13][1] - center[1], coor[-13][2] - center[2]))
    f.write('%s %.9f %.9f %.9f\n' % (ele[-12], coor[-12][0] - center[0], coor[-12][1] - center[1], coor[-12][2] - center[2]))
    '''f.write('%s %.9f %.9f %.9f\n' % (tcae[c], tcac[c][0] - coor[-13][0],
                                     tcac[c][1] - coor[-13][1],
                                     tcac[c][2] - coor[-13][2]))'''

with open(folder + r'\local\%s.txt' % name, 'w') as f:
    for i in range(loe.size):
        f.write('%s %.9f %.9f %.9f %.3f\n' % (loe[i], loc[i][0] - coor[-13][0],
                                              loc[i][1] - coor[-13][1], loc[i][2] - coor[-13][2],
                                              sqrt(((loc[i] - coor[-13])**2).sum())))