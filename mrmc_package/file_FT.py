import numpy as np
import os.path
from os import makedirs
from time import strftime, localtime


def Read_phase(filename='feff_Pt.dat'):
    # reading FEFF data
    f_phase = open(filename, 'r')
    k = np.array([])
    amp = np.array([])
    phase = np.array([])
    lamb = np.array([])
    while True:
        lines = f_phase.readline()
        if not lines or not lines.find('real[2*phc]') == -1:
            break
    while True:
        data = f_phase.readline()
        if not data:
            break
        temp = [i for i in data.split()]
        k = np.append(k, float(temp[0]))
        amp = np.append(amp, float(temp[2]))
        phase = np.append(phase, complex(temp[1]) + complex(temp[3]))
        lamb = np.append(lamb, float(temp[5]))
    f_phase.close()
    return k, amp, phase, lamb


def Read_exp(filename='703K_diff_H2-dry.rex'):
    # reading oscillation data
    f_exp = open(filename, 'r')
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
        xi = np.append(xi, complex(temp[1]))
    f_exp.close()
    return k, xi


'''def Read_bk(filename='703K_diff_H2-dry.rex', max=20.0):
    from larch.xafs import autobk
    from larch.io import read_ascii
    # reading oscillation data
    f_exp = open(filename, 'r')
    f_bk = open('temp_bk.dat', 'w')
    f_bk.write('#   energy        xmu\n')
    while True:
        lines = f_exp.readline(11)
        if not lines or not lines.find('[EX_BEGIN]') == -1:
            break
    while True:
        data = f_exp.readline()
        if data.isspace():
            break
        f_bk.write(data)
    f_exp.close()
    f_bk.close()
    x_data = read_ascii('temp_bk.dat', labels='energy xmu')
    autobk(x_data.energy, x_data.xmu, group=x_data, rbkg=0.9, kmax=max)
    k = np.asarray(x_data.k)
    xi = np.multiply(np.asarray(x_data.chi), k ** 3)
    return k, xi'''


def Read_common(target, index='[Phase Corrected FT]'):
    # read format data
    data1 = np.array([])
    data2 = np.array([])
    while True:
        position = target.tell()
        lines = target.readline()
        if not lines.isspace():
            if not lines.find(index) == -1:
                break
            else:
                target.seek(position)
                return data1, data2
    while True:
        data = target.readline()
        if data.isspace():
            break
        temp = [i for i in data.split()]
        data1 = np.append(data1, float(temp[0]))
        data2 = np.append(data2, complex(j_position(temp[1])))
    return data1, data2


def j_position(x=''):
    half1, half2 = x.split('j')
    return half1 + half2 + 'j'


'''def Read_result(filename='f_result.txt'):
    # reading result
    f_exp = open(filename, 'r')
    head = np.array([])
    tail = np.array([])
    E = np.array([])
    r = np.array([])
    t_head = np.array([])
    t_tail = np.array([])
    t_E = np.array([])
    t_r = np.array([])
    r_aver = np.array([])
    E_aver = np.array([])
    while True:
        lines = f_exp.readline()
        if not lines or not lines.find('k_head') == -1:
            break
    while True:
        data = f_exp.readline()
        if not data.find('[END]') == -1:
            break
        if data.isspace():
            if head.size == 0:
                head = t_head[:t_head.size]
                tail = t_tail[:t_tail.size]
                E = t_E[:t_E.size]
                r = t_r[:t_r.size]
            else:
                head = np.vstack((head, t_head))
                tail = np.vstack((tail, t_tail))
                E = np.vstack((E, t_E))
                r = np.vstack((r, t_r))
            t_head = np.array([])
            t_tail = np.array([])
            t_E = np.array([])
            t_r = np.array([])
        else:
            temp = [i for i in data.split()]
            t_head = np.append(t_head, float(temp[0]))
            t_tail = np.append(t_tail, float(temp[1]))
            t_E = np.append(t_E, float(temp[2]))
            t_r = np.append(t_r, float(temp[3]))
            if not float(temp[3]) == 1.8:
                r_aver = np.append(r_aver, float(temp[3]))
                E_aver = np.append(E_aver, float(temp[2]))
    f_exp.close()
    print('Average distance: %f' % np.mean(r_aver))
    print('Average detalE: %f' % np.mean(E_aver))
    return np.transpose(head), np.transpose(tail), E, r'''


def file_creat(name='f_result.txt', backf=True):
    index = int(1)
    folder = os.getcwd() + '\\result'
    if not os.path.exists(folder):
        makedirs(folder)
    name_abs = folder + '\\' + name
    while True:
        if os.path.isfile(name_abs):
            name_new = name_abs[:name_abs.rfind('.')] + str(index) + name_abs[name_abs.rfind('.'):]
            if os.path.isfile(name_new):
                index += 1
            else:
                break
        else:
            name_new = name_abs
            break
    if backf:
        file = open(name_new, 'w')
        return file
    else:
        return name_new


def folder_create(name):
    index = int(1)
    while True:
        if os.path.isdir(name):
            name_new = name + str(index)
            if os.path.isdir(name_new):
                index += 1
            else:
                break
        else:
            name_new = name
            break

    makedirs(name_new)
    return name_new


def head_info(file, xi_name1='null', xi_name2='null', ps_name='null', k_range=[0, 0, 0], r_range=[0, 0]):
    file.write('--Phase Corrected Fourier Transform--     Version 0.1\n')
    file.write('Oscillation file name:%s\n' % xi_name1)
    file.write('Subtracting oscillation file name:%s\n' % xi_name2)
    file.write('Phase shift file name:%s\n' % ps_name)
    file.write('Date:%s\n' % strftime("%Y-%m-%d %H:%M:%S", localtime()))
    file.write('K_range:%.2f~%.2f , interval:%.2f\n' % (k_range[0], k_range[1], k_range[2]))
    file.write('R_range:0~%.3f , interval:%.3f\n' % (r_range[0], r_range[1]))


def d_write(file, name='null', x=np.array([]), y=np.array([]), x_bit=2, y_bit=8):
    file.write(f'\n[{name}]\n')
    if y.dtype == complex:
        for i in range(x.size):
            if y[i].imag >= 0:
                file.write('%.*f    %.*f+j%.*f\n' % (x_bit, x[i], y_bit, y[i].real, y_bit, y[i].imag))
            elif y[i].imag < 0:
                file.write('%.*f    %.*f-j%.*f\n' % (x_bit, x[i], y_bit, y[i].real, y_bit, -y[i].imag))
    else:
        for i in range(x.size):
            file.write('%.*f    %.*f\n' % (x_bit, x[i], y_bit, y[i]))
    file.write('\n\n\n')
