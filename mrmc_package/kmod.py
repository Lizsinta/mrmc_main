from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d
import numpy as np
from math import pi, sqrt


# birn_fft
# intp1D
# win_function
# k_range
# deltaE_shift

def norm_fft(array=np.array([]), size=1):
    # FFT,normalization, cutting R range
    result = fft(array, 2048)
    return result[range(size)] / 1024

def intp1D(k_old=np.array([]), k_new=np.array([]), array=np.array([])):
    # interpolate data
    f_array = interp1d(k_old, array)
    return f_array(k_new)


def win_function(k_range=0, name='hanning', win_width=10):
    # Choose windows
    width = int(k_range / win_width)
    if name == 'hanning':
        t_window = np.array([0.5 - 0.5 * np.cos(pi * n / (width - 1)) for n in range(width)])
        t_window = np.append(t_window, np.ones(k_range - 2 * width))
        t_window = np.append(t_window, np.array([0.5 + 0.5 * np.cos(pi * n / (width - 1)) for n in range(width)]))
    elif name == 'hamming':
        t_window = np.array([0.5 - 0.46 * np.cos(pi * n / (width - 1)) for n in range(width)])
        t_window = np.append(t_window, np.ones(k_range - 2 * width))
        t_window = np.append(t_window, np.array([0.5 + 0.46 * np.cos(pi * n / (width - 1)) for n in range(width)]))
    elif name == 'rectangular':
        t_window = np.ones(k_range)
    elif name == 'hanningl':
        t_window = np.array([0.5 - 0.5 * np.cos(pi * n / (width - 1)) for n in range(width)])
        t_window = np.append(t_window, np.ones(k_range - width))
    return t_window


def k_range(k=np.array([]), xi=np.array([]), start=3.0, end=12.0, padding=True, win_name='hanning', get_k=True):
    # choose k range,applying window fuctions
    k_start = np.where(k > start)[0][0] if not start == k[0] else 0
    try:
        k_end = np.where(k > end)[0][0]
    except IndexError:
        k_end = k.size
    window = win_function(k_end - k_start, name=win_name)
    # xi_t = xi[k_start:k_end]
    xi_t = np.multiply(xi[k_start:k_end], window)
    if padding:
        return np.concatenate((np.zeros(k_start), xi_t, np.zeros(k.size - k_end)))
    else:
        return k[k_start:k_end], xi_t


def deltaE_shift(k=np.array([]), xi=np.array([]), dE=0.0, factor=0.2624683854935682):
    k_shift = np.zeros(k.size)
    shift = factor * dE
    padding = np.array([])
    for _ in range(k.size):
        temp = k[_] ** 2 + shift
        if temp > 0:
            k_shift[_] = sqrt(temp)
    if shift < 0:
        k_end = np.where(k < sqrt(k[-1] ** 2 + shift))[0][-1]
        xi_align = intp1D(k_shift, k[:k_end], xi)
        padding = np.concatenate((xi_align, np.zeros(k.size - k_end)))
    elif shift == 0:
        padding = xi
    elif shift > 0:
        k_start = np.where(k > k_shift[0])[0][0]
        xi_align = intp1D(k_shift, k[k_start:], xi)
        padding = np.concatenate((np.zeros(k_start), xi_align))
    return padding


def back_k_space(array=np.array([]), r=np.array([]), k_size=401, r_head=0.0, r_tail=6.0):
    chi_ft = fft(array, 2048)[:r.size]
    chi_ft_cut = k_range(r, chi_ft, r_head, r_tail)
    return ifft(chi_ft_cut, 2048)[:k_size].real * 2, np.abs(chi_ft_cut) / 1024


def fft_cut(array=np.array([]), r=np.array([]), r_head=0.0, r_tail=6.0):
    return np.abs(k_range(r, fft(array, 2048)[:r.size], r_head, r_tail)) / 1024


