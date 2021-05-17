import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

def filtercoeffs(fs, fl, fh):
    Ts = 1 / float(fs)
    wl = 2 * np.pi * fl
    wh = 2 * np.pi * fh
    a0 = 4 + 2 * Ts * (wl + wh) + wl * wh * Ts ** 2
    b0 = 2.0 * Ts * wh / a0
    b1 = 0
    b2 = -b0
    a1 = (-8 + 2 * wl * wh * Ts ** 2) / a0
    a2 = (4 - 2 * Ts * (wl + wh) + wl * wh * Ts ** 2) / a0

    coeffs = ([b0, b1, b2], [a1, a2])
    return coeffs

class PythonBPF:
    def __init__(self,fs,fl,fh):

        self.num_channels = len(fl)
        self.b = [np.empty(len(fl)) for _ in range(3)]
        self.a = [np.empty(len(fl)) for _ in range(2)]
        for i,fli in enumerate(fl):
            b, a = filtercoeffs(fs, fli, fh[i])
            self.b[0][i] = b[0]
            self.b[1][i] = b[1]
            self.b[2][i] = b[2]
            self.a[0][i] = a[0]
            self.a[1][i] = a[1]

        self.previous_filtered_values = None
        self.previous_values = None

    def calculate_initial_values(self, x):
        '''The filter state and previous values parameters'''
        self.previous_filtered_values = [x, x]
        self.previous_values = [x, x]

    def filter(self, x):
        '''carries out one iteration of the filtering step. self.calculate_initial_values
        must be called once before this can be called. Updates internal filter states.

        input:
        x : vector of values to filter. Must be 1D lost or numpy array len self.num_channels
        out:
        x_f : vector of filtered values, 1D list with shape = x.shape'''
        x_f = []
        for i in range(6):
            x_f.append(self.b[0][i]*x[i] + self.b[1][i]*self.previous_values[0][i] + self.b[2][i]*self.previous_values[1][i] - self.a[0][i]*self.previous_filtered_values[0][i] - self.a[1][i]*self.previous_filtered_values[1][i])

        self.previous_values[1] = self.previous_values[0]
        self.previous_values[0] = x
        self.previous_filtered_values[1] = self.previous_filtered_values[0]
        self.previous_filtered_values[0] = x_f
        return x_f
