import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

def filtercoeffs(fs, fc):
    Ts=1/float(fs)
    zeta = 2 ** 0.5 / 2
    w0 = 2 * np.pi * fc
    K = w0 / np.tan(w0 * Ts / 2)
    phi = K ** 2 + 2 * zeta * w0 * K + w0 ** 2
    b0 = w0 ** 2 / phi
    b1 = 2 * b0
    b2 = b0
    a1 = (2 * w0 ** 2 - 2 * K ** 2) / phi
    a2 = (K ** 2 - 2 * zeta * w0 * K + w0 ** 2) / phi

    coeffs = ([b0, b1, b2], [a1, a2])
    return coeffs

class PythonFilter:
    def __init__(self,fc,fs):

        self.num_channels = len(fc)
        self.b = [np.empty(len(fc)) for _ in range(3)]
        self.a = [np.empty(len(fc)) for _ in range(2)]
        for i,fci in enumerate(fc):
            b, a = filtercoeffs(fs, fci)
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
            x_f.append(self.b[0][i]*x[i] + self.b[1][i]*self.previous_values[0][i] + \
            self.b[2][i]*self.previous_values[1][i] - self.a[0][i]*self.previous_filtered_values[0][i] - \
            self.a[1][i]*self.previous_filtered_values[1][i])

        self.previous_values[1] = self.previous_values[0]
        self.previous_values[0] = x
        self.previous_filtered_values[1] = self.previous_filtered_values[0]
        self.previous_filtered_values[0] = x_f
        return x_f
