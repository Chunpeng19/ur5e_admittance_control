import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

def filtercoeffs(fs, fc):
    Ts=1/float(fs)
    wc = 2 * np.pi * fc
    b0 = 2.0 / (2.0 + wc * Ts)
    b1 = -b0
    a1 = (2.0 - wc * Ts) / (2.0 + wc * Ts)

    coeffs = ([b0, b1], [a1])
    return coeffs

class PythonHPF:
    def __init__(self,fc,fs):

        self.num_channels = len(fc)
        self.b = [np.empty(len(fc)) for _ in range(2)]
        self.a = [np.empty(len(fc)) for _ in range(1)]
        for i,fci in enumerate(fc):
            b, a = filtercoeffs(fs, fci)
            self.b[0][i] = b[0]
            self.b[1][i] = b[1]
            self.a[0][i] = a[0]

        self.previous_filtered_values = None
        self.previous_values = None

    def calculate_initial_values(self, x):
        '''The filter state and previous values parameters'''
        self.previous_filtered_values = x
        self.previous_values = x

    def filter(self, x):
        '''carries out one iteration of the filtering step. self.calculate_initial_values
        must be called once before this can be called. Updates internal filter states.

        input:
        x : vector of values to filter. Must be 1D lost or numpy array len self.num_channels
        out:
        x_f : vector of filtered values, 1D list with shape = x.shape'''
        x_f = []
        for i in range(6):
            x_f.append(self.b[0][i]*x[i] + self.b[1][i]*self.previous_values[i] + \
            self.a[0][i]*self.previous_filtered_values[i])

        self.previous_values = x
        self.previous_filtered_values = x_f
        return x_f
