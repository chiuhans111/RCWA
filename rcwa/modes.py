import numpy as np
from .common_matrix_equation import homogeneous_isotropic_matrix

def get_modes(harmonics_x, harmonics_y):
    # get flatten modes
    mx = np.arange(-harmonics_x, harmonics_x+1)
    my = np.arange(-harmonics_y, harmonics_y+1)
    mx, my = np.meshgrid(mx, my)
    mode_shape = mx.shape
    mx = mx.flatten()
    my = my.flatten()
    return mx, my, mode_shape


def convolution_matrix(u, mx, my):
    # build convolution matrix for the last 2 axis of u
    uf = np.fft.fft2(u) / u.shape[-1] / u.shape[-2]
    ix = mx[:, None] - mx[None, :]
    iy = my[:, None] - my[None, :]
    cm = uf[..., iy, ix]
    return cm


class Modes:
    def __init__(self, wavelength, kx0, ky0, period_x, period_y, harmonics_x, harmonics_y):
        self.wavelength = wavelength
        self.period_x = period_x
        self.period_y = period_y
        self.gx = self.wavelength / self.period_x
        self.gy = self.wavelength / self.period_y
        self.harmonics_x = harmonics_x
        self.harmonics_y = harmonics_y
        self.mx, self.my, self.shape = get_modes(harmonics_x, harmonics_y)
        self.k0 = 2*np.pi / self.wavelength
        self.set_direction(kx0, ky0)
        self.n_modes = self.mx.size


 
    def set_direction(self, kx0, ky0):
        self.kx0 = kx0
        self.ky0 = ky0
        self.kx = self.kx0 + self.gx * self.mx
        self.ky = self.ky0 + self.gy * self.my

        self.LAM0, self.W0 = homogeneous_isotropic_matrix(1, 1, self.kx, self.ky)


    def convolution_matrix(self, u):
        # build convolution matrix for the last 2 axis of u
        return convolution_matrix(u, self.mx, self.my)
