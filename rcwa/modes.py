import numpy as np
from .isotropic import homogeneous_isotropic_matrix
from .utils import block_matrix

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
        self.n_modes = self.mx.size
        self.k0 = 2*np.pi / self.wavelength
        self.set_direction(kx0, ky0)

 
    def set_direction(self, kx0, ky0):
        self.kx0 = kx0
        self.ky0 = ky0
        self.kx = self.kx0 + self.gx * self.mx
        self.ky = self.ky0 + self.gy * self.my
        self.LAM0, self.W0 = homogeneous_isotropic_matrix(1, 1, self.kx, self.ky)


    def convolution_matrix(self, u):
        # build convolution matrix for the last 2 axis of u
        return convolution_matrix(u, self.mx, self.my)




def build_omega(er, ur, modes: Modes):
    kxd = np.diag(modes.kx)
    kyd = np.diag(modes.ky)

    k1term = 1j * np.concatenate([kxd, kyd], axis=0)  # T
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    e1term = np.array([er[1, 2], -er[0, 2]])  # T
    u1term = np.array([ur[1, 2], -ur[0, 2]])  # T

    e2term = np.array([er[2, 0], er[2, 1]])
    u2term = np.array([ur[2, 0], ur[2, 1]])

    erz = er[2, 2]
    urz = ur[2, 2]

    e_mat = block_matrix(modes.convolution_matrix(np.array([
        [er[1, 0], er[1, 1]],
        [-er[0, 0], -er[0, 1]]
    ])))

    u_mat = block_matrix(modes.convolution_matrix(np.array([
        [ur[1, 0], ur[1, 1]],
        [-ur[0, 0], -ur[0, 1]]
    ])))

    e1erz = np.concatenate(modes.convolution_matrix(
        e1term/erz[None]), axis=0)
    u1urz = np.concatenate(modes.convolution_matrix(
        u1term/urz[None]), axis=0)
    e2erz = np.concatenate(modes.convolution_matrix(
        e2term/erz[None]), axis=1)
    u2urz = np.concatenate(modes.convolution_matrix(
        u2term/urz[None]), axis=1)

    u1term = np.concatenate(modes.convolution_matrix(u1term), axis=0)
    e1term = np.concatenate(modes.convolution_matrix(e1term), axis=0)

    erzi = modes.convolution_matrix(1/erz)
    urzi = modes.convolution_matrix(1/urz)

    EE_mat = - k1term @ e2erz + u1urz @ k2term
    EH_mat = k1term @ erzi @ k2term + u_mat - u1term @ e2erz
    HH_mat = - k1term @ u2urz + e1erz @ k2term
    HE_mat = k1term @ urzi @ k2term + e_mat - e1term @ e2erz

    omega = block_matrix([
        [EE_mat, EH_mat],
        [HE_mat, HH_mat]
    ])

    return omega


def build_omega2_diagonal(er, ur, modes: Modes):
    kxd = np.diag(modes.kx)
    kyd = np.diag(modes.ky)

    k1term = 1j * np.concatenate([kxd, kyd], axis=0) 
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    erz = er[2]
    urz = ur[2]

    zero = np.zeros_like(erz)

    e_mat = block_matrix(modes.convolution_matrix(np.array([
        [zero, er[1]],
        [-er[0], zero]
    ])))

    u_mat = block_matrix(modes.convolution_matrix(np.array([
        [zero, ur[1]],
        [-ur[0], zero]
    ])))

    erzi = modes.convolution_matrix(1/erz)
    urzi = modes.convolution_matrix(1/urz)

    EH_mat = k1term @ erzi @ k2term + u_mat
    HE_mat = k1term @ urzi @ k2term + e_mat

    return EH_mat, HE_mat


def build_omega2_isotropic(er, ur, modes: Modes):
    kxd = np.diag(modes.kx)
    kyd = np.diag(modes.ky)

    k1term = 1j * np.concatenate([kxd, kyd], axis=0) 
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    erc = modes.convolution_matrix(er)
    urc = modes.convolution_matrix(ur)
    erci = modes.convolution_matrix(1/er)
    urci = modes.convolution_matrix(1/ur)

    zero = np.zeros_like(erc)

    e_mat = block_matrix([
        [zero, erc],
        [-erc, zero]
    ])
    u_mat = block_matrix([
        [zero, urc],
        [-urc, zero]
    ])

    EH_mat = k1term @ erci @ k2term + u_mat
    HE_mat = k1term @ urci @ k2term + e_mat

    return EH_mat, HE_mat
