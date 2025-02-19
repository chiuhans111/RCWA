import numpy as np
from .utils import block_matrix
from .modes import Modes
from .common_matrix_equation import homogeneous_isotropic_matrix as homogeneous_isotropic_matrix_common



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

    erz = er
    urz = ur

    erc = modes.convolution_matrix(er)
    urc = modes.convolution_matrix(ur)
    erci = modes.convolution_matrix(1/er)
    urci = modes.convolution_matrix(1/ur)

    zero = np.zeros_like(erz)

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


def homogeneous_isotropic_matrix(er, ur, modes: Modes):
    return homogeneous_isotropic_matrix_common(er, ur, modes.kx, modes.ky)