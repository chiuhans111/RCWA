import numpy as np
from .utils import block_matrix

def homogeneous_isotropic_matrix(er, ur, kx, ky):
    n2 = er * ur
    W = np.eye(kx.shape[0]*2)
    kz = np.sqrt((n2-kx**2-ky**2).astype('complex'))
    LAM = np.concatenate([1j * kz, 1j * kz], axis=0)
    LAM = np.concatenate([LAM, -LAM], axis=0)

    V11 = np.diag(kx*ky/kz)
    V12 = np.diag((n2-kx**2)/kz)
    V21 = np.diag((ky**2-n2)/kz)
    V22 = -V11

    V = -1j/ur * block_matrix([
        [V11, V12],
        [V21, V22]
    ])

    W = block_matrix([
        [W, W],
        [V, -V]
    ])

    return LAM, W