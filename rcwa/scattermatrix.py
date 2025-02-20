import numpy as np
from .utils import block_matrix
from .isotropic import homogeneous_isotropic_matrix


def star_product(A, B):
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(a11.shape[0], dtype=a11.dtype)
    Ib11a22 = I - np.matmul(b11, a22)
    Ia22b11 = I - np.matmul(a22, b11)
    s11 = a11 + np.matmul(a12, np.linalg.solve(Ib11a22, np.matmul(b11, a21)))
    s22 = b22 + np.matmul(b21, np.linalg.solve(Ia22b11, np.matmul(a22, b12)))
    s12 = np.matmul(a12, np.linalg.solve(Ib11a22, b12))
    s21 = np.matmul(b21, np.linalg.solve(Ia22b11, a21))
    return (s11, s12, s21, s22)


def build_scatter_from_AB(A, B):
    # build scatter matrix considering:  A c1 = B c2
    half = A.shape[0]//2
    a11 = A[:half, :half]
    a12 = A[:half, half:]
    a21 = A[half:, :half]
    a22 = A[half:, half:]

    b11 = B[:half, :half]
    b12 = B[:half, half:]
    b21 = B[half:, :half]
    b22 = B[half:, half:]

    S = np.linalg.solve(
        block_matrix([
            [-a12, b11],
            [-a22, b21]
        ]),
        block_matrix([
            [a11, -b12],
            [a21, -b22]
        ])
    )

    s11 = S[:half, :half]
    s12 = S[:half, half:]
    s21 = S[half:, :half]
    s22 = S[half:, half:]

    return (s11, s12, s21, s22)


def build_scatter_from_omega(omega, W0, k0L):
    # build scatter matrix
    LAM, W = np.linalg.eig(omega)
    order = np.argsort(-np.imag(LAM)*1000+np.real(LAM))
    LAM = LAM[order]
    W = W[:, order]
    WiW0 = np.linalg.solve(W, W0)
    A = np.diag(np.exp(LAM*k0L/2)) @ WiW0
    B = np.diag(np.exp(-LAM*k0L/2)) @ WiW0
    return build_scatter_from_AB(A, B), W, LAM

def build_scatter_from_WV(W, V, W0, V0, LAM, k0L):
    WiW0 = np.linalg.solve(W, W0)
    ViV0 = np.linalg.solve(V, V0)
    A = WiW0 + ViV0
    B = WiW0 - ViV0
    X = np.diag(np.exp(LAM*k0L))
    AiXB = np.linalg.solve(A, X@B)
    AiXA = np.linalg.solve(A, X@A)
    XBAiXB = X - B @ AiXB
    XBAiXA = X - B @ AiXA

    AXBAXB = A-XBAiXB

    s11 = np.linalg.solve(AXBAXB, XBAiXA-B)
    s12 = np.linalg.solve(AXBAXB, X@(A-B@np.linalg.solve(A, B)))
    s21 = s12
    s22 = s11

    W = block_matrix([
        [W, W],
        [V, -V]
    ])

    LAM = np.concatenate([LAM, -LAM], axis=0)

    return (s11, s12, s21, s22), W, LAM


def build_scatter_from_homo(er, ur, kx, ky, W0, k0L):
    LAM, W = homogeneous_isotropic_matrix(er, ur, kx, ky)
    n_half = kx.shape[0]*2
    return build_scatter_from_WV(
        W[:n_half, :n_half],
        W[n_half:, :n_half],
        W0[:n_half, :n_half],
        W0[n_half:, :n_half], LAM[:n_half], k0L)

    # build scatter matrix
    # LAM, W = np.linalg.eig(omega)
    # order = np.argsort(-np.imag(LAM)*1000+np.real(LAM))
    # LAM = LAM[order]
    # W = W[:, order]
    WiW0 = np.linalg.solve(W, W0)
    A = np.diag(np.exp(LAM*k0L/2)) @ WiW0
    B = np.diag(np.exp(-LAM*k0L/2)) @ WiW0
    return build_scatter_from_AB(A, B), W, LAM


def build_scatter_from_omega2(EH_mat, HE_mat, W0, k0L):
    # build scatter matrix
    omega2 = EH_mat @ HE_mat
    LAM2, W = np.linalg.eig(omega2)
    # LAM = np.conj(np.sqrt(LAM2))
    LAM = np.sqrt(LAM2)
    V = HE_mat @ W @ np.diag(1/LAM)
    n_half = W0.shape[0]//2
    return build_scatter_from_WV(W, V, W0[:n_half, :n_half],  W0[n_half:, :n_half], LAM, k0L)


def build_scatter_side(er, ur, kx, ky, W0, transmission_side=False):
    # build scatter matrix for reflection and transmission side
    LAM, W = homogeneous_isotropic_matrix(er, ur, kx, ky)
    if transmission_side:
        A = W0
        B = W
    else:
        A = W
        B = W0
    return build_scatter_from_AB(A, B), W, LAM


# def build_scatter_side(er, ur, kx, ky, W0, transmission_side=False):
#     # build scatter matrix for reflection and transmission side
#     LAM, W = homogeneous_isotropic_matrix(er, ur, kx, ky)
#     n_half = kx.shape[0]*2
#     WiW0 = np.linalg.solve(W[:n_half, :n_half], W0[:n_half, :n_half])
#     ViV0 = np.linalg.solve(W[n_half:, :n_half], W0[n_half:, :n_half])
#     A = WiW0 + ViV0
#     B = WiW0 - ViV0
#     Ai = np.linalg.inv(A)
#     s11 = B @ Ai
#     s12 = 0.5 * (A-s11@B)
#     s21 = 2 * Ai
#     s22 = -Ai@B

#     if transmission_side:
#         s11, s12, s21, s22 = s22, s21, s12, s11
#     return (s11, s12, s21, s22), W, LAM


def get_field_incide(c1p, c1m, c3p, c3m, A, B):
    # this function breaks the scatter matrix,
    # to get the mode value inside
    # c1p -> A -> c2p -> B -> c3p
    # c1m <- A <- c2m <- B <- c3m
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(*a11.shape)
    c2p = np.linalg.solve(I-a22@b11, a21@c1p+a22@b12@c3m)
    c2m = np.linalg.solve(I-b11@a22, b12@c3m+b11@a21@c1p)
    # Alternative formula, not working very well
    # c2p = np.linalg.solve(b21, c3p - b22@c3m)
    # c2m = np.linalg.solve(a12, c1m - a11@c1p)
    return c2p, c2m
