import numpy as np
import matplotlib.pyplot as plt


def omega_matrix(er, ur, kx, ky):
    # omega matrix describes the differential equation

    # d  [Ex]   [M1 .  M2 . ] [Ex]
    # -- [Ey] = [ . .   . . ] [Ey]
    # dx [Hx]   [M3 .  M4 . ] [Hx]
    #    [Hy]   [ . .   . . ] [Hy]

    M = np.array([
        [0, 1],
        [-1, 0]
    ])
    M1 = (-np.array([[1j*kx], [1j*ky]]) / er[2, 2]) @ er[2, None, 0:2] \
        + (np.array([[ur[1, 2]], [-ur[0, 2]]]) /
           ur[2, 2]) @ np.array([[-1j*ky, 1j*kx]])
    M2 = (np.array([[1j*kx], [1j*ky]]) / er[2, 2]) @ np.array([[-1j*ky, 1j*kx]]) \
        + M@ur[0:2, 0:2] - (np.array([[ur[1, 2]], [-ur[0, 2]]]) /
                            ur[2, 2]) @ ur[2, None, 0:2]
    M4 = (-np.array([[1j*kx], [1j*ky]]) / ur[2, 2]) @ ur[2, None, 0:2] \
        + (np.array([[er[1, 2]], [-er[0, 2]]]) /
           er[2, 2]) @ np.array([[-1j*ky, 1j*kx]])
    M3 = (np.array([[1j*kx], [1j*ky]]) / ur[2, 2]) @ np.array([[-1j*ky, 1j*kx]]) \
        + M@er[0:2, 0:2] - (np.array([[er[1, 2]], [-er[0, 2]]]) /
                            er[2, 2]) @ er[2, None, 0:2]

    return np.block([
        [M1, M2],
        [M3, M4]
    ])


def eig_from_isotropic(er, ur, kx, ky):
    kz = np.sqrt(np.complex64(er*ur-kx**2-ky**2))
    urikz = ur*1j*kz
    W = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        np.array([-kx*ky, kx**2-er*ur, kx*ky, -kx**2+er*ur])/urikz,
        np.array([er*ur-ky**2, kx*ky, -er*ur+ky**2, -kx*ky])/urikz,
    ])
    LAM = np.array([
        -1j*kz,
        -1j*kz,
        1j*kz,
        1j*kz,
    ])
    return LAM, W


def scatter_matrix_from_AB(A, B):
    A11 = A[:2, :2]
    A12 = A[:2, 2:]
    A21 = A[2:, :2]
    A22 = A[2:, 2:]

    B11 = B[:2, :2]
    B12 = B[:2, 2:]
    B21 = B[2:, :2]
    B22 = B[2:, 2:]

    D = np.block([
        [A12, -B11],
        [A22, -B21],
    ])

    E = np.block([
        [-A11, B12],
        [-A21, B22],
    ])

    S = np.linalg.solve(D, E)


    s11 = S[:2, :2]
    s12 = S[:2, 2:]
    s21 = S[2:, :2]
    s22 = S[2:, 2:]

    return (s11, s12, s21, s22)


def scatter_matrix(W, W1, W2, LAM, k0L):
    WiW1 = np.linalg.solve(W, W1)
    WiW2 = np.linalg.solve(W, W2)
    A = np.diag([1, 1, *np.exp(-LAM[:2]*k0L)])@WiW2
    B = np.diag([*np.exp(LAM[2:]*k0L), 1, 1])@WiW1
    return scatter_matrix_from_AB(A, B)


def scatter_matrix_symmetric(W, W0, LAM, k0L):
    WiW0 = np.linalg.solve(W, W0)
    A = np.diag([1, 1, *np.exp(-LAM[:2]*k0L)])@WiW0
    B = np.diag([*np.exp(LAM[2:]*k0L), 1, 1])@WiW0
    return scatter_matrix_from_AB(A, B)


def scatter_matrix_from_anisotropic(er, ur, kx, ky, W0, k0L):
    omega = omega_matrix(er, ur, kx, ky)
    LAM, W = np.linalg.eig(omega)
    order = np.argsort(np.imag(LAM))
    W = W[:, order]
    LAM = LAM[order]
    return scatter_matrix_symmetric(W, W0, LAM, k0L)


def scatter_matrix_from_isotropic(er, ur, kx, ky, W0, k0L):
    LAM, W = eig_from_isotropic(er, ur, kx, ky)
    return scatter_matrix_symmetric(W, W0, LAM, k0L)


def scatter_matrix_from_reflection_side(er, ur, kx, ky, W0):
    LAM, W = eig_from_isotropic(er, ur, kx, ky)
    WiW0 = np.linalg.solve(W, W0)
    A = np.eye(4)
    B = WiW0
    return scatter_matrix_from_AB(A, B)


def scatter_matrix_from_transmission_side(er, ur, kx, ky, W0):
    LAM, W = eig_from_isotropic(er, ur, kx, ky)
    WiW0 = np.linalg.solve(W, W0)
    A = WiW0
    B = np.eye(4)
    return scatter_matrix_from_AB(A, B)


def star_product(A, B):
    # Redheffer star product

    # Extract elements from matrices A and B
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    I = np.eye(A11.shape[0])

    # Compute intermediate matrices
    I_B11A22 = I-B11@A22
    I_A22B11 = I-A22@B11

    # Calculate elements of the Star Product matrix
    S11 = A12@np.linalg.solve(I_B11A22, B11@A21)+A11
    S12 = A12@np.linalg.solve(I_B11A22, B12)
    S21 = B21@np.linalg.solve(I_A22B11, A21)
    S22 = B21@np.linalg.solve(I_A22B11, A22@B12)+B22

    # Return the Star Product matrix
    return (S11, S12, S21, S22)


def get_intensity(Ex, Ey, kx, ky, kz):
    Ez = -(Ex*kx+Ey*ky)/kz
    I = np.abs(Ex)**2+np.abs(Ey)**2+np.abs(Ez)**2
    return I


def simulate(wavelength, kx, ky, Ex, Ey, n_ref, n_tra, layers):
    k0 = np.pi*2/wavelength
    kz = np.sqrt(np.complex64(n_ref**2-kx**2-ky**2))
    W0 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1j*kx*ky/kz, 1j*(1-kx**2)/kz, -1j*kx*ky/kz, -1j*(1-kx**2)/kz],
        [-1j*(1-ky**2)/kz, -1j*kx*ky/kz, 1j*(1-ky**2)/kz, 1j*kx*ky/kz],
    ])

    L = 100
    Sref = scatter_matrix_from_reflection_side(n_ref**2, 1, kx, ky, W0)
    Stra = scatter_matrix_from_transmission_side(n_tra**2, 1, kx, ky, W0)

    Sglobal = Sref

    for er, ur, L in layers:
        if len(np.array(er).shape) > 1 or len(np.array(ur).shape) > 1:
            S = scatter_matrix_from_anisotropic(er, ur, kx, ky, W0, k0*L)
        else:
            S = scatter_matrix_from_isotropic(er, ur, kx, ky, W0, k0*L)
        Sglobal = star_product(Sglobal, S)

    Sglobal = star_product(Sglobal, Stra)

    kz_ref = np.sqrt(np.complex64(n_ref**2-kx**2-ky**2))
    kz_tra = np.sqrt(np.complex64(n_tra**2-kx**2-ky**2))

    I = get_intensity(Ex, Ey, kx, ky, kz)

    E_ref = Sglobal[0] @ np.array([[Ex], [Ey]])
    E_tra = Sglobal[2] @ np.array([[Ex], [Ey]])

    I_ref = get_intensity(E_ref[0, 0], E_ref[1, 0], kx, ky, kz_ref)
    I_tra = get_intensity(E_tra[0, 0], E_tra[1, 0], kx, ky, kz_tra)

    R = I_ref/I*np.real(kz_ref)/np.real(kz)
    T = I_tra/I*np.real(kz_tra)/np.real(kz)
    return R, T
