import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

complex_type = tf.dtypes.complex128
float_type = tf.dtypes.float64
int_type = tf.dtypes.int32


@tf.function
def pinv(M):
    A = tf.math.real(M)
    C = tf.math.imag(M)
    r0 = tf.linalg.pinv(A, rcond=1e-32) @ C
    y11 = tf.linalg.pinv(C @ r0 + A, rcond=1e-32)
    y10 = -r0 @ y11
    M_inverse = tf.cast(tf.complex(y11, y10), dtype=M.dtype)
    return M_inverse


@tf.function
def block2x2(A, B, C, D):
    """Make 2x2 block matrix
    """
    return tf.concat([
        tf.concat([A, B], 1),
        tf.concat([C, D], 1)
    ], 0)


@tf.function
def grating_vector(wavelength, period_x, period_y):
    """Compute grating vector"""
    gx0 = wavelength/period_x
    gy0 = wavelength/period_y
    return gx0, gy0


@tf.function
def calculate_modes(harmonics_x: int, harmonics_y: int):
    """Calculate modes from given harmonics
    """
    modes_x = harmonics_x*2+1
    modes_y = harmonics_y*2+1
    modes = modes_x*modes_y

    # mode number
    mode_x = tf.range(-harmonics_x, harmonics_x+1, dtype=int_type)
    mode_y = tf.range(-harmonics_y, harmonics_y+1, dtype=int_type)
    mode_x, mode_y = tf.meshgrid(mode_x, mode_y)

    # flattened mode number
    mode_x_flat = tf.reshape(mode_x, [-1])
    mode_y_flat = tf.reshape(mode_y, [-1])
    return modes_x, modes_y, modes, mode_x, mode_y, mode_x_flat, mode_y_flat


@tf.function
def conv_modes(mode_x_flat, mode_y_flat):
    """compute mode number for building convolution matrix
    """
    conv_mode_x = mode_x_flat[:, None]-mode_x_flat[None, :]
    conv_mode_y = mode_y_flat[:, None]-mode_y_flat[None, :]
    return conv_mode_x, conv_mode_y


@tf.function
def mode_vectors(kx0, ky0, gx0, gy0, mode_x_flat, mode_y_flat):
    """grating vector and wave vector
    """
    gx = gx0*tf.cast(mode_x_flat, float_type)
    gy = gy0*tf.cast(mode_y_flat, float_type)
    kx = kx0 + gx
    ky = ky0 + gy
    return kx, ky


@tf.function
def homogeneous_matrix(er, ur, kx, ky):
    """Compute matrix for homogeneous layer"""

    n2 = er * ur
    W = tf.eye(kx.shape[0]*2, dtype=complex_type)
    kz = tf.sqrt(tf.cast(n2-kx**2-ky**2, complex_type))
    LAM = tf.concat([1j * kz, 1j * kz], 0)

    V11 = tf.linalg.diag(tf.cast(kx*ky, complex_type)/kz)
    V12 = tf.linalg.diag(tf.cast(n2-kx**2, complex_type)/kz)
    V21 = tf.linalg.diag(tf.cast(ky**2-n2, complex_type)/kz)
    V22 = -V11

    V = -1j/ur*block2x2(
        V11, V12,
        V21, V22
    )

    return LAM, W, V


@tf.function
def side_scatter_mat(er, ur, kx, ky, W0, V0, transmission_side=False):
    LAM, W, V = homogeneous_matrix(er, ur, kx, ky)
    W_W0 = tf.linalg.solve(W, W0)
    # V_V0 = pinv(V)@V0
    V_V0 = tf.linalg.solve(V, V0)
    A = W_W0 + V_V0
    B = W_W0 - V_V0
    A_inv = tf.linalg.inv(A)
    s11 = tf.transpose(
        tf.linalg.solve(tf.transpose(A), tf.transpose(B)))
    s12 = 0.5*(A-s11@B)
    s21 = 2*A_inv
    s22 = -tf.linalg.solve(A, B)
    if transmission_side:
        return (s22, s21, s12, s11)
    return (s11, s12, s21, s22)


@tf.function
def star_product(A, B):
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = tf.eye(a11.shape[0], dtype=a11.dtype)
    Ib11a22 = I-b11@a22
    Ia22b11 = I-a22@b11
    s11 = a11+a12@tf.linalg.solve(Ib11a22, b11@a21)
    s22 = b22+b21@tf.linalg.solve(Ia22b11, a22@b12)
    s12 = a12@tf.linalg.solve(Ib11a22, b12)
    s21 = b21@tf.linalg.solve(Ia22b11, a21)
    return (s11, s12, s21, s22)

# prepare for building matrix per layer


@tf.function
def sampling_coordinate(modes_x, modes_y, period_x, period_y, over_sampling=16):
    period_x = tf.cast(period_x, float_type)
    period_y = tf.cast(period_y, float_type)
    nx = modes_x*over_sampling
    ny = modes_y*over_sampling
    x = tf.range(nx, dtype=float_type)/tf.cast(nx, float_type)*period_x
    y = tf.range(ny, dtype=float_type)/tf.cast(ny, float_type)*period_y
    x, y = tf.meshgrid(x, y)
    return nx, ny, x, y


@tf.function
def conv_mat(conv_mode_x, conv_mode_y, u):
    ny = u.shape[0]
    nx = u.shape[1]
    ft = tf.signal.fft2d(tf.cast(u, complex_type))/(nx*ny)
    indices = tf.stack([conv_mode_y % ny, conv_mode_x % nx], -1)
    return tf.gather_nd(ft, indices)


@tf.function
def build_PQMat(X, Y, E, U):
    Ei = pinv(E)
    X = tf.cast(X, E.dtype)
    Y = tf.cast(Y, E.dtype)
    return block2x2(
        X[:, None]*Ei*Y[None, :],
        U-X[:, None]*Ei*X[None, :],
        Y[:, None]*Ei*Y[None, :]-U,
        -Y[:, None]*Ei*X[None, :]
    )

@tf.function
def build_scatter(kx, ky, er_conv, ur_conv, W0, V0, k0L):
    k0L = tf.cast(k0L, complex_type)
    P = build_PQMat(kx, ky, er_conv, ur_conv)
    Q = build_PQMat(kx, ky, ur_conv, er_conv)

    omega2 = P@Q

    LAM2, W = tf.linalg.eig(omega2)
    LAM = tf.sqrt(LAM2)
    V = Q@W*tf.math.divide_no_nan(1, LAM)[None, :]

    # W_W0 = pinv(W)@W0
    # V_V0 = pinv(V)@V0

    W_W0 = tf.linalg.solve(W, W0)
    V_V0 = tf.linalg.solve(V, V0)

    A = W_W0 - V_V0
    B = W_W0 + V_V0
    X = tf.linalg.diag(tf.exp(-LAM*k0L))

    XB = X@B
    AXBAXB = A-XB@tf.linalg.solve(A, XB)

    s11 = tf.linalg.solve(AXBAXB, XB@tf.linalg.solve(A, X@A)-B)
    s12 = tf.linalg.solve(AXBAXB, X@(A-B@tf.linalg.solve(A, B)))
    s21 = s12
    s22 = s11
    return (s11, s12, s21, s22)


@tf.function
def get_center_mode(harmonics_x, harmonics_y):
    modes_x = harmonics_x*2+1
    modes_y = harmonics_y*2+1
    modes = modes_x * modes_y
    center_mode_x = harmonics_x + harmonics_y*modes_x
    center_mode_y = center_mode_x + modes
    return center_mode_x, center_mode_y


@tf.function
def coefficient_to_field(coefficient, kx, ky, kz):
    U = tf.reshape(coefficient, [2, -1])
    uxkx = U[0]*tf.cast(kx, complex_type)
    uyky = U[1]*tf.cast(ky, complex_type)
    U = tf.concat([U, tf.math.divide_no_nan(-(uxkx+uyky), kz)[None]], 0)
    return U


class Layer:
    def __init__(self, er_conv, ur_conv, thickness):
        self.er_conv = er_conv
        self.ur_conv = ur_conv
        self.thickness = thickness
        self.scatter_mat = None

    def initialize(self):
        self.scatter_mat = None
        pass

    def get_scatter_mat(self, kx, ky, W0, V0, k0):
        if self.scatter_mat is None:
            self.scatter_mat = build_scatter(kx, ky,
                                             self.er_conv, self.ur_conv,
                                             W0, V0, k0*self.thickness)
        return self.scatter_mat


class RCWA:
    def __init__(self):
        self.layers = []
        pass

    def domain(self, period_x, period_y, harmonics_x, harmonics_y, over_sampling):
        self.period_x = period_x
        self.period_y = period_y

        # total harmonics
        self.harmonics_x = harmonics_x
        self.harmonics_y = harmonics_y

        # Modes
        (self.modes_x, self.modes_y, self.modes,
         self.mode_x, self.mode_y,
         self.mode_x_flat, self.mode_y_flat) = calculate_modes(harmonics_x, harmonics_y)
        self.conv_mode_x, self.conv_mode_y = conv_modes(
            self.mode_x_flat, self.mode_y_flat)

        # build shape
        self.nx, self.ny, x, y = sampling_coordinate(self.modes_x, self.modes_y,
                                                     self.period_x, self.period_y,
                                                     over_sampling)
        self.center_mode_x, self.center_mode_y = get_center_mode(
            harmonics_x, harmonics_y)
        return x, y

    def add_layer(self, er, ur, thickness):
        er_conv = conv_mat(self.conv_mode_y, self.conv_mode_x, er)
        ur_conv = conv_mat(self.conv_mode_y, self.conv_mode_x, ur)
        layer = Layer(er_conv, ur_conv, thickness)
        self.layers.append(layer)
        return layer

    def incidence(self, wavelength, kx0, ky0):
        self.k0 = np.pi*2/wavelength
        self.kx0 = kx0
        self.ky0 = ky0
        gx0, gy0 = grating_vector(wavelength, self.period_x, self.period_y)

        self.kx, self.ky = mode_vectors(kx0, ky0, gx0, gy0,
                                        self.mode_x_flat, self.mode_y_flat)

        self.LAM0, self.W0, self.V0 = homogeneous_matrix(
            1, 1, self.kx, self.ky)

    def environment(self, er1=1, ur1=1, er2=1, ur2=1):
        self.SR = side_scatter_mat(
            er1, ur1, self.kx, self.ky, self.W0, self.V0)
        self.ST = side_scatter_mat(er2, ur2, self.kx, self.ky, self.W0, self.V0,
                                   transmission_side=True)

    def build_layers(self):
        for layer in self.layers:
            layer.initialize()

        Sglobal = self.SR
        for layer in self.layers:
            S = layer.get_scatter_mat(
                self.kx, self.ky, self.W0, self.V0, self.k0)
            Sglobal = star_product(Sglobal, S)
        Sglobal = star_product(Sglobal, self.ST)
        self.Sglobal = Sglobal
        return Sglobal

    def default_plot():

        extent = [-harmonics_x, harmonics_x, -harmonics_y, harmonics_y]
        R = tf.reshape(
            coefficient_to_field(s11[:, center_mode_x], kx, ky),
            [3, modes_y, modes_x])
        T = tf.reshape(
            coefficient_to_field(s12[:, center_mode_x], kx, ky),
            [3, modes_y, modes_x])

        plt.subplot(2, 2, 1)
        plt.imshow(tf.reduce_sum(tf.abs(R)**2, 0), vmin=0, vmax=1,
                   cmap='jet', extent=extent, origin='lower')
        plt.subplot(2, 2, 2)
        plt.imshow(tf.reduce_sum(tf.abs(T)**2, 0), vmin=0, vmax=1,
                   cmap='jet', extent=extent, origin='lower')

        R = tf.reshape(
            coefficient_to_field(s11[:, center_mode_y], kx, ky),
            [3, modes_y, modes_x])
        T = tf.reshape(
            coefficient_to_field(s12[:, center_mode_y], kx, ky),
            [3, modes_y, modes_x])

        plt.subplot(2, 2, 3)
        plt.imshow(tf.reduce_sum(tf.abs(R)**2, 0), vmin=0, vmax=1,
                   cmap='jet', extent=extent, origin='lower')
        plt.subplot(2, 2, 4)
        plt.imshow(tf.reduce_sum(tf.abs(T)**2, 0), vmin=0, vmax=1,
                   cmap='jet', extent=extent, origin='lower')
        plt.show()
