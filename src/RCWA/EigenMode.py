import tensorflow as tf
import numpy as np
from src.RCWA.Modes import Modes
from src.RCWA.Utils import block
from scipy import linalg as LA
import matplotlib.pyplot as plt
# SciPy's linalg module for solving eigen problem


class EigenMode:
    def __init__(self, modes: Modes):
        self.modes = modes
        pass

    def conv_matrix(self, a):
        ny, nx = a.shape

        # Convert indices to mode numbers
        mx_prime = self.modes.mx[None, :]
        my_prime = self.modes.my[None, :]
        mx = self.modes.mx[:, None]
        my = self.modes.my[:, None]

        # Calculate indices for convolution
        ind_x = (mx - mx_prime) % nx
        ind_y = (my - my_prime) % ny

        # Perform convolution operation
        indices = tf.stack([ind_y, ind_x], -1)
        conv = tf.gather_nd(a, indices) / (nx*ny)
        return tf.cast(conv, tf.dtypes.complex128)

    def Pmat(self, conv_er, conv_ur):
        conv_er = tf.cast(conv_er, tf.dtypes.complex128)
        conv_ur = tf.cast(conv_ur, tf.dtypes.complex128)
        KX = tf.cast(tf.linalg.diag(self.modes.kx), tf.dtypes.complex128)
        KY = tf.cast(tf.linalg.diag(self.modes.ky), tf.dtypes.complex128)

        # Calculate the P matrix based on the convolution of permittivity (conv_er)
        # and permeability (conv_ur)
        er_inv_KX = tf.linalg.solve(conv_er, KX)
        er_inv_KY = tf.linalg.solve(conv_er, KY)
        P11 = KX @ er_inv_KY
        P12 = conv_ur - KX @ er_inv_KX
        P21 = KY @ er_inv_KY - conv_ur
        P22 = -KY @ er_inv_KX

        # Construct the P matrix using block matrix notation
        P = block([[P11, P12], [P21, P22]])
        return P

    def PQmat(self, conv_er, conv_ur):
        # Calculate the P and Q matrices based on the convolution of
        # permittivity (conv_er) and permeability (conv_ur)
        P = self.Pmat(conv_er, conv_ur)
        Q = self.Pmat(conv_ur, conv_er)
        return P, Q

    def from_PQ_mat(self, P, Q):
        """
        Calculates the eigen modes of a system given P and Q matrices.

        Args:
            P (numpy.ndarray): P matrix.
            Q (numpy.ndarray): Q matrix.

        Returns:
            W (numpy.ndarray): Mode matrix describing the E field.
            V (numpy.ndarray): Mode matrix describing the H field .
            LAM (numpy.ndarray): Eigen-values corresponding to the mode propagation.
        """
        omega2 = P@Q
        LAM2, W = LA.eig(omega2)
        LAM = tf.sqrt(LAM2)
        V = Q@W@tf.linalg.diag(1/LAM)

        self.W = tf.cast(W, tf.dtypes.complex128)
        self.V = tf.cast(V, tf.dtypes.complex128)
        self.LAM = tf.cast(LAM, tf.dtypes.complex128)

    def from_material(self, er, ur):
        ft_er = tf.signal.fft2d(tf.cast(er, tf.dtypes.complex128))
        ft_ur = tf.signal.fft2d(tf.cast(ur, tf.dtypes.complex128))
        conv_er = self.conv_matrix(ft_er)  # Convolution matrix for er
        conv_ur = self.conv_matrix(ft_ur)  # Convolution matrix for ur
        # Calculate P and Q matrices using the given convolutions
        P, Q = self.PQmat(conv_er, conv_ur)
        self.from_PQ_mat(P, Q)

    def from_material_er(self, er):
        ft_er = tf.signal.fft2d(tf.cast(er, tf.dtypes.complex128))
        conv_er = self.conv_matrix(ft_er)  # Convolution matrix for er
        P, Q = self.PQmat(conv_er, tf.eye(conv_er.shape[0]))
        self.from_PQ_mat(P, Q)

    def from_homogeneous(self, er, ur):
        """
        Function to calculate the homogeneous layer parameters for RCWA.

        Args:
            er: Relative permittivity of the layer.
            ur: Relative permeability of the layer.

        Returns:
            W: Eigenmode matrix for E field.
            V: Eigenmode matrix for H field.
            LAM: Eigenvalues for the layer.
        """
        n2 = er * ur

        kx = self.modes.kx
        ky = self.modes.ky

        W = tf.eye(self.modes.num_modes*2, dtype=tf.dtypes.complex128)
        kz = tf.sqrt(tf.cast(n2-kx**2-ky**2, tf.dtypes.complex128))
        LAM = tf.concat([1j * kz, 1j * kz], 0)
        V11 = tf.linalg.diag(kx*ky/kz)
        V12 = tf.linalg.diag((n2-kx**2)/kz)
        V21 = tf.linalg.diag((ky**2-n2)/kz)
        V22 = -V11

        V = -1j/ur*block([
            [V11, V12],
            [V21, V22],
        ])

        self.W = tf.cast(W, tf.dtypes.complex128)
        self.V = tf.cast(V, tf.dtypes.complex128)
        self.LAM = tf.cast(LAM, tf.dtypes.complex128)

    def vis_mode(self, mode_ind: int, offset=0):
        modes = self.modes
        domain = self.modes.domain

        mode_x = modes.mx[mode_ind]
        mode_y = modes.my[mode_ind]

        mode_mask = np.zeros(modes.num_modes*2)
        mode_mask[mode_ind + offset*modes.num_modes] = 1

        W = self.W
        V = self.V
        LAM = self.LAM

        X = tf.cast(tf.abs(tf.exp(-LAM*100000)), tf.dtypes.complex128)
        C = tf.linalg.diag(X)@tf.linalg.solve(W, mode_mask[:, None])

        # Extract the desired columns from W
        w = tf.reshape(W@C, [
            2, modes.num_modes_y, modes.num_modes_x
        ])

        v = tf.reshape(V@C, [
            2, modes.num_modes_y, modes.num_modes_x
        ])

        # Calculate the field by performing an inverse Fourier transform

        field_w = tf.signal.fftshift(tf.abs(tf.signal.ifft2d(w)))
        field_v = tf.signal.fftshift(tf.abs(tf.signal.ifft2d(v)))

        # Display the field as an image
        plt.title(
            f"(mx, my) = ({mode_x}, {mode_y}), {'EX' if offset==0 else 'EY'}")
        plt.imshow(block([
            [field_w[0], field_w[0], field_w[1], field_w[1]],
            [field_w[0], field_w[0], field_w[1], field_w[1]],
            [field_v[0], field_v[0], field_v[1], field_v[1]],
            [field_v[0], field_v[0], field_v[1], field_v[1]],
        ]), cmap='gray', extent=[0, domain.period_x, 0, domain.period_y])

        plt.axvline(domain.period_x/2)
        plt.axhline(domain.period_y/2)

        # Hide the axis
        plt.axis(False)
