import tensorflow as tf
from RCWA.EigenMode import EigenMode
from RCWA.Modes import Modes
from scipy import linalg as LA


def StarProduct(A, B):
    # Redheffer star product

    # Extract elements from matrices A and B
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    I = tf.eye(A11.shape[0], dtype=tf.dtypes.complex128)

    # Compute intermediate matrices
    A12_I_B11A22 = tf.transpose(
        LA.solve(tf.transpose(I-B11@A22), tf.transpose(A12)))
    B21_I_A22B11 = tf.transpose(
        LA.solve(tf.transpose(I-A22@B11), tf.transpose(B21)))

    # Calculate elements of the Star Product matrix
    S11 = A12_I_B11A22@B11@A21+A11
    S12 = A12_I_B11A22@B12
    S21 = B21_I_A22B11@A21
    S22 = B21_I_A22B11@A22@B12+B22

    # Return the Star Product matrix
    return (S11, S12, S21, S22)


class ScatterMat:
    """
    Represents a scattering matrix for multilayer structures.

    The ScatterMat class encapsulates a scattering matrix that describes the
    transmission and reflection properties of a multilayer device. It provides
    functionality to perform matrix multiplication between scattering matrices
    using the '@' operator.

    Args:
        value (tuple): Tuple containing the elements of the scattering matrix.

    Attributes:
        value (tuple): The scattering matrix represented as a tuple of elements.

    Methods:
        __matmul__(self, other): Performs Redheffer star product with another scattering matrix.

    Usage Example:
        A = ScatterMat((A11, A12, A21, A22))
        B = ScatterMat((B11, B12, B21, B22))
        S = A @ B  # Perform Redheffer star product between scattering matrices A and B
    """

    def __init__(self, value) -> None:
        self.value = value

    def __matmul__(self, other):
        #
        A = self.value
        B = other.value
        S = ScatterMat(StarProduct(A, B))
        return S


class ScatterMatBuilder:
    def __init__(self, modes: Modes) -> None:
        eigenmode = EigenMode(modes)
        eigenmode.from_homogeneous(1, 1)

        self.W0 = eigenmode.W
        self.V0 = eigenmode.V
        self.LAM0 = eigenmode.LAM

        self.k0 = modes.k0
        pass

    def BuildScatter(self, eigenmode: EigenMode, L):
        """
        Function to build the scattering matrix
        for a multilayer structure using given inputs.

        Args:
            W: Matrix W.
            V: Matrix V.
            LAM: Eigen-value matrix.
            L: Layer thicknesses.

        Returns:
            ScatterMat: Scattering matrix of the multilayer structure.
        """
        W = eigenmode.W
        V = eigenmode.V
        LAM = eigenmode.LAM

        WW = LA.solve(W, self.W0)
        VV = LA.solve(V, self.V0)
        A = WW-VV  # Swap sign improves accuracy???
        B = WW+VV
        X = tf.linalg.diag(tf.exp(-LAM*self.k0*L))
        AiX = LA.solve(A, X)
        M = A-X@B@AiX@B
        S11 = LA.solve(M, X@B@AiX@A-B)
        S12 = LA.solve(M, X@(A-B@LA.solve(A, B)))
        return ScatterMat((S11, S12, S12, S11))

    def BuildScatterRef(self, eigenmode: EigenMode):
        """
        Function to build the reflection-side scattering matrix
        for a multilayer structure using given inputs.

        Args:
            W: Matrix W.
            V: Matrix V.

        Returns:
            ScatterMat: Reflection-side scattering matrix of the multilayer structure.
        """
        W = eigenmode.W
        V = eigenmode.V

        WW = LA.solve(W, self.W0)
        VV = LA.solve(V, self.V0)
        A = WW+VV
        B = WW-VV
        Ai = tf.linalg.inv(A)
        S11 = tf.transpose(LA.solve(tf.transpose(A), tf.transpose(B)))
        S12 = 0.5*(A-S11@B)
        S21 = 2*Ai
        S22 = -LA.solve(A, B)
        return ScatterMat((S11, S12, S21, S22))

    def BuildScatterTrn(self, eigenmode: EigenMode):
        """
        Function to build the transmission-side scattering matrix
        for a multilayer structure using given inputs.

        Args:
            W: Matrix W.
            V: Matrix V.

        Returns:
            ScatterMat: Transmission-side scattering matrix of the multilayer structure.
        """
        W = eigenmode.W
        V = eigenmode.V

        WW = LA.solve(W, self.W0)
        VV = LA.solve(V, self.V0)
        A = WW+VV
        B = WW-VV
        Ai = tf.linalg.inv(A)
        S11 = -LA.solve(A, B)
        S12 = 2*Ai
        S22 = tf.transpose(LA.solve(tf.transpose(A), tf.transpose(B)))
        S21 = 0.5*(A-S22@B)
        return ScatterMat((S11, S12, S21, S22))
