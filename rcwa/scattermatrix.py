"""
Scattering Matrix (S-matrix) Kernel and Redheffer Star Product.

This module implements the mathematical core of the RCWA algorithm, including
S-matrix cascading and specialized construction methods for different media types.
"""

from typing import Tuple, List, Any
import numpy as np
from .utils import block_matrix, logger
from .isotropic import homogeneous_isotropic_matrix
import matplotlib.pyplot as plt


def star_product(A: Tuple[np.ndarray, ...], B: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    """
    Computes the Redheffer Star Product of two scattering matrices A and B.
    
    The star product cascades two S-matrices such that the resulting matrix 
    describes the combined response of the stacked layers.
    
    Args:
        A, B: Tuples of (s11, s12, s21, s22) sub-matrices.
        
    Returns:
        S: Tuple of (s11, s12, s21, s22) for the combined system.
    """
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(a11.shape[0], dtype=a11.dtype)
    
    # Pre-compute inversion terms common to S-matrix cascading
    Ib11a22 = I - np.matmul(b11, a22)
    Ia22b11 = I - np.matmul(a22, b11)
    
    # Calculate global sub-blocks
    s11 = a11 + np.matmul(a12, np.linalg.solve(Ib11a22, np.matmul(b11, a21)))
    s22 = b22 + np.matmul(b21, np.linalg.solve(Ia22b11, np.matmul(a22, b12)))
    s12 = np.matmul(a12, np.linalg.solve(Ib11a22, b12))
    s21 = np.matmul(b21, np.linalg.solve(Ia22b11, a21))
    
    return (s11, s12, s21, s22)


def build_scatter_from_AB(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Constructs an S-matrix from the boundary matching equation: A*c1 = B*c2.
    
    Args:
        A, B: Block matrices derived from E/H field continuity at an interface.
        
    Returns:
        (s11, s12, s21, s22): Corresponding scattering sub-blocks.
    """
    half = A.shape[0] // 2
    a11, a12 = A[:half, :half], A[:half, half:]
    a21, a22 = A[half:, :half], A[half:, half:]

    b11, b12 = B[:half, :half], B[:half, half:]
    b21, b22 = B[half:, :half], B[half:, half:]

    # Solve the linear system for the S-matrix blocks
    S = np.linalg.solve(
        block_matrix([[-a12, b11], [-a22, b21]]),
        block_matrix([[a11, -b12], [a21, -b22]])
    )

    return (S[:half, :half], S[:half, half:], S[half:, :half], S[half:, half:])


def build_scatter_from_omega(omega: np.ndarray, W0: np.ndarray, k0L: float) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Constructs an S-matrix for a layer using the general state-space Omega matrix.
    """
    LAM, W = np.linalg.eig(omega)
    # Sort eigenvalues to ensure physical stability (decaying waves)
    order = np.argsort(-np.imag(LAM) * 10000 + np.real(LAM))
    LAM, W = LAM[order], W[:, order]
    
    WiW0 = np.linalg.solve(W, W0)
    # Match boundary coefficients
    A = np.diag(np.exp(LAM * k0L / 2)) @ WiW0
    B = np.diag(np.exp(-LAM * k0L / 2)) @ WiW0
    
    return build_scatter_from_AB(A, B), W, LAM


def build_scatter_from_WV(W: np.ndarray, V: np.ndarray, W0: np.ndarray, V0: np.ndarray, 
                          LAM: np.ndarray, k0L: float) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Efficient implementation of S-matrix construction from E (W) and H (V) mode matrices.
    Reference: Rumpf, "Improved formulation of scattering matrices", PIER B (2011).
    """
    WiW0 = np.linalg.solve(W, W0)
    ViV0 = np.linalg.solve(V, V0)

    A, B = WiW0 + ViV0, WiW0 - ViV0
    X = np.diag(np.exp(LAM * k0L))

    # Perform matrix operations in a stable, consolidated manner
    AiXB = np.linalg.solve(A, X @ B)
    AiXA = np.linalg.solve(A, X @ A)
    XBAiXB = X @ B @ AiXB
    XBAiXA = X @ B @ AiXA
    AXBAXB = A - XBAiXB

    s11 = np.linalg.solve(AXBAXB, XBAiXA - B)
    s12 = np.linalg.solve(AXBAXB, X @ (A - B @ np.linalg.solve(A, B)))
    
    # Symmetry applies for isotropic/symmetric layers
    s21, s22 = s12, s11

    # Full mode matrix reconstruction
    W_full = block_matrix([[W, W], [V, -V]])
    LAM_full = np.concatenate([LAM, -LAM], axis=0)

    return (s11, s12, s21, s22), W_full, LAM_full


def build_scatter_from_homo(er: Any, ur: Any, kx: np.ndarray, ky: np.ndarray, 
                            W0: np.ndarray, k0L: float) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Specialized constructor for homogeneous layers.
    """
    LAM, W = homogeneous_isotropic_matrix(er, ur, kx, ky)
    n_half = kx.shape[0] * 2
    return build_scatter_from_WV(
        W[:n_half, :n_half], W[n_half:, :n_half],
        W0[:n_half, :n_half], W0[n_half:, :n_half], 
        LAM[:n_half], k0L
    )


def build_scatter_from_omega2(EH_mat: np.ndarray, HE_mat: np.ndarray, W0: np.ndarray, k0L: float) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Builds the S-matrix from the squared Omega format (P*Q), suitable for isotropic media.
    """
    omega2 = EH_mat @ HE_mat
    LAM2, W = np.linalg.eig(omega2)
    LAM = np.sqrt(LAM2)
    # Ensure physical branch selection (Negative real part for forward propagation)
    LAM = -np.abs(LAM.real) + 1j * np.abs(LAM.imag)
    
    V = HE_mat @ W @ np.diag(1 / LAM)
    n_half = W0.shape[0] // 2
    return build_scatter_from_WV(W, V, W0[:n_half, :n_half], W0[n_half:, :n_half], LAM, k0L)


def build_scatter_side(er: Any, ur: Any, kx: np.ndarray, ky: np.ndarray, 
                       W0: np.ndarray, transmission_side: bool = False) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """
    Constructs an semi-infinite S-matrix for a boundary interface.
    """
    LAM, W = homogeneous_isotropic_matrix(er, ur, kx, ky)
    A, B = (W0, W) if transmission_side else (W, W0)
    return build_scatter_from_AB(A, B), W, LAM


def get_field_incide(c1p: np.ndarray, c1m: np.ndarray, 
                     c3p: np.ndarray, c3m: np.ndarray, 
                     A: Tuple[np.ndarray, ...], B: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the internal mode coefficients (c2p, c2m) from a cascaded system.
    
    Logic: c1p -> A -> c2p -> B -> c3p
           c1m <- A <- c2m <- B <- c3m
    """
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(a11.shape[0])
    
    # Internal coefficients via matching through the A and B blocks
    c2p = np.linalg.solve(I - a22 @ b11, a21 @ c1p + a22 @ b12 @ c3m)
    c2m = np.linalg.solve(I - b11 @ a22, b12 @ c3m + b11 @ a21 @ c1p)
    
    return c2p, c2m
