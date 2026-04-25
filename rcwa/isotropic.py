"""
Analytical solutions for homogeneous isotropic media.
"""

from typing import Tuple, Any
import numpy as np
from .utils import block_matrix


def homogeneous_isotropic_matrix(er: Any, ur: Any, kx: np.ndarray, ky: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the analytical eigenvalues (LAM) and eigenvectors (W) for a homogeneous isotropic medium.
    
    This function provides the reference mode expansion used at simulation boundaries.
    
    Args:
        er, ur: Relative permittivity and permeability (scalars).
        kx, ky: Transverse wavevector components for all diffraction orders.
        
    Returns:
        (LAM, W): Complex eigenvalues and the block-eigenvector matrix.
    """
    n2 = er * ur
    kz = np.sqrt((n2 - kx**2 - ky**2).astype('complex'))
    
    # Eigenvalues (propagation constants) for forward and backward waves
    LAM = np.concatenate([1j * kz, 1j * kz], axis=0)
    LAM = np.concatenate([LAM, -LAM], axis=0)

    # V matrix: Relates E and H fields via Maxwell's curl equations
    # Derived from Faraday's law in the Fourier domain
    V11 = np.diag(kx * ky / kz)
    V12 = np.diag((n2 - kx**2) / kz)
    V21 = np.diag((ky**2 - n2) / kz)
    V22 = -V11

    V = -1j / ur * block_matrix([
        [V11, V12],
        [V21, V22]
    ])

    # W matrix (Eigenvector matrix): [E; H]
    W_identity = np.eye(kx.shape[0] * 2)
    W = block_matrix([
        [W_identity, W_identity],
        [V, -V]
    ])

    return LAM, W