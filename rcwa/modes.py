"""
Fourier Space and Mode Management for RCWA simulations.
"""

from typing import Tuple, List, Optional
import numpy as np
from .isotropic import homogeneous_isotropic_matrix
from .utils import block_matrix


def get_modes(harmonics_x: int, harmonics_y: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Generates a flattened grid of diffraction orders (indices).
    
    Args:
        harmonics_x, harmonics_y: Number of positive harmonics to include.
                                  Total modes = (2*hx + 1) * (2*hy + 1).
                                  
    Returns:
        (mx, my, mode_shape): Flattened indices and the 2D grid shape.
    """
    mx = np.arange(-harmonics_x, harmonics_x + 1)
    my = np.arange(-harmonics_y, harmonics_y + 1)
    mx, my = np.meshgrid(mx, my)
    mode_shape = mx.shape
    return mx.flatten(), my.flatten(), mode_shape


def convolution_matrix(u: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """
    Constructs a Toeplitz-like convolution matrix (Laurent's Rule) from a spatial distribution.
    
    Args:
        u: Spatial distribution (Permittivity/Permeability) [NY, NX].
        mx, my: Flattened diffraction order indices.
        
    Returns:
        cm: Convolution matrix in Fourier space.
    """
    # Transform to Fourier space and normalize by grid size
    uf = np.fft.fft2(u) / (u.shape[-1] * u.shape[-2])
    
    # Calculate index differences for the Toeplitz mapping
    ix = mx[:, None] - mx[None, :]
    iy = my[:, None] - my[None, :]
    
    # Extract Fourier coefficients corresponding to requested harmonic differences
    cm = uf[..., iy, ix]
    return cm


class Modes:
    """
    Manages the Fourier-space representation of the simulation.
    Maps diffraction orders to physical wavevectors and builds fundamental mode matrices.
    """
    def __init__(self, wavelength: float, kx0: float, ky0: float, 
                 period_x: float, period_y: float, 
                 harmonics_x: int, harmonics_y: int):
        """
        Initializes the mode space.
        
        Args:
            wavelength: Free-space wavelength (um).
            kx0, ky0: Incident wavevector components.
            period_x, period_y: Periodicity of the unit cell (um).
            harmonics_x, harmonics_y: Number of diffraction orders.
        """
        self.wavelength = wavelength
        self.period_x = period_x
        self.period_y = period_y
        
        # Grating vectors (normalized by wavelength)
        self.gx = self.wavelength / self.period_x
        self.gy = self.wavelength / self.period_y
        
        self.harmonics_x = harmonics_x
        self.harmonics_y = harmonics_y
        self.mx, self.my, self.shape = get_modes(harmonics_x, harmonics_y)
        self.n_modes = self.mx.size
        self.k0 = 2 * np.pi / self.wavelength
        
        self.set_direction(kx0, ky0)

    def set_direction(self, kx0: float, ky0: float):
        """
        Updates the incident wavevector and re-computes the fundamental mode matrix (W0).
        """
        self.kx0 = kx0
        self.ky0 = ky0
        # Floquet-Bloch diffraction orders
        self.kx = self.kx0 + self.gx * self.mx
        self.ky = self.ky0 + self.gy * self.my
        
        # Fundamental Homogeneous mode matrix (Free space)
        self.LAM0, self.W0 = homogeneous_isotropic_matrix(1.0, 1.0, self.kx, self.ky)

    def convolution_matrix(self, u: np.ndarray) -> np.ndarray:
        """
        Utility for building convolution matrices using the internal mode set.
        """
        return convolution_matrix(u, self.mx, self.my)


# --- Omega Matrix Construction (Physical State-Space Kernel) ---

def build_omega(er: np.ndarray, ur: np.ndarray, modes: Modes) -> np.ndarray:
    """
    Builds the Omega matrix for general anisotropic media.
    Reference: Li, "New formulation of the Fourier modal method for crossed-surface gratings", JOSA A (1997).
    """
    kxd, kyd = np.diag(modes.kx), np.diag(modes.ky)

    k1term = 1j * np.concatenate([kxd, kyd], axis=0)
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    # Permittivity/Permeability components and their inverses (Li's Rule)
    erz, urz = er[2, 2], ur[2, 2]
    erzi = modes.convolution_matrix(1 / erz)
    urzi = modes.convolution_matrix(1 / urz)

    # Construct sub-matrix convolutions
    e_mat = block_matrix(modes.convolution_matrix(np.array([
        [er[1, 0], er[1, 1]], [-er[0, 0], -er[0, 1]]
    ])))
    u_mat = block_matrix(modes.convolution_matrix(np.array([
        [ur[1, 0], ur[1, 1]], [-ur[0, 0], -ur[0, 1]]
    ])))

    # Coupling terms (Convolution matrices of component ratios)
    e1term_arr = np.array([er[1, 2], -er[0, 2]])
    u1term_arr = np.array([ur[1, 2], -ur[0, 2]])
    e2term_arr = np.array([er[2, 0], er[2, 1]])
    u2term_arr = np.array([ur[2, 0], ur[2, 1]])

    # Pre-compute specific Fourier convolutions required by Li's rules
    e1term = np.concatenate(modes.convolution_matrix(e1term_arr), axis=0)
    u1term = np.concatenate(modes.convolution_matrix(u1term_arr), axis=0)
    
    e1erz = np.concatenate(modes.convolution_matrix(e1term_arr / erz[None]), axis=0)
    u1urz = np.concatenate(modes.convolution_matrix(u1term_arr / urz[None]), axis=0)
    e2erz = np.concatenate(modes.convolution_matrix(e2term_arr / erz[None]), axis=1)
    u2urz = np.concatenate(modes.convolution_matrix(u2term_arr / urz[None]), axis=1)

    # State-space sub-matrices
    EE_mat = - k1term @ e2erz + u1urz @ k2term
    EH_mat = k1term @ erzi @ k2term + u_mat - u1term @ e2erz
    HH_mat = - k1term @ u2urz + e1erz @ k2term
    HE_mat = k1term @ urzi @ k2term + e_mat - e1term @ e2erz

    return block_matrix([[EE_mat, EH_mat], [HE_mat, HH_mat]])


def build_omega2_diagonal(er: np.ndarray, ur: np.ndarray, modes: Modes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the P, Q matrices for diagonal anisotropic media.
    """
    kxd, kyd = np.diag(modes.kx), np.diag(modes.ky)
    k1term = 1j * np.concatenate([kxd, kyd], axis=0) 
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    erz, urz = er[2], ur[2]
    zero = np.zeros_like(erz)

    e_mat = block_matrix(modes.convolution_matrix(np.array([[zero, er[1]], [-er[0], zero]])))
    u_mat = block_matrix(modes.convolution_matrix(np.array([[zero, ur[1]], [-ur[0], zero]])))

    erzi, urzi = modes.convolution_matrix(1 / erz), modes.convolution_matrix(1 / urz)

    return (k1term @ erzi @ k2term + u_mat), (k1term @ urzi @ k2term + e_mat)


def build_omega2_isotropic(er: np.ndarray, ur: np.ndarray, modes: Modes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the P, Q matrices for isotropic patterned media (Li's Factorization Rules).
    """
    kxd, kyd = np.diag(modes.kx), np.diag(modes.ky)
    k1term = 1j * np.concatenate([kxd, kyd], axis=0) 
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    erc, urc = modes.convolution_matrix(er), modes.convolution_matrix(ur)
    erci, urci = modes.convolution_matrix(1 / er), modes.convolution_matrix(1 / ur)

    zero = np.zeros_like(erc)
    e_mat = block_matrix([[zero, erc], [-erc, zero]])
    u_mat = block_matrix([[zero, urc], [-urc, zero]])

    return (k1term @ erci @ k2term + u_mat), (k1term @ urci @ k2term + e_mat)
