"""
Rigorous Coupled Wave Analysis (RCWA) - Core Physics and Simulation Engine.

This module provides the primary Simulation orchestrator and the fundamental 
wavevector and intensity calculations for electromagnetic scattering simulations.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from .utils import block_matrix
from .modes import Modes
from .layer import Layer
from .scattermatrix import star_product, get_field_incide
from .utils import logger


# --- Physics Kernel ---

def get_kz(n: Any, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Computes the longitudinal wavevector component (kz) in a medium.
    
    Args:
        n: Refractive index of the medium.
        kx: Transverse wavevector component along X.
        ky: Transverse wavevector component along Y.
        
    Returns:
        kz: Complex longitudinal wavevector component.
    """
    kz = np.sqrt(np.array(n**2 - kx**2 - ky**2, dtype=complex))
    return kz


def get_intensity(Ex: np.ndarray, Ey: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    """
    Computes the electromagnetic intensity (Poynting-like magnitude) of a field mode.
    
    Args:
        Ex, Ey: Transverse electric field components.
        kx, ky: Transverse wavevector components.
        kz: Longitudinal wavevector component.
        
    Returns:
        I: Intensity distribution, masking evanescent components.
    """
    kzr = np.real(kz)
    valid = np.abs(kzr) > 0
    # Boundary case handling for near-evanescent fields
    Ez = -(Ex * kx + Ey * ky) / (kzr + (~valid))
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return I * valid


def get_k_from_angle(AOI: float, POI: float) -> Tuple[float, float, float]:
    """
    Converts spherical incidence angles (AOI, POI) to normalized k-vector components.
    
    Args:
        AOI: Angle of Incidence (radians).
        POI: Polar Angle of Incidence (radians).
        
    Returns:
        (kx0, ky0, kz0): Normalized k-vector components.
    """
    kx0 = np.cos(POI) * np.sin(AOI)
    ky0 = np.sin(POI) * np.sin(AOI)
    kz0 = np.cos(AOI)
    return kx0, ky0, kz0


# --- Simulation Core ---

class Simulation:
    """
    Main orchestrator for RCWA simulations. 
    Handles cascading of S-matrices through a stack of layers.
    """
    def __init__(self, modes: Modes, layers: List[Layer] = [], keep_modes: bool = False):
        """
        Initializes the simulation stack and builds the global scattering matrix.
        
        Args:
            modes: The Modes object defining the Fourier space.
            layers: List of Layer objects forming the simulation stack.
            keep_modes: If True, stores internal mode matrices for field reconstruction.
        """
        self.modes = modes

        # Initialization of reflection and transmission side S-matrices
        Sref, Wref, LAMref = layers[0].build_scatter_side(modes)
        Stra, Wtra, LAMtra = layers[-1].build_scatter_side(
            modes, transmission_side=True)

        Sglobal = Sref
        if keep_modes:
            self.mode_matrices = []
            self.scatter_mats = []

        # Cascade through internal layers
        for layer in layers[1:-1]:
            S, W, LAM = layer.build_scatter(modes)
            if keep_modes:
                self.scatter_mats.append((Sglobal, S))
                self.mode_matrices.append([W, LAM, layer.t])
            Sglobal = star_product(Sglobal, S)

        # Final global S-matrix assembly
        if keep_modes:
            self.scatter_mats.append([Sglobal, Stra])

        Sglobal = star_product(Sglobal, Stra)
        self.Sglobal = Sglobal

        # Pre-compute kvectors for reference/transmission regions
        kx, ky = modes.kx, modes.ky
        n_ref, n_tra = layers[0].n, layers[-1].n

        self.kz_ref = get_kz(n_ref, kx, ky)
        self.kz_tra = get_kz(n_tra, kx, ky)
        self.kz_ref_0 = get_kz(n_ref, self.modes.kx0, self.modes.ky0)

        self.Wref = Wref
        self.Wtra = Wtra

    def run(self, Ex: float, Ey: float) -> 'Simulation':
        """
        Executes the simulation for a specific incident polarization (Ex, Ey).
        
        Args:
            Ex, Ey: Complex incident electric field amplitudes.
            
        Returns:
            self: Updated simulation object with computed reflection/transmission coefficients.
        """
        modes = self.modes
        delta = (modes.mx == 0) * (modes.my == 0)
        self.C_inc = np.concatenate([Ex * delta, Ey * delta], axis=0)[:, None]
        self.C_ref = self.Sglobal[0] @ self.C_inc
        self.C_tra = self.Sglobal[2] @ self.C_inc
        return self

    def compute_mode_coefficients(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Unwraps the global scattering matrix to find internal layer-by-layer mode coefficients.
        
        Returns:
            c_modes: List of (forward, backward) coefficients for each layer.
        """
        c1p = self.C_inc  # Incidence
        c1m = self.C_ref  # Reflection
        c3p = self.C_tra  # Transmission
        c3m = np.zeros_like(c1p) # No backward wave from transmission side
        c_modes = []

        for A, B in self.scatter_mats[::-1]:
            c2p, c2m = get_field_incide(c1p, c1m, c3p, c3m, A, B)
            c_modes.append((c2p, c2m))
            c3p = c2p
            c3m = c2m
        
        return c_modes[::-1]

    def get_internal_field(self, dz: float = 0.05, mode_mask: Optional[np.ndarray] = None,
                           mode_mask_seperate_direction: bool = True) -> Tuple[List[float], List[np.ndarray]]:
        """
        Reconstructs the spatial electric field distribution throughout the stack.
        
        Args:
            dz: Resolution along the Z-axis (um).
            mode_mask: Optional mask to isolate specific diffraction orders.
            
        Returns:
            (zs, fields): Z-positions and corresponding electric field slices.
        """
        n_modes = self.modes.n_modes
        W0 = self.modes.W0
        k0 = self.modes.k0

        c_modes = self.compute_mode_coefficients()
        fields, zs, z0 = [], [], 0
        c_modes = np.array(c_modes)

        for i in range(len(c_modes)-1):
            W, LAM, L = self.mode_matrices[i]
            c1p, c1m = c_modes[i]
            c2p, c2m = c_modes[i+1]
            nz = int(np.ceil(L / dz))

            WiW0 = np.linalg.solve(W, W0)
            C1 = WiW0 @ np.concatenate([c1p, c1m], axis=0)
            C2 = WiW0 @ np.concatenate([c2p, c2m], axis=0)

            if mode_mask is not None:
                if mode_mask_seperate_direction:
                    order1 = np.argsort(np.argsort(-np.abs(C1[:, 0][:n_modes*2])))
                    order2 = np.argsort(np.argsort(-np.abs(C1[:, 0][n_modes*2:])))
                    order = np.concatenate([order1, order2 + n_modes*2])
                else:
                    order = np.argsort(np.argsort(-np.abs(C1[:, 0])))
                C1 = C1 * mode_mask[order][:, None]
                C2 = C2 * mode_mask[order][:, None]

            for z in (np.arange(nz) + 0.5) / nz * L:
                modep = W[:, :n_modes * 2] @ (np.exp(LAM[:n_modes*2] * k0 * z)[:, None] * C1[:n_modes*2])
                modem = W[:, n_modes*2:] @ (np.exp(LAM[n_modes*2:] * k0 * (z - L))[:, None] * C2[n_modes*2:])
                fields.append((modep + modem)[:, 0])
                zs.append(z + z0)
            z0 += L
            
        return zs, fields

    def render_fields(self, points_x: int, points_y: int, fields: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts mode-space fields into spatial/real-space distributions via IFFT.
        
        Args:
            points_x, points_y: Spatial resolution of the output grid.
            fields: List of mode-space field slices.
            
        Returns:
            (px, py, EX, EY): Spatial coordinates and field distributions.
        """
        n_modes_x, n_modes_y = self.modes.shape[1], self.modes.shape[0]
        pad_x, pad_y = points_x - n_modes_x, points_y - n_modes_y

        fields_arr = np.fft.ifftshift(np.reshape(fields, [len(fields), 4, n_modes_y, n_modes_x]), (2, 3))
        hy, hx = (n_modes_y+1)//2, (n_modes_x+1)//2

        # Zero-pad for higher spatial resolution
        fields_padded = block_matrix([
            [fields_arr[:, :, :hy, :hx], np.zeros([len(fields), 4, hy, pad_x]), fields_arr[:, :, :hy, hx:]],
            [np.zeros([len(fields), 4, pad_y, points_x])],
            [fields_arr[:, :, hy:, :hx], np.zeros([len(fields), 4, n_modes_y-hy, pad_x]), fields_arr[:, :, hy:, hx:]],
        ], axis1=3, axis2=2)

        px, py = np.meshgrid((np.arange(points_x)/points_x)*self.modes.period_x, (np.arange(points_y)/points_y)*self.modes.period_y)
        k_phase = np.exp(1j * ((self.modes.kx0 * px + self.modes.ky0 * py) * self.modes.k0))[None] * points_x * points_y

        EX = np.fft.ifft2(fields_padded[:, 0]) * k_phase
        EY = np.fft.ifft2(fields_padded[:, 1]) * k_phase

        return px, py, EX, EY

    def get_efficiency(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the diffraction efficiency for all reflected and transmitted orders.
        
        Returns:
            (R, T): Reflectance and Transmittance arrays per diffraction order.
        """
        n_modes = self.modes.n_modes
        kx, ky = self.modes.kx, self.modes.ky
        kz_ref, kz_tra = self.kz_ref, self.kz_tra

        E_inc, E_ref, E_tra = self.C_inc, self.C_ref, self.C_tra

        I_inc = get_intensity(E_inc[:n_modes, 0], E_inc[n_modes:, 0], kx, ky, kz_ref)
        I_ref = get_intensity(E_ref[:n_modes, 0], E_ref[n_modes:, 0], kx, ky, kz_ref)
        I_tra = get_intensity(E_tra[:n_modes, 0], E_tra[n_modes:, 0], kx, ky, kz_tra)

        I_tot = np.sum(I_inc)
        R = I_ref / I_tot
        T = I_tra / I_tot * np.real(kz_tra) / np.real(self.kz_ref_0)

        logger.debug(f'R_sum = {np.sum(R)}, T_sum = {np.sum(T)}, A = {1-np.sum(R)-np.sum(T)}')
        return R, T


# --- Visualization Helpers ---

def visualize_field(pitch_x: float, pitch_y: float, fields: List[np.ndarray], 
                    n_modes_x: int, n_modes_y: int, points_x: int, points_y: int, 
                    zs: List[float], k0: float, k0x: float, k0y: float, 
                    index_xs: np.ndarray, index_zs: List[float], index_contour: List[np.ndarray]):
    """
    Plots the spatial field distribution with an overlaid index contour.
    """
    pad_x, pad_y = points_x - n_modes_x, points_y - n_modes_y
    fields_arr = np.fft.ifftshift(np.reshape(fields, [len(fields), 4, n_modes_y, n_modes_x]), (2, 3))

    fields_padded = block_matrix([
        [fields_arr[:, :, :n_modes_y//2, :n_modes_x//2], np.zeros([len(fields), 4, n_modes_y//2, pad_x]), fields_arr[:, :, :n_modes_y//2, n_modes_x//2:]],
        [np.zeros([len(fields), 4, pad_y, points_x])],
        [fields_arr[:, :, n_modes_y//2:, :n_modes_x//2], np.zeros([len(fields), 4, n_modes_y-n_modes_y//2, pad_x]), fields_arr[:, :, n_modes_y//2:, n_modes_x//2:]],
    ], axis1=3, axis2=2)

    px, py = np.meshgrid((np.arange(points_x)/points_x)*pitch_x, (np.arange(points_y)/points_y)*pitch_y)
    k_phase = np.exp(1j * ((k0x * px + k0y * py) * k0))[None] * points_x * points_y

    EX = np.fft.ifft2(fields_padded[:, 0]) * k_phase
    EY = np.fft.ifft2(fields_padded[:, 1]) * k_phase

    plt.subplot(1, 2, 1)
    plt.title('EX')
    plt.pcolormesh(px[0], zs, np.real(EX[:, points_y//2, :]), vmin=-2, vmax=2, cmap='RdBu', shading='gouraud')
    plt.axis('equal')
    plt.contour(index_xs, index_zs, index_contour, colors=['k'])

    plt.subplot(1, 2, 2)
    plt.title('EY')
    plt.pcolormesh(px[0], zs, np.real(EY[:, points_y//2, :]), vmin=-2, vmax=2, cmap='RdBu', shading='gouraud')
    plt.axis('equal')
    plt.contour(index_xs, index_zs, index_contour, colors=['k'])


def get_index_contour(layers: List[Any]) -> Tuple[List[float], List[np.ndarray]]:
    """
    Computes coordinates for overlaying the physical structure onto field plots.
    Supports both Layer objects and (er, ur, L) tuples for testing flexibility.
    """
    index_contour, index_zs, z0 = [], [], 0
    for layer in layers:
        # Handle both Layer objects and legacy tuples
        if hasattr(layer, 'er'):
            er, ur, L = layer.er, layer.ur, layer.t
        else:
            er, ur, L = layer
            
        # Compute refractive index from permittivity and permeability
        n = np.sqrt(er * ur)
        
        # Extract a 1D index profile for the contour plot (handling both patterned and homogeneous)
        if np.isscalar(n):
            n_prof = np.array([n, n])
        elif n.ndim == 4:
            # 4D Anisotropic case [3, 3, NY, NX]
            n_prof = np.maximum(
                np.maximum(n[0, 0, n.shape[2]//2, :], n[1, 1, n.shape[2]//2, :]), 
                n[2, 2, n.shape[2]//2, :]
            )
        else:
            # 2D Isotropic case [NY, NX]
            n_prof = n[n.shape[0]//2, :]
            
        index_zs.extend([z0, z0 + L])
        index_contour.extend([n_prof, n_prof])
        z0 += L
        
    return index_zs, index_contour
