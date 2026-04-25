"""
Physical Layer representation for RCWA simulations.
"""

from typing import Union, Dict, Any, Tuple
import numpy as np
from .scattermatrix import (
    build_scatter_side, 
    build_scatter_from_homo, 
    build_scatter_from_omega2, 
    build_scatter_from_omega
)
# From .modes import build_omega2_isotropic, build_omega2_diagonal, build_omega (consolidated below)
from .modes import Modes, build_omega2_isotropic, build_omega2_diagonal, build_omega
from .utils import logger


class Layer:
    """
    Represents a single physical layer in the RCWA stack. 
    Can represent homogeneous, isotropic, or anisotropic materials.
    """
    def __init__(self, **kwargs: Any):
        """
        Initializes a Layer with material and thickness parameters.

        Args:
            n: Refractive index (scalar or array-like). 
            er: Relative permittivity (scalar or array-like). 
            ur: Relative permeability (scalar or array-like). Default is 1.
            t: Thickness of the layer (um). Default is 0.

        Note:
            Input dimensions determine the simulation complexity:
            - Scalar: Homogeneous medium.
            - [NY, NX]: Isotropic patterned medium.
            - [3, NY, NX]: Diagonal anisotropic medium.
            - [3, 3, NY, NX]: Fully anisotropic medium.
        """
        self.n = kwargs.get('n', 1.0)
        self.er = kwargs.get('er', self.n**2)
        self.ur = kwargs.get('ur', 1.0)
        self.t = kwargs.get('t', 0.0)
        
        # Consistent recalculation of index for complex numbers
        self.n = np.sqrt(self.er * self.ur)

        # Dimension-based broadcasting for matrix properties
        if not np.isscalar(self.er):
            self.er = np.array(self.er)
            if self.er.ndim == 1:
                self.er = self.er[None, :]
            
            if self.er.ndim in (2, 3):
                if np.isscalar(self.ur):
                    self.ur = np.ones_like(self.er) * self.ur
            
            if self.er.ndim == 4:
                if np.isscalar(self.ur):
                    # Broadcast scalar permeability to a 3x3 identity matrix for anisotropic cases
                    self.ur = np.eye(3)[:, :, None, None] * np.ones_like(self.er) * self.ur

    def build_scatter_side(self, modes: Modes, transmission_side: bool = False) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Builds the scattering matrix for a semi-infinite boundary (incident or transmission side).
        """
        logger.info(
            f"building {'transmission' if transmission_side else 'reflection'} side scatter matrix, "
            f"er_avg={np.mean(self.er)}, ur_avg={np.mean(self.ur)}")
            
        return build_scatter_side(self.er, self.ur, modes.kx, modes.ky, modes.W0, transmission_side=transmission_side)

    def build_scatter(self, modes: Modes) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Builds the full scattering matrix for the layer based on its material dimensionality.
        """
        if np.isscalar(self.er):
            logger.info(f'building homogeneous layer, er={self.er}, ur={self.ur}')
            return build_scatter_from_homo(self.er, self.ur, modes.kx, modes.ky, modes.W0, modes.k0 * self.t)
            
        if self.er.ndim == 2:
            logger.info('building isotropic layer')
            P, Q = build_omega2_isotropic(self.er, self.ur, modes)
            return build_scatter_from_omega2(P, Q, modes.W0, modes.k0 * self.t)
            
        if self.er.ndim == 3:
            logger.info('building diagonal anisotropic layer')
            P, Q = build_omega2_diagonal(self.er, self.ur, modes)
            return build_scatter_from_omega2(P, Q, modes.W0, modes.k0 * self.t)
            
        if self.er.ndim == 4:
            logger.info('building anisotropic layer')
            omega = build_omega(self.er, self.ur, modes)
            return build_scatter_from_omega(omega, modes.W0, modes.k0 * self.t)
            
        raise ValueError(f"Unsupported permittivity dimensions: {self.er.ndim}")
