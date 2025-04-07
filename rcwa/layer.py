import numpy as np
from .scattermatrix import *
from .modes import *
from .utils import logger


class Layer:
    def __init__(self, **kwargs):
        """
        Initialize a Layer.

        Args:
            n (float or array-like): Refractive index. 
            er (float or array-like): Permittivity. 
            ur (float or array-like): Permeability. 
            t (float): Thickness of the layer.

        n, er, ur Can be a scalar for homogeneous, a matrix with shape [NY, NX] for isotropic, 
        [3, NY, NX] for diagonal anisotropic, or [3, 3, NX, NY] for anisotropic cases. 
        If NX=NY=1, it is assumed to be homogeneous.

        ur is assumed to be 1 if not given
        er is set to n**2 if not given
        """

        self.n = kwargs.get('n', 1)
        self.er = kwargs.get('er', self.n**2)
        self.ur = kwargs.get('ur', 1)
        self.t = kwargs.get('t', 0)
        self.n = np.sqrt(self.er*self.ur)

        if not np.isscalar(self.er):
            if self.er.ndim == 2 or self.er.ndim == 3:
                if np.isscalar(self.ur):
                    self.ur = np.ones_like(self.er) * self.ur
            if self.er.ndim == 4:
                if np.isscalar(self.ur):
                    self.ur = np.ones(self.er.shape[:2])[
                        None, None] * np.eye(3)[:, :, None, None] * self.ur

    def build_scatter_side(self, modes: Modes, transmission_side=False):
        logger.info(
            f"building {'transmission' if transmission_side else 'reflection'} side scatter matrix, er={self.er}, ur={self.ur}")
        return build_scatter_side(self.er, self.ur, modes.kx, modes.ky, modes.W0, transmission_side=transmission_side)

    def build_scatter(self, modes: Modes):
        if np.isscalar(self.er):
            logger.info(
                f'building homogeneous layer, er={self.er}, ur={self.ur}')
            return build_scatter_from_homo(self.er, self.ur, modes.kx, modes.ky, modes.W0, modes.k0 * self.t)
        if self.er.ndim == 2:
            logger.info('building isotropic layer')
            P, Q = build_omega2_isotropic(self.er, self.ur, modes)
            return build_scatter_from_omega2(P, Q, modes.W0, modes.k0*self.t)
        if self.er.ndim == 3:
            logger.info('building diagonal anisotropic layer')
            P, Q = build_omega2_diagonal(self.er, self.ur, modes)
            return build_scatter_from_omega2(P, Q, modes.W0, modes.k0*self.t)
        if self.er.ndim == 4:
            logger.info('building anisotropic layer')
            omega = build_omega(self.er, self.ur, modes)
            return build_scatter_from_omega(omega, modes.W0, modes.k0*self.t)
