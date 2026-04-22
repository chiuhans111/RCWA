import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.rcwa import get_kz, get_intensity, get_k_from_angle
from rcwa.modes import get_modes, Modes
from rcwa.utils import block_matrix
from rcwa.scattermatrix import build_scatter_from_homo

class TestAPIExtensions(unittest.TestCase):
    """
    Concept: API Completeness.
    Exercises low-level utility functions and internal builders that 
    are usually hidden behind the Simulation/Layer high-level API.
    """

    def test_math_utilities(self):
        # 1. get_kz
        kz = get_kz(1.5, 0.5, 0.5)
        expected_kz = np.sqrt(1.5**2 - 0.5**2 - 0.5**2)
        self.assertAlmostEqual(kz, expected_kz)

        # 2. get_intensity
        I = get_intensity(1, 0, 0, 0, 1) # Normal incidence, unit field
        self.assertEqual(I, 1)

        # 3. get_k_from_angle
        kx, ky, kz = get_k_from_angle(np.radians(30), 0)
        self.assertAlmostEqual(kx, 0.5)
        self.assertAlmostEqual(ky, 0)
        self.assertAlmostEqual(kz, np.cos(np.radians(30)))

    def test_k_vector_physics(self):
        """
        Verifies that get_k_from_angle correctly maps AOI/POI to k-vector components
        while maintaining the spherical energy invariant: kx^2 + ky^2 + kz^2 = 1.
        """
        aoi = np.radians(45)
        poi = np.radians(30)
        
        # Calculate components
        kx, ky, kz = get_k_from_angle(aoi, poi)
        
        # 1. Plane of Incidence check: tan(POI) = ky / kx
        expected_poi = np.arctan2(ky, kx)
        self.assertAlmostEqual(poi, expected_poi)
        
        # 2. Angle of Incidence check: sin(AOI) = sqrt(kx^2 + ky^2)
        k_parallel = np.sqrt(kx**2 + ky**2)
        computed_aoi = np.arcsin(k_parallel)
        self.assertAlmostEqual(aoi, computed_aoi)
        
        # 3. Spherical invariant: kx^2 + ky^2 + kz^2 = 1 (unit radius)
        self.assertAlmostEqual(kx**2 + ky**2 + kz**2, 1.0, places=12)

    def test_mode_generation(self):
        # get_modes should return expected shapes
        hx, hy = 2, 1
        mx, my, shape = get_modes(hx, hy)
        # (2*2 + 1) * (2*1 + 1) = 5 * 3 = 15 total modes
        self.assertEqual(mx.size, 15)
        self.assertEqual(shape, (3, 5))

    def test_block_matrix_utility(self):
        a = np.ones((2, 2))
        b = np.zeros((2, 2))
        res = block_matrix([[a, b], [b, a]])
        self.assertEqual(res.shape, (4, 4))
        self.assertEqual(res[0, 0], 1)
        self.assertEqual(res[0, 3], 0)

    def test_specialized_builders(self):
        # build_scatter_from_homo
        # For N=1 mode, Internal RCWA expects 4N x 4N matrices (Ex, Ey, Hx, Hy for each mode)
        kx, ky = np.array([0.1]), np.array([0.0])
        n_modes = 1 
        W0 = np.eye(n_modes * 4, dtype=complex) # Should be 4x4 for N=1
        
        S, W, LAM = build_scatter_from_homo(2.25, 1.0, kx, ky, W0, 0.1)
        
        # S-matrix should be a 4-tuple
        self.assertEqual(len(S), 4)
        # Dimensions should be correct for N=1 (2 polarizations * 1 mode)
        self.assertEqual(S[0].shape, (2, 2)) 
        
if __name__ == '__main__':
    unittest.main()
