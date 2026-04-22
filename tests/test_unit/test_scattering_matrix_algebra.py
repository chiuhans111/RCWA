import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.scattermatrix import star_product, build_scatter_from_omega, build_scatter_from_omega2
from rcwa.modes import Modes, build_omega, build_omega2_isotropic

class TestCascadeInvariants(unittest.TestCase):
    """
    Concept: System Cascading Algebra.
    Verifies that the Redheffer Star Product satisfies the fundamental 
    mathematical invariants required for stable system combination.
    """

    def setUp(self):
        self.dim = 10
        # Create a random S-matrix (4 sub-blocks)
        self.S1 = (np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim),
                   np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim),
                   np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim),
                   np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim))

    def test_star_identity(self):
        """
        Concept: S * Identity_System = S.
        Verify that combining a system with an identity system returns the same system.
        """
        I = np.eye(self.dim)
        Z = np.zeros((self.dim, self.dim))
        # The identity S-matrix is [[0, I], [I, 0]]
        S_id = (Z, I, I, Z)
        
        S_res = star_product(self.S1, S_id)
        
        for i in range(4):
            np.testing.assert_allclose(S_res[i], self.S1[i], atol=1e-12)

    def test_star_associativity(self):
        """
        Concept: (A * B) * C = A * (B * C).
        Order of accumulation should not change the total system response.
        """
        S2 = tuple(np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim) for _ in range(4))
        S3 = tuple(np.random.rand(self.dim, self.dim) + 1j*np.random.rand(self.dim, self.dim) for _ in range(4))
        
        res1 = star_product(star_product(self.S1, S2), S3)
        res2 = star_product(self.S1, star_product(S2, S3))
        
        for i in range(4):
            np.testing.assert_allclose(res1[i], res2[i], atol=1e-12)

    def test_builder_equivalence(self):
        """
        Concept: Builder Cross-Validation.
        Verify that build_omega (General Anisotropic) and build_omega2_isotropic 
        produce consistent results when given an isotropic material.
        """
        wavelength = 0.532
        modes = Modes(wavelength, 0.1, 0.1, 1.0, 1.0, 2, 2)
        n = 1.5
        er_scalar = n**2
        
        # 1. Isotropic path
        er_iso = np.ones((10, 10)) * er_scalar
        ur_iso = np.ones((10, 10))
        P, Q = build_omega2_isotropic(er_iso, ur_iso, modes)
        omega2_from_iso = P @ Q
        
        # 2. General Anisotropic path (constructed as isotropic)
        # er is [3, 3, NY, NX]
        er_aniso = np.zeros((3, 3, 10, 10), dtype=complex)
        for i in range(3):
            er_aniso[i, i] = er_scalar
        ur_aniso = np.zeros((3, 3, 10, 10), dtype=complex)
        for i in range(3):
            ur_aniso[i, i] = 1.0
            
        omega_aniso = build_omega(er_aniso, ur_aniso, modes)
        # In RCWA, omega^2 corresponds to the P@Q in isotropic formulation
        # but omega relates [Ex, Ey, Hx, Hy]. 
        # For isotropic case, it should be consistent.
        # Actually, let's just check if they produce the same S-matrix.
        
        W0 = modes.W0
        k0L = 2*np.pi / wavelength * 0.1
        
        S_aniso, W_a, L_a = build_scatter_from_omega(omega_aniso, W0, k0L)
        S_iso, W_i, L_i = build_scatter_from_omega2(P, Q, W0, k0L)
        
        # Compare S11 (reflection)
        np.testing.assert_allclose(S_aniso[0], S_iso[0], atol=1e-10)
        # Compare S21 (transmission)
        np.testing.assert_allclose(S_aniso[2], S_iso[2], atol=1e-10)

if __name__ == '__main__':
    unittest.main()
