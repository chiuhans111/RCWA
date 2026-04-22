import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.modes import Modes, convolution_matrix

class TestFourierLogic(unittest.TestCase):
    """
    Concept: Spectral Representation.
    This suite verifies that spatial material patterns are correctly transformed 
    into frequency-domain coupling operators (Convolution Matrices).
    """

    def setUp(self):
        self.wavelength = 0.532
        self.period_x = 1.0
        self.period_y = 1.0
        self.harmonics_x = 2
        self.harmonics_y = 2
        self.modes = Modes(self.wavelength, 0, 0, self.period_x, self.period_y, 
                           self.harmonics_x, self.harmonics_y)

    def test_convolution_identity(self):
        """
        Logic: Convolution in frequency is multiplication in space.
        conv(A) @ conv(B) should be approximately conv(A * B).
        Note: Exact equality only holds for infinite harmonics, 
        so we check for reasonable convergence.
        """
        # Create two non-homogeneous patterns
        x = np.linspace(0, self.period_x, 100)
        y = np.linspace(0, self.period_y, 100)
        X, Y = np.meshgrid(x, y)
        
        A = 1.5 + 0.5 * np.cos(2 * np.pi * X / self.period_x)
        B = 2.0 + 0.2 * np.sin(2 * np.pi * Y / self.period_y)
        AB = A * B
        
        # Spectral operators
        conv_A = self.modes.convolution_matrix(A)
        conv_B = self.modes.convolution_matrix(B)
        conv_AB_actual = self.modes.convolution_matrix(AB)
        
        conv_AB_approx = conv_A @ conv_B
        
        # Check center region (lower harmonics are more accurate)
        n = conv_AB_actual.shape[0]
        center = n // 2
        # We expect small error in the central coupling coefficients
        np.testing.assert_allclose(conv_AB_approx[center, center], 
                                   conv_AB_actual[center, center], 
                                   rtol=1e-2, err_msg="Convolution identity failed at DC component")

    def test_homogeneous_diagonal(self):
        """
        Concept: A constant material should yield a diagonal convolution matrix.
        """
        constant_val = 2.25
        A = np.ones((10, 10)) * constant_val
        conv_A = self.modes.convolution_matrix(A)
        
        # Expect diagonal matrix
        expected = np.eye(conv_A.shape[0]) * constant_val
        np.testing.assert_allclose(conv_A, expected, atol=1e-10)

    def test_dimensional_shapes(self):
        """
        Concept: Verify that the convolution builder handles scalar-like inputs
        and different permittivity array dimensions (Isotropic/Anisotropic).
        """
        mx = self.modes.mx
        my = self.modes.my
        
        # 2D case (Isotropic spatially varying)
        u2d = np.random.rand(10, 10)
        cm2d = convolution_matrix(u2d, mx, my)
        self.assertEqual(cm2d.shape, (self.modes.n_modes, self.modes.n_modes))
        
        # 3D case (Diagonal Anisotropic spatially varying)
        u3d = np.random.rand(3, 10, 10)
        cm3d = convolution_matrix(u3d, mx, my)
        self.assertEqual(cm3d.shape, (3, self.modes.n_modes, self.modes.n_modes))

if __name__ == '__main__':
    unittest.main()
