import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.layer import Layer
from rcwa.modes import Modes

class TestLayerInterface(unittest.TestCase):
    """
    Concept: Material Dispatch API.
    Ensures that the Layer class correctly interprets the dimensionality 
    of the input data (permittivity/permeability) and interacts with the 
    Modes object to generate a valid S-matrix.
    """

    def setUp(self):
        self.wavelength = 0.532
        self.modes = Modes(self.wavelength, 0, 0, 1.0, 1.0, 2, 2)

    def test_dispatch_scalar(self):
        """Logic: Float input -> Homogeneous builder."""
        layer = Layer(n=1.5, t=0.1)
        S, W, LAM = layer.build_scatter(self.modes)
        self.assertEqual(len(S), 4)
        # S11 for air-glass interface should not be zero
        self.assertNotAlmostEqual(np.sum(np.abs(S[0])), 0)

    def test_dispatch_2d_isotropic(self):
        """Logic: 2D array -> Isotropic builder."""
        er = np.ones((10, 10)) * 2.25
        layer = Layer(er=er, t=0.1)
        # This should trigger build_omega2_isotropic internally
        S, W, LAM = layer.build_scatter(self.modes)
        self.assertEqual(len(S), 4)

    def test_dispatch_3d_diagonal(self):
        """Logic: 3D array [3, NY, NX] -> Diagonal Anisotropic builder."""
        er = np.ones((3, 10, 10))
        er[0] = 2.25 # exx
        er[1] = 2.0  # eyy
        er[2] = 1.0  # ezz
        layer = Layer(er=er, t=0.1)
        # This should trigger build_omega2_diagonal internally
        S, W, LAM = layer.build_scatter(self.modes)
        self.assertEqual(len(S), 4)

    def test_dispatch_4d_anisotropic(self):
        """Logic: 4D array [3, 3, NY, NX] -> Full Anisotropic builder."""
        er = np.zeros((3, 3, 10, 10))
        for i in range(3):
            er[i, i] = 2.25
        layer = Layer(er=er, t=0.1)
        # This should trigger build_omega internally
        S, W, LAM = layer.build_scatter(self.modes)
        self.assertEqual(len(S), 4)

if __name__ == '__main__':
    unittest.main()
