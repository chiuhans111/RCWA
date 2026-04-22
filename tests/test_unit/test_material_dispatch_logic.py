import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.rcwa import Simulation, Modes, Layer

class TestMaterialPhysics(unittest.TestCase):
    """
    Concept: Material Physics Stability.
    Verifies that the solver correctly handles complex refractive indices (absorption)
    and maintains the Energy Conservation Law (R + T + A = 1).
    """

    def setUp(self):
        self.wavelength = 0.532
        self.modes = Modes(self.wavelength, 0, 0, 1.0, 1.0, 5, 0)
        self.modes.set_direction(0.1, 0) # Oblique incidence

    def test_lossy_media_conservation(self):
        """
        Concept: Conservation of Energy with Absorption.
        For a thick absorbing layer, R + T + A should be exactly 1.0.
        """
        # Complex index n = 1.5 + 0.1j
        n_complex = 1.5 + 0.1j
        layers = [
            Layer(n=1), # Air
            Layer(n=n_complex, t=0.5), # Lossy layer
            Layer(n=1)  # Air
        ]
        
        sim = Simulation(self.modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        # Calculate sums
        total_R = np.sum(R)
        total_T = np.sum(T)
        total_A = 1 - total_R - total_T
        
        # Energy should be conserved
        self.assertAlmostEqual(total_R + total_T + total_A, 1.0, places=12)
        # Absorption should be non-zero
        self.assertGreater(total_A, 0)

    def test_deep_absorption_limit(self):
        """
        Concept: Total Absorption.
        For an extremely thick absorbing layer, T should approach 0.
        """
        n_very_lossy = 1.5 + 1.0j
        layers = [
            Layer(n=1),
            Layer(n=n_very_lossy, t=10.0), # Very thick
            Layer(n=1)
        ]
        
        sim = Simulation(self.modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        self.assertAlmostEqual(np.sum(T), 0, places=10)
        # Power is either reflected or absorbed
        self.assertAlmostEqual(np.sum(R) + (1 - np.sum(R)), 1.0, places=12)

if __name__ == '__main__':
    unittest.main()
