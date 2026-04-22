import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.rcwa import Simulation, Modes, Layer

class TestGoldenPhysics(unittest.TestCase):
    """
    Concept: Optical Invariants (Golden Scenarios).
    Verifies that the simulation produces correct results for fundamental
    optical systems with known analytical solutions.
    """

    def setUp(self):
        self.wavelength = 0.532
        self.k0 = 2 * np.pi / self.wavelength

    def test_brewster_window(self):
        """
        Scenario: The Brewster Window.
        At the Brewster angle for an air-glass interface, the reflection 
        of p-polarized light should be zero.
        """
        n1 = 1.0 # Air
        n2 = 1.5 # Glass
        theta_b = np.arctan(n2 / n1)
        
        modes = Modes(self.wavelength, 0, 0, 1.0, 1.0, 0, 0)
        # p-polarization is in the plane of incidence (kx, kz)
        modes.set_direction(kx0=np.sin(theta_b) * n1, ky0=0)
        
        layers = [Layer(n=n1), Layer(n=n2)]
        sim = Simulation(modes, layers)
        
        # p-polarization (TM) for this setup is Ex=1, Ey=0 (since incidence is in XZ plane)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        # R should be very small
        total_R = np.sum(R)
        self.assertLess(total_R, 1e-4, f"Brewster reflection too high: {total_R}")

    def test_ideal_arc(self):
        """
        Scenario: Ideal Anti-Reflective Coating.
        A quarter-wave layer with n = sqrt(n1*n2) should yield zero reflection.
        """
        n1 = 1.0
        n2 = 1.5
        n_arc = np.sqrt(n1 * n2)
        t_arc = self.wavelength / (4 * n_arc)
        
        modes = Modes(self.wavelength, 0, 0, 1.0, 1.0, 0, 0)
        layers = [Layer(n=n1), Layer(n=n_arc, t=t_arc), Layer(n=n2)]
        
        sim = Simulation(modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        total_R = np.sum(R)
        self.assertLess(total_R, 1e-5, f"ARC reflection too high: {total_R}")

    def test_bragg_mirror(self):
        """
        Scenario: Distributed Bragg Reflector (DBR).
        Stacked high/low index layers should create a high-reflectivity band.
        """
        n_h = 2.0
        n_l = 1.5
        t_h = self.wavelength / (4 * n_h)
        t_l = self.wavelength / (4 * n_l)
        
        modes = Modes(self.wavelength, 0, 0, 1.0, 1.0, 0, 0)
        
        # Create a stack of 10 pairs
        layers = [Layer(n=1.0)] # Air
        for _ in range(10):
            layers.append(Layer(n=n_h, t=t_h))
            layers.append(Layer(n=n_l, t=t_l))
        layers.append(Layer(n=1.5)) # Substrate
        
        sim = Simulation(modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        total_R = np.sum(R)
        # 10 pairs should give high reflection (> 95%)
        self.assertGreater(total_R, 0.95, f"DBR reflection too low: {total_R}")
        # Energy must still be conserved (Lossless)
        self.assertAlmostEqual(total_R + np.sum(T), 1.0, places=12)

if __name__ == '__main__':
    unittest.main()
