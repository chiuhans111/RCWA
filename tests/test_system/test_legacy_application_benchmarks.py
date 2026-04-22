import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ensure we can import rcwa and tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

from rcwa.rcwa import Simulation, Modes, Layer
from tests.golden_manager import GoldenManager

class TestLegacyApplicationBenchmarks(unittest.TestCase):
    """
    Concept: Real Application Demonstrations (Legacy Reproduction).
    Consolidates complex sweeps, convergence studies, and field exports 
    from legacy manual tests into the robust golden verification system.
    """

    def setUp(self):
        self.gm = GoldenManager()

    def test_thickness_sweep_efficiency(self):
        """
        Scenario: Thickness Sweep Reproduction (Legacy test03_isotropic_1D_grating_parameter).
        Verifies that efficiency curves vs grating thickness are stable.
        """
        x = np.linspace(-1, 1, 101)
        mask = x < 0
        
        thickness_range = np.linspace(0, 0.5, 11) # Reduced density for speed, keeping physics
        efficiencies = []

        for t in thickness_range:
            layers = [
                Layer(n=1.0),
                Layer(n=1.0, t=0.1),
                Layer(er=np.where(mask, 1.0**2, 2.0**2), t=t),
                Layer(n=1.5)
            ]
            modes = Modes(0.532, 0, 0, 0.4, 2.0, 5, 0)
            modes.set_direction(kx0=np.sin(np.radians(15)), ky0=0)
            
            sim = Simulation(modes, layers)
            R, T = sim.run(Ex=0, Ey=1).get_efficiency()
            efficiencies.append(np.sum(R))

        # 1. Numerical Check
        self.gm.verify("test_system_thickness_sweep_efficiency", efficiencies)

        # 2. Visual Check
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thickness_range, efficiencies, marker='o')
        ax.set_title("Total Reflection vs Thickness")
        ax.set_xlabel("Thickness (um)")
        ax.set_ylabel("Efficiency")
        self.gm.verify_image("test_system_thickness_sweep_efficiency", fig)
        plt.close(fig)

    def test_angular_response_sweep(self):
        """
        Scenario: Angular Sweep Reproduction (Inspired by Legacy test01_3).
        Verifies the angular response across sub-critical and TIR regimes.
        """
        n1, n2 = 1.5, 1.0
        angles = np.linspace(0, 80, 21)
        reflectivities = []

        for deg in angles:
            aoi = np.radians(deg)
            modes = Modes(0.633, 0, 0, 1.0, 1.0, 0, 0)
            modes.set_direction(kx0=np.sin(aoi) * n1, ky0=0)
            
            layers = [Layer(n=n1), Layer(n=n2)]
            sim = Simulation(modes, layers)
            R, T = sim.run(Ex=1, Ey=0).get_efficiency()
            reflectivities.append(np.sum(R))

        # 1. Numerical Check
        self.gm.verify("test_system_angular_response_sweep", reflectivities)

        # 2. Visual Check
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(angles, reflectivities, 'r-')
        ax.axvline(np.degrees(np.arcsin(n2/n1)), color='k', linestyle='--', label='Critical Angle')
        ax.set_title("Angular Response (R) - Air/Glass")
        ax.set_xlabel("AOI (deg)")
        ax.legend()
        self.gm.verify_image("test_system_angular_response_sweep", fig)
        plt.close(fig)

    def test_fourier_harmonics_convergence(self):
        """
        Scenario: Convergence Study (Inspired by Legacy test04).
        Verifies that the solver converges as the number of Fourier harmonics increases.
        """
        x = np.linspace(0, 1, 101)
        mask = x < 0.7
        
        harmonics_list = [1, 3, 5, 7, 9]
        reflection_order_0 = []

        for h in harmonics_list:
            modes = Modes(0.532, 0, 0, 0.3, 1.0, h, 0)
            modes.set_direction(kx0=np.sin(np.radians(45)) * 2, ky0=0)
            
            layers = [
                Layer(n=2.0),
                Layer(er=np.where(mask, 2.0**2, 1.0**2), t=0.04),
                Layer(n=1.0)
            ]
            
            sim = Simulation(modes, layers)
            R, T = sim.run(Ex=1, Ey=1).get_efficiency()
            # Pick the center (0-th order) reflectivity
            reflection_order_0.append(R[len(R)//2])

        # 1. Numerical Check (The convergence curve itself)
        self.gm.verify("test_system_fourier_harmonics_convergence", reflection_order_0)

        # 2. Visual Check
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(harmonics_list, reflection_order_0, 'g--o')
        ax.set_title("Convergence vs Harmonics")
        ax.set_xlabel("N Harmonics")
        ax.set_ylabel("R0 Efficiency")
        self.gm.verify_image("test_system_fourier_harmonics_convergence", fig) # Matched to method
        plt.close(fig)

    def test_homogeneous_thin_film_refraction(self):
        """
        Scenario: Simple Homogeneous Interface (Legacy test01_1).
        Verifies correct Fresnel-limit reproduction for n=1 to n=2.
        """
        layers = [Layer(n=1), Layer(n=1, t=1), Layer(n=2, t=1), Layer(n=2)]
        modes = Modes(0.532, 0, 0, 2.0, 2.0, 0, 0)
        modes.set_direction(kx0=np.sin(np.radians(45)), ky0=0)
        
        sim = Simulation(modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        results = {"R": float(np.sum(R)), "T": float(np.sum(T))}
        self.gm.verify("test_system_homogeneous_thin_film_refraction", results)

    def test_grating_field_export_decomposition(self):
        """
        Scenario: Grating Field Export (Legacy test03_visualization_export).
        Verifies the field reconstruction for specific mode masks.
        """
        x = np.linspace(-1, 1, 101)
        mask = np.abs(x) > 0.35
        layers = [Layer(n=1), Layer(n=1, t=2), Layer(n=np.where(mask, 1, 2), t=2), Layer(n=2, t=2), Layer(n=2)]
        modes = Modes(0.532, 0, 0, 1.0, 2.0, 1, 0)
        modes.set_direction(kx0=np.sin(np.radians(30)), ky0=0)
        
        sim = Simulation(modes, layers, keep_modes=True)
        sim.run(Ex=0, Ey=1)
        
        # Verify a specific mode-masked field (mode 0)
        mode_mask = np.zeros(12) # (2 harmonics*2 + 1) * 1 * 4 = 12 total internal modes?
        # Actually (2*1+1)*(2*0+1)*4 = 3*1*4 = 12.
        mode_mask[0] = 1
        zs, fields = sim.get_internal_field(dz=0.5, mode_mask=mode_mask)
        px, py, EX, EY = sim.render_fields(20, 2, fields)
        
        # We'll save a summary of EY to ensure field decomposition works without huge files
        results = {
            "EY_sum": float(np.sum(np.abs(EY))),
            "EY_center_sample": EY[0, 0, len(EY[0,0])//2]
        }
        self.gm.verify("test_system_grating_field_export_decomposition", results)

if __name__ == '__main__':
    unittest.main()
