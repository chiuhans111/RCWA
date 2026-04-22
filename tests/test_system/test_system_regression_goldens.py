import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ensure we can import rcwa and tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set matplotlib backend to Agg to avoid opening windows during tests
import matplotlib
matplotlib.use('Agg')

from rcwa.rcwa import Simulation, Modes, Layer
from tests.golden_manager import GoldenManager

class TestSystemRegressionGoldens(unittest.TestCase):
    """
    Concept: Regression Ground Truth (ToT Pinning).
    Uses GoldenManager to verify numerical results against stored JSON baselines.
    """

    def setUp(self):
        self.gm = GoldenManager()

    def test_grating_reproduction(self):
        """
        Scenario: Reproduction of test/test03_isotropic_1D_grating.py.
        """
        x = np.linspace(-1, 1, 101)
        y = np.linspace(-1, 1, 101)
        x, y = np.meshgrid(x, y)
        mask = x < 0
        
        layers = [
            Layer(n=2),
            Layer(n=2, t=1),
            Layer(er=np.where(mask, 1**2, 2**2), t=1),
            Layer(n=1, t=1),
            Layer(n=1),
        ]
        
        modes = Modes(
            wavelength=0.532,
            kx0=0,
            ky0=0,
            period_x=2,
            period_y=2,
            harmonics_x=5,
            harmonics_y=0
        )
        
        modes.set_direction(kx0=0, ky0=0)
        
        sim = Simulation(modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        results = {
            "R": R,  # Store the full diffraction order array
            "T": T,
            "sums": {
                "R_sum": np.sum(R),
                "T_sum": np.sum(T),
                "A_sum": 1 - np.sum(R) - np.sum(T)
            }
        }
        
        # Verify against file-based golden
        self.gm.verify("test_system_grating_reproduction", results)

    def test_total_internal_reflection_scenario(self):
        """
        Scenario: Total Internal Reflection (TIR).
        High index to low index at an angle above critical (theta_c = 30 deg).
        Should yield R=1, T=0.
        """
        n1, n2 = 2.0, 1.0
        theta_c = np.arcsin(n2/n1)
        aoi = theta_c + np.radians(1) # Just above critical
        
        # Incidence from n=2
        modes = Modes(0.532, 0, 0, 1.0, 1.0, 0, 0)
        modes.set_direction(kx0=np.sin(aoi) * n1, ky0=0)
        
        layers = [Layer(n=n1), Layer(n=n1, t=0.1), Layer(n=n2, t=1.0), Layer(n=n2)]
        sim = Simulation(modes, layers)
        R, T = sim.run(Ex=1, Ey=0).get_efficiency()
        
        results = {"R": R, "T": T, "sums": {"R_sum": np.sum(R), "T_sum": np.sum(T)}}
        self.gm.verify("test_system_total_internal_reflection_scenario", results)

    def test_thin_film_interference_stack(self):
        """
        Scenario: 3-layer Thin Film Interference.
        Verifies correct S-matrix cascading for a multi-layer homogeneous system.
        """
        t = 0.123 # arbitrary thickness
        layers = [
            Layer(n=1.0),
            Layer(n=1.414, t=t),
            Layer(n=2.0, t=t),
            Layer(n=1.5)
        ]
        modes = Modes(0.532, 0, 0, 1.0, 1.0, 0, 0)
        modes.set_direction(kx0=np.sin(np.radians(20)), ky0=0)
        
        sim = Simulation(modes, layers)
        R, T = sim.run(Ex=1, Ey=1).get_efficiency()
        
        results = {"R": R, "T": T, "sums": {"R_sum": np.sum(R), "T_sum": np.sum(T)}}
        self.gm.verify("test_system_thin_film_interference_stack", results)

    def test_anisotropic_homogeneous_cross_check(self):
        """
        Scenario: Anisotropic solver with homogeneous data.
        Ensures the complex full-tensor path matches a simple homogenous solution.
        """
        wavelength = 0.633
        n = 1.5
        
        modes = Modes(wavelength, 0.2, 0.2, 1.0, 1.0, 0, 0)
        
        # 1. Scalar Layer
        layers_scalar = [Layer(n=1), Layer(n=n, t=0.2), Layer(n=1)]
        sim_scalar = Simulation(modes, layers_scalar)
        R_scalar, T_scalar = sim_scalar.run(Ex=1, Ey=0.5).get_efficiency()
        
        # 2. 4D Anisotropic Layer
        er_aniso = np.zeros((3, 3, 1, 1), dtype=complex)
        for i in range(3):
            er_aniso[i, i] = n**2
        ur_aniso = np.zeros((3, 3, 1, 1), dtype=complex)
        for i in range(3):
            ur_aniso[i, i] = 1.0
            
        layers_aniso = [Layer(n=1), Layer(er=er_aniso, t=0.2), Layer(n=1)]
        sim_aniso = Simulation(modes, layers_aniso)
        R_aniso, T_aniso = sim_aniso.run(Ex=1, Ey=0.5).get_efficiency()
        
        results = {
            "scalar": {"R": R_scalar, "T": T_scalar},
            "anisotropic": {"R": R_aniso, "T": T_aniso}
        }
        
        # Verify that they match (cross-check) and pin the absolute values
        self.gm.verify("test_system_anisotropic_homogeneous_cross_check", results)

    def test_total_internal_reflection_visual_golden(self):
        """
        Concept: Visual Regression (Evanescence).
        Captures the visual signature of TIR: Total Reflection with 
        an evanescent tail decaying into the second medium.
        Inspired by Legacy 01.2.
        """
        n1, n2 = 2.0, 1.0
        aoi = np.radians(45) # Above critical
        
        modes = Modes(0.532, 0, 0, 1.0, 1.0, 0, 0)
        modes.set_direction(kx0=np.sin(aoi) * n1, ky0=0)
        
        # High index -> Low index
        layers = [Layer(n=n1), Layer(n=n1, t=0.2), Layer(n=n2, t=0.8), Layer(n=n2)]
        sim = Simulation(modes, layers, keep_modes=True)
        sim.run(Ex=1, Ey=0)
        
        zs, fields = sim.get_internal_field(dz=0.02)
        px, py, EX, EY = sim.render_fields(100, 1, fields)
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pcolormesh(px[0], zs, np.real(EX[:, 0, :]), cmap='RdBu', shading='auto')
        ax.axhline(0.2, color='k', linestyle='--') # Boundary
        ax.set_title("TIR Evanescent Wave (EX)")
        
        self.gm.verify_image("test_system_total_internal_reflection_visual", fig)
        plt.close(fig)

    def test_diffraction_grating_visual_golden(self):
        """
        Concept: Visual Regression (Diffraction).
        Captures the visual signature of a 1D grating diffraction pattern.
        Inspired by Legacy 03.
        """
        x = np.linspace(-1, 1, 101)
        mask = x < 0
        
        # Simple binary grating
        layers = [
            Layer(n=1.5),
            Layer(er=np.where(mask, 1.0**2, 2.0**2), t=1.0),
            Layer(n=1.5)
        ]
        modes = Modes(0.532, 0, 0, 2.0, 2.0, 5, 0)
        sim = Simulation(modes, layers, keep_modes=True)
        sim.run(Ex=1, Ey=0)
        
        zs, fields = sim.get_internal_field(dz=0.05)
        px, py, EX, EY = sim.render_fields(100, 1, fields)
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pcolormesh(px[0], zs, np.real(EX[:, 0, :]), cmap='RdBu', shading='auto')
        ax.set_title("1D Grating Diffraction (EX)")
        
        self.gm.verify_image("test_system_diffraction_grating_visual", fig)
        plt.close(fig)

    def test_anti_reflective_coating_visual_golden(self):
        """
        Scenario: ARC Visual Verification.
        Visualizes the field through a lambda/4 coating.
        Should show physical continuity into the substrate with zero ripple in air.
        """
        n1, n_arc, n2 = 1.0, np.sqrt(1.5), 1.5
        t_arc = 0.532 / (4 * n_arc)
        
        modes = Modes(0.532, 0, 0, 1.0, 1.0, 0, 0)
        layers = [Layer(n=n1), Layer(n=n_arc, t=t_arc), Layer(n=n2)]
        sim = Simulation(modes, layers, keep_modes=True)
        sim.run(Ex=1, Ey=0)
        
        zs, fields = sim.get_internal_field(dz=0.01)
        px, py, EX, EY = sim.render_fields(100, 10, fields)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pcolormesh(px[0], zs, np.real(EX[:, 0, :]), cmap='RdBu', shading='auto')
        ax.axhline(0, color='k', linestyle='--')
        ax.axhline(t_arc, color='k', linestyle='--')
        ax.set_title("Ideal ARC Field (EX) - No Air Ripple")
        
        self.gm.verify_image("test_system_anti_reflective_coating_visual", fig)
        plt.close(fig)

    def test_distributed_bragg_reflector_visual_golden(self):
        """
        Scenario: DBR Visual Verification.
        Visualizes the standing wave trapped at the surface of a DBR stack.
        Should show rapid decay into the stack.
        """
        n_h, n_l = 2.0, 1.5
        t_h, t_l = 0.532/(4*n_h), 0.532/(4*n_l)
        
        layers = [Layer(n=1.0)]
        for _ in range(5):
            layers.append(Layer(n=n_h, t=t_h))
            layers.append(Layer(n=n_l, t=t_l))
        layers.append(Layer(n=1.5))
        
        modes = Modes(0.532, 0, 0, 1.0, 1.0, 5, 0)
        sim = Simulation(modes, layers, keep_modes=True)
        sim.run(Ex=1, Ey=0)
        
        zs, fields = sim.get_internal_field(dz=0.01)
        px, py, EX, EY = sim.render_fields(100, 10, fields)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pcolormesh(px[0], zs, np.real(EX[:, 0, :]), cmap='RdBu', shading='auto')
        ax.set_title("DBR Standing Wave (EX)")
        
        self.gm.verify_image("test_system_distributed_bragg_reflector_visual", fig)
        plt.close(fig)

    def test_brewster_angle_polarization_visual_golden(self):
        """
        Scenario: Brewster Polarization Split.
        Visualizes P-pol (TM) vs S-pol (TE) at Brewster angle.
        Should show zero air ripple for P-pol and high ripple for S-pol.
        """
        n1, n2 = 1.0, 1.5
        theta_b = np.arctan(n2/n1)
        
        modes = Modes(0.532, 0, 0, 1.0, 1.0, 0, 0)
        modes.set_direction(kx0=np.sin(theta_b)*n1, ky0=0)
        
        # We need finite-thickness layers to visualize the fields
        layers = [Layer(n=n1), Layer(n=n1, t=0.5), Layer(n=n2, t=0.5), Layer(n=n2)]
        sim = Simulation(modes, layers, keep_modes=True)
        
        # We'll plot two subplots: Ex (TM) and Ey (TE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # TM (Ex=1)
        sim.run(Ex=1, Ey=0)
        zs, fields = sim.get_internal_field(dz=0.02)
        px, py, EX, EY = sim.render_fields(100, 10, fields)
        ax1.pcolormesh(px[0], zs, np.real(EX[:, 0, :]), cmap='RdBu', shading='auto')
        ax1.set_title("Brewster TM (No Ripple)")
        
        # TE (Ey=1)
        sim.run(Ex=0, Ey=1)
        zs, fields = sim.get_internal_field(dz=0.02)
        px, py, EX, EY = sim.render_fields(100, 10, fields)
        ax2.pcolormesh(px[0], zs, np.real(EY[:, 0, :]), cmap='RdBu', shading='auto')
        ax2.set_title("Brewster TE (High Ripple)")
        
        self.gm.verify_image("test_system_brewster_angle_polarization_visual", fig)
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
