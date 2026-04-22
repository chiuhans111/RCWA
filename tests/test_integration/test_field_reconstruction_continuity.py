import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.rcwa import Simulation, Modes, Layer

class TestFieldFeatures(unittest.TestCase):
    """
    Concept: Field Reconstruction Handshake.
    Verifies the interaction between the global solver and the field 
    visualization logic, ensuring physical continuity and correct masking.
    """

    def setUp(self):
        self.wavelength = 0.532
        self.modes = Modes(self.wavelength, 0.1, 0, 1.0, 1.0, 5, 0)

    def test_field_continuity(self):
        """
        Concept: Artificial Boundary Continuity.
        If we split a single layer into two identical sub-layers, 
        the reconstructed field should be continuous across the boundary.
        """
        n = 1.5
        t_total = 0.4
        
        # Scenario A: One thick layer
        layers_a = [Layer(n=1), Layer(n=n, t=t_total), Layer(n=1)]
        sim_a = Simulation(self.modes, layers_a, keep_modes=True)
        sim_a.run(Ex=1, Ey=0)
        zs_a, fields_a = sim_a.get_internal_field(dz=t_total/2)
        
        # Scenario B: Two half-layers
        layers_b = [Layer(n=1), Layer(n=n, t=t_total/2), Layer(n=n, t=t_total/2), Layer(n=1)]
        sim_b = Simulation(self.modes, layers_b, keep_modes=True)
        sim_b.run(Ex=1, Ey=0)
        zs_b, fields_b = sim_b.get_internal_field(dz=t_total/2)
        
        # Compare fields at the center of the total thickness
        # They should match exactly as the split is purely mathematical
        np.testing.assert_allclose(fields_a[1], fields_b[1], atol=1e-10, 
                                   err_msg="Field continuity failed across artificial boundary")

    def test_mode_masking_logic(self):
        """
        Concept: Feature Verification.
        Verify that applying a mode_mask correctly zeros out specific Fourier components
        in the internal field reconstruction.
        """
        layers = [Layer(n=1), Layer(n=1.5, t=0.1), Layer(n=1)]
        sim = Simulation(self.modes, layers, keep_modes=True)
        sim.run(Ex=1, Ey=0)
        
        # Mask all modes (should result in zero field)
        mask = np.zeros(self.modes.n_modes * 4)
        zs, fields = sim.get_internal_field(dz=0.05, mode_mask=mask)
        
        for f in fields:
            np.testing.assert_allclose(f, 0, atol=1e-12)

    def test_rendering_logic(self):
        """
        Concept: Spatial Reconstruction.
        Verify that render_fields executes without error and returns 
        the expected spatial grid dimensions.
        """
        layers = [Layer(n=1), Layer(n=1.5, t=0.1), Layer(n=1)]
        sim = Simulation(self.modes, layers, keep_modes=True)
        sim.run(Ex=1, Ey=0)
        zs, fields = sim.get_internal_field(dz=0.05)
        
        # Grid sizes
        nx, ny = 20, 20
        px, py, EX, EY = sim.render_fields(nx, ny, fields)
        
        self.assertEqual(EX.shape, (len(zs), ny, nx))
        self.assertEqual(px.shape, (ny, nx))
        # Verify that the rendered field is not all zeros
        self.assertGreater(np.sum(np.abs(EX)), 0)

if __name__ == '__main__':
    unittest.main()
