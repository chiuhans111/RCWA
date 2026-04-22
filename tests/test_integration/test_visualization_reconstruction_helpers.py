import unittest
import numpy as np
import sys
import os

# Ensure we can import rcwa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rcwa.rcwa import get_index_contour, Layer

class TestVizHelpers(unittest.TestCase):
    """
    Concept: Visualization Utility Validation.
    Verifies the helper functions used for plotting and data post-processing.
    """

    def test_get_index_contour(self):
        """
        Concept: Refractive Index Mapping.
        Verify that get_index_contour correctly extracts and replicates 
        the index profile across layer thicknesses for plotting.
        """
        n1 = 1.5
        n2 = 2.0
        t1 = 0.5
        t2 = 0.3
        
        # Test with simple layers
        # layers input to this helper is expected to be [er, ur, L] list of tuples?
        # Let's check rdwa.py line 293: for er, ur, L in layers:
        # But Simulation.layers are Layer objects. 
        # Actually, let's look at the implementation of get_index_contour in rcwa.py
        
        # It seems it expects tuples or list of lists:
        mock_layers = [
            (np.ones((3, 3, 10, 10)) * n1**2, np.ones((3, 3, 10, 10)), t1),
            (np.ones((3, 3, 10, 10)) * n2**2, np.ones((3, 3, 10, 10)), t2)
        ]
        
        zs, contour = get_index_contour(mock_layers)
        
        # We expect zs to be [0, t1, t1, t1+t2]
        self.assertEqual(zs, [0, t1, t1, t1+t2])
        # We expect contour values to match n1 and n2
        self.assertAlmostEqual(contour[0][0], n1)
        self.assertAlmostEqual(contour[1][0], n1)
        self.assertAlmostEqual(contour[2][0], n2)
        self.assertAlmostEqual(contour[3][0], n2)

if __name__ == '__main__':
    unittest.main()
