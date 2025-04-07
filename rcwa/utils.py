import numpy as np
import logging
logging.basicConfig()
logger = logging.getLogger("RCWA")

# Set the logging level
# logger.setLevel(logging.DEBUG)
"""
logging.NOTSET
logging.DEBUG
logging.INFO
logging.WARNING
logging.ERROR
logging.CRITICAL
"""

def block_matrix(arrays, axis1=1, axis2=0):
    # block matrix but with specific concatenate axis
    return np.concatenate([
        np.concatenate(sub_array, axis=axis1)
        for sub_array in arrays
    ], axis=axis2)


