"""ray.py
"""

import numpy as np

class Ray:
    def __init__(self, p: np.array):
        """
        p: np.array : Position of the ray source
        """
        self.p = p

    def cast(self, dest: np.array):
        """ Cast the ray towards dest
        dest: np.array : Position of the ray destination
        """
        return 0
