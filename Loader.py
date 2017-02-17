"""
author: Melanie Daeschinger
description: Load and Save .npy files
"""

import numpy as np

def load_stack(filename):
    return np.load(filename)

def save_stack(stack, filename):
    np.save(filename, stack)