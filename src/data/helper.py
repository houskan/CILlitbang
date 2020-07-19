import numpy as np
import skimage.color

def discretize(result_cont):
    result_disc = result_cont.copy()
    result_disc[result_disc > 0.5] = 1.0
    result_disc[result_disc <= 0.5] = 0.0
    return result_disc
