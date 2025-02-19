import numpy as np
import tensorflow as tf
from RCWA.Modes import Modes
from RCWA.EigenMode import EigenMode
from RCWA.ScatterMat import ScatterMatBuilder
import matplotlib.pyplot as plt


class Layer:
    def __init__(self):
        pass


class Structure:
    def __init__(self):
        self.layers = []
        pass


def SlantGrating(modes: Modes, Sbuild: ScatterMatBuilder, n1=1, n2=4, ff=0.5, slant_angle=0, depth=0.4, dff=-1, dz=0.02):
    domain = modes.domain
    x, y = domain.get_coordinate(modes.num_modes_x*20, modes.num_modes_y*4)

    S = None

    device = []

    nz = depth//dz+1

    def er_fun(x, z):
        center = -z * tf.tan(slant_angle)
        x = x - center
        x = (x + domain.period_x/2) % domain.period_x-domain.period_x/2
        f = z/depth
        ff2 = ff * (1 + dff) * (1-f) + ff * (1 - dff) * f
        width = domain.period_x * ff2
        mask = np.abs(x) < width/2
        er = (n2**2 - n1**2) * mask + n1**2
        return er

    for z in np.arange(nz) * dz:
        er = er_fun(x, z)
        device.append(er[0])

        eigenmode = EigenMode(modes)
        eigenmode.from_material_er(er)
        S1 = Sbuild.BuildScatter(eigenmode, dz)
        if S is None:
            S = S1
        else:
            S = S@S1

    # plt.imshow(device)
    # plt.show()
    return S
