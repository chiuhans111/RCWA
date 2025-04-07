from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
# step1 build your structure
layers = [
    rcwa.Layer(n=1.5),
    rcwa.Layer(n=1.5, t=1),
    rcwa.Layer(n=1, t=1),
    rcwa.Layer(n=1),
]
# define modes

aoi_list = np.arange(90)

for i, (Ex, Ey, polarization) in enumerate([(1, 0, 'P'), (0, 1, 'S')]):
    R_list = []
    T_list = []

    for aoi in aoi_list:
        AOI = np.radians(aoi)
        POI = np.radians(0)

        modes = rcwa.Modes(
            wavelength=0.532,
            kx0=0,
            ky0=0,
            period_x=2,
            period_y=2,
            harmonics_x=0,
            harmonics_y=0
        )

        modes.set_direction(
            kx0=np.cos(POI) * np.sin(AOI)*layers[0].n,
            ky0=np.sin(POI) * np.sin(AOI)*layers[0].n
        )

        # step3 run your simulation
        simulation = rcwa.Simulation(
            modes=modes,
            layers=layers,
            keep_modes=True
        )

        R, T = simulation.run(Ex=Ex, Ey=Ey).get_efficiency()
        R_list.append(R)
        T_list.append(T)

    R_list = np.array(R_list)
    T_list = np.array(T_list)
    plt.subplot(1, 2, i+1)
    plt.title(f'{polarization}-Polarization')
    plt.plot(aoi_list, R_list, label = 'R')
    plt.plot(aoi_list, T_list, label = 'T')
    plt.plot(aoi_list, R_list + T_list, label = 'Total')
    plt.legend()
plt.show()

