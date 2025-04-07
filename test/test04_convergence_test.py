from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

R_list = []
T_list = []
harmonics_list = np.arange(1, 40)


# step1 build your structure
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
x, y = np.meshgrid(x, y)

mask = x < 0.7

layers = [
    rcwa.Layer(n=2),
    rcwa.Layer(n=2, t=1),
    rcwa.Layer(n=np.where(mask, 2, 1), t=0.04),
    rcwa.Layer(n=1, t=1),
    rcwa.Layer(n=1),
]

# define modes
AOI = np.radians(45)
POI = np.radians(30)

for harmonics in harmonics_list:

    modes = rcwa.Modes(
        wavelength=0.532,
        kx0=0,
        ky0=0,
        period_x=0.364,
        period_y=1,
        harmonics_x=harmonics,
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

    R, T = simulation.run(Ex=1, Ey=1).get_efficiency()

    R_list.append(R[len(R)//2])
    T_list.append(R[len(R)//2-1])

plt.plot(harmonics_list, T_list, label='R1')
plt.axhline(0, c='gray')
plt.axhline(T_list[-1], c='k')
plt.legend()
plt.xlabel('harmonics')
plt.show()