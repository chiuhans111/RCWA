from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

t_list = np.linspace(0, 0.5, 100)
Rp_list = []
Rs_list = []

for t in t_list:
    ## Step 1 build your structure -------------
    layers = [
        rcwa.Layer(n=1),
        rcwa.Layer(n=1.4142, t=t),
        rcwa.Layer(n=2),
    ]

    ## Step 2 define modes -------------
    AOI = np.radians(20)
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

    ## Step 3 run your simulation -------------
    simulation = rcwa.Simulation(
        modes=modes,
        layers=layers,
        keep_modes=True
    )

    Rp, Tp = simulation.run(Ex=1, Ey=0).get_efficiency()
    Rs, Ts = simulation.run(Ex=0, Ey=1).get_efficiency()
    
    Rp_list.append(Rp)
    Rs_list.append(Rs)

# Step 4 visualize the result -------------
plt.axvline(0.532/1.4142/4/np.cos(AOI))

plt.plot(t_list, Rp_list, label='R')
plt.plot(t_list, Rs_list, label='T')
plt.show()