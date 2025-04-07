from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# step1 build your structure
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)

mask = x < 0

layers = [
    rcwa.Layer(n=2),
    rcwa.Layer(n=2, t=1),
    rcwa.Layer(n=np.where(mask, 1, 2), t=1),
    rcwa.Layer(n=1, t=1),
    rcwa.Layer(n=1),
]

# define modes
AOI = np.radians(0)
POI = np.radians(0)

modes = rcwa.Modes(
    wavelength=0.532,
    kx0=0,
    ky0=0,
    period_x=2,
    period_y=2,
    harmonics_x=5,
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

R, T = simulation.run(Ex=1, Ey=0).get_efficiency()

print(f"R = {np.sum(R)}")
print(f"T = {np.sum(T)}")
print(f"A = {1-np.sum(R)-np.sum(T)}")

zs, fields = simulation.get_internal_field(dz=0.01)
xs, ys, EX, EY = simulation.render_fields(200, 1, fields)
plt.figure()
plt.pcolormesh(xs, zs, np.real(EX[:, 0, :]), vmin=-1, vmax=1, cmap='RdBu')
plt.axis('equal')
plt.show()
