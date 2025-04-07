from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from tqdm import trange

"""
This script will show animated visualization
"""

# build structure
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)

mask = np.abs(x) > 0.5

layers = [
    rcwa.Layer(n=1),
    rcwa.Layer(n=1, t=1),
    rcwa.Layer(n=np.where(mask, 1, 2), t=0.15),
    rcwa.Layer(n=2, t=1),
    rcwa.Layer(n=2),
]

# define modes
# AOI = np.radians(15)
AOI = np.radians(10)
POI = np.radians(0)

modes = rcwa.Modes(
    wavelength=0.532,
    kx0=0,
    ky0=0,
    period_x=0.4,
    period_y=2,
    harmonics_x=10,
    harmonics_y=0
)

modes.set_direction(
    kx0=np.cos(POI) * np.sin(AOI)*layers[0].n,
    ky0=np.sin(POI) * np.sin(AOI)*layers[0].n
)

# run simulation
simulation = rcwa.Simulation(
    modes=modes,
    layers=layers,
    keep_modes=True
)

R, T = simulation.run(Ex=0, Ey=1).get_efficiency()

print(f"R = {np.sum(R)}")
print(f"T = {np.sum(T)}")
print(f"A = {1-np.sum(R)-np.sum(T)}")


# visualization

zs, fields = simulation.get_internal_field(dz=0.01)
xs, ys, EX, EY = simulation.render_fields(200, 1, fields)


xs_tile = []
EY_tile = []

for i in range(4):
    xs_tile.append(xs + i*modes.period_x)
    EY_tile.append(EY*np.exp(1j * i * modes.kx0 * modes.k0 * modes.period_x))
    

xs = np.concatenate(xs_tile, axis=1)
EY = np.concatenate(EY_tile, axis=2)

plt.figure()
plt.margins(0)
i = 0
while True:
    i += 1
    i = i % 16
    phase = np.exp(-1j * i / 16 * np.pi*2)

    scale = 1
    plt.clf()
    plt.pcolormesh(xs, zs, np.real(
        EY[:, 0, :] * phase), vmin=-scale, vmax=scale, cmap='RdBu')
    plt.axis('equal')

    # plt.xlim(np.min(xs), np.max(xs))
    # plt.ylim(np.min(zs), np.max(zs))

    plt.pause(0.1)
    # plt.savefig(f'./output/250407 Grating Visualization/{i:04d}.jpg')
    # plt.savefig(f'./output/250407 Grating Visualization 2/{i:04d}.jpg')
# plt.close()
# plt.show()
