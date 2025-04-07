from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from tqdm import trange
"""
This script will export images for visualization !!
"""

x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)

mask = np.abs(x) > 0.5
# mask1 = np.abs(x-0.5) < 0.5
# mask2 = np.abs(x-0.4) < 0.5
# mask3 = np.abs(x-0.3) < 0.5
# mask4 = np.abs(x-0.2) < 0.5

layers = [
    rcwa.Layer(n=1),
    rcwa.Layer(n=1, t=0.5),
    # rcwa.Layer(n=0.54, t=0.5),
    rcwa.Layer(n=np.where(mask, 1, 2), t=0.3),
    # rcwa.Layer(n=np.where(mask1, 1, 2), t=0.1),
    # rcwa.Layer(n=np.where(mask2, 1, 2), t=0.1),
    # rcwa.Layer(n=np.where(mask3, 1, 2), t=0.1),
    # rcwa.Layer(n=np.where(mask4, 1, 2), t=0.1),
    rcwa.Layer(n=2, t=0.5),
    rcwa.Layer(n=2),
]

# define modes
# AOI = np.radians(15)
AOI = np.radians(0)
POI = np.radians(0)

modes = rcwa.Modes(
    wavelength=0.532,
    kx0=0,
    ky0=0,
    period_x=0.4,
    period_y=2,
    harmonics_x=5,
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
E_fields = []

for i in trange(7):
    mode_mask = None
    if i>0:
        mode = [0, 1, 2, 22, 23, 24][i-1]
        mode_mask = np.zeros(44)
        mode_mask[mode] = 1

    zs, fields = simulation.get_internal_field(dz=0.001, mode_mask=mode_mask)
    xs, ys, EX, EY = simulation.render_fields(200, 1, fields)
    E_fields.append(EY)


# output visualization
for i in trange(16):
    phase = np.exp(-1j * i / 16 * np.pi*2)
    plt.figure(figsize=(10, 5))
    imgid = 1

    for EY in E_fields:
        scale = 0.5
        plt.subplot(1, 7, imgid)
        plt.pcolormesh(xs, zs, np.real(EY[:, 0, :] * phase), vmin=-scale, vmax=scale, cmap='RdBu')
        plt.axis('equal')
        plt.margins(0)
        imgid+=1
    plt.tight_layout()
    for j in range(len(E_fields)):
        plt.subplot(1, 7, j+1)
        plt.xlim(np.min(xs), np.max(xs))
        plt.ylim(np.min(zs), np.max(zs))

    # plt.pause(0.1)
    plt.savefig(f'./output/250407 Grating Visualization/{i:04d}.jpg')
    # plt.savefig(f'./output/250407 Grating Visualization 2/{i:04d}.jpg')
    plt.close()
# plt.show()
