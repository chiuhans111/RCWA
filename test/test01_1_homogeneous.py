from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt

"""
This is a basic example for creating two layer of homogeneous material,
Equivalent to calculate the fresnel reflection and transmission
"""

## Step 1 build your structure -------------
layers = [
    rcwa.Layer(n=1),
    rcwa.Layer(n=1, t=1), # this intermediate layer is here for visualization
    rcwa.Layer(n=2, t=1), # this intermediate layer is here for visualization
    rcwa.Layer(n=2),
]

## Step 2 define modes -------------
# the kx0 and ky0 are normalized x and y component of the k-vector
modes = rcwa.Modes(
    wavelength=0.532,
    kx0=0, 
    ky0=0,
    period_x=2,
    period_y=2,
    harmonics_x=0,
    harmonics_y=0
)

# incidence angle
AOI = np.radians(45)
POI = np.radians(0)

# You may modify the kx0 and ky0 using this function
modes.set_direction(
    kx0=np.cos(POI) * np.sin(AOI) * layers[0].n,
    ky0=np.sin(POI) * np.sin(AOI) * layers[0].n
)

## Step 3 run your simulation -------------
simulation = rcwa.Simulation(
    modes=modes,
    layers=layers,
    keep_modes=True # set to True for visualization
)

# A simulation can be re run with different Ex, Ey (polarization)
R, T = simulation.run(Ex=1, Ey=0).get_efficiency()
print(f"R = {R}")
print(f"T = {T}")
print(f"A = {1-R-T}")

# Step 4 visualize the field -------------
zs, fields = simulation.get_internal_field(dz=0.01)
xs, ys, EX, EY = simulation.render_fields(200, 1, fields)
plt.figure()
plt.pcolormesh(xs, zs, np.real(EX[:, 0, :]), vmin=-1, vmax=1, cmap='RdBu')
plt.axis('equal')
plt.show()
