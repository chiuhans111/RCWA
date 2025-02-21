from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
logger = logging.getLogger("RCWA")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
# step1 build your structure
x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)

mask = x<0

layers = [
    rcwa.Layer(n=1),
    rcwa.Layer(n=1, t=2),
    # rcwa.Layer(n=1, t=2),
    rcwa.Layer(n=1),
]

# define modes
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
    kx0=np.cos(POI) * np.sin(AOI), 
    ky0=np.sin(POI) * np.sin(AOI)
)

# step3 run your simulation 
simulation = rcwa.Simulation(
    modes=modes,
    layers=layers,
    keep_modes=True
)

R, T = simulation.run(Ex=1, Ey=0).get_efficiency()
# plt.figure()
# plt.plot(R)
# plt.plot(T)


# zs, fields = simulation.get_internal_field(dz=0.01)
# xs, ys, EX, EY = simulation.render_fields(200, 1, fields)
# plt.figure()
# v = np.max(np.abs(EX))
# v = 1
# plt.pcolormesh(xs, zs, np.real(EX[:, 0, :]), vmin=-v, vmax=v)
# plt.axis('equal')
# plt.show()