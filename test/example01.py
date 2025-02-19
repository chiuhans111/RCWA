from rcwa import rcwa
import numpy as np

# step1 build your structure
layers = [
    rcwa.Layer(n=1.5),
    rcwa.Layer(n=2, t=1),
    rcwa.Layer(n=np.random.random([10, 10])+1, t=1),
    rcwa.Layer(er=np.random.random([10, 10])+1, t=1),
    rcwa.Layer(n=1.5, t=1),
    rcwa.Layer(n=1),
]

# define modes
AOI = np.radians(0)
POI = np.radians(0)

modes = rcwa.Modes(
    wavelength=0.532,
    kx0=0,
    ky0=0,
    period_x=1,
    period_y=1,
    harmonics_x=10,
    harmonics_y=10
)

modes.set_direction(
    kx0=np.cos(POI) * np.sin(AOI), 
    ky0=np.sin(POI) * np.sin(AOI)
)

simulation = rcwa.Simulation(
    modes=modes,
    layers=layers
)

# step3 run your simulation