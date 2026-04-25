from rcwa import rcwa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

R_list = []
T_list = []
A_list = []
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

for harmonics in tqdm(harmonics_list):

    modes = rcwa.Modes(
        wavelength=0.532,
        kx0=0,
        ky0=0,
        period_x=0.3,
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

    R_list.append(R[len(R)//2-1])  # mode 0
    T_list.append(T[len(R)//2-1])  # mode -1
    A_list.append(1 - np.sum(R) - np.sum(T))  # total absorption


R_list = np.array(R_list)
T_list = np.array(T_list)
A_list = np.array(A_list)

R_converge = R_list[-1]

Error = np.abs(R_list - R_converge) / R_converge
# Find the first index where error is below 1%
Converged = np.where(Error < 0.01)[0][0]

print(
    f'Achieving < 1% error at {harmonics_list[Converged]} harmonics, error = {Error[Converged]}.')

plt.plot(harmonics_list, R_list, label='R')
plt.plot(harmonics_list, T_list, label='T')
plt.plot(harmonics_list, A_list, label='A')
plt.axhline(0, c='gray')
plt.axhline(R_list[-1], c='k')
plt.axvline(harmonics_list[Converged], c='r', ls='--')
plt.legend()
plt.xlabel('harmonics')
plt.show()
