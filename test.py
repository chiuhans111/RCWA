import tensorflow as tf
from src.RCWA.Domain import Domain
from src.RCWA.Modes import Modes
from src.RCWA.EigenMode import EigenMode
from src.RCWA.ScatterMat import ScatterMatBuilder
import numpy as np
from src.RCWA import Utils
import matplotlib.pyplot as plt

period = 0.6

n1 = 1
n2 = 2

er1 = n1**2
er2 = n2**2
ur1 = 1
ur2 = 1

domain = Domain()
domain.set_period_centered(period, period)

modes = Modes(domain)
modes.set_harmonics(10, 0)


modes.set_incidence_AOI_POI(
    AOI=np.deg2rad(10), 
    POI=np.deg2rad(45))
modes.set_wavelength(0.532)

# Create Device
x, y = domain.get_coordinate(modes.num_modes_x*4, modes.num_modes_y*4)
mask = x > 0
er = (er2 - er1) * mask + er1
eigenmode = EigenMode(modes)
eigenmode.from_material_er(er)


plt.figure(figsize=(10, 8))
mode_order = np.argsort(np.abs(modes.mx)+np.abs(modes.my))
# Loop through the range 20
for i in range(6):
    # Create subplots in a 5x4 grid
    plt.subplot(2, 3, i+1)
    eigenmode.vis_mode(mode_order[i//1], i % 2)
plt.tight_layout()
plt.show()

# Default Matrix
sbuilder = ScatterMatBuilder(modes)
ref_mode = EigenMode(modes)
trn_mode = EigenMode(modes)
ref_mode.from_homogeneous(er1, ur1)
trn_mode.from_homogeneous(er2, ur2)
Sref = sbuilder.BuildScatterRef(ref_mode)
Strn = sbuilder.BuildScatterTrn(trn_mode)


# Device Matrix
S = sbuilder.BuildScatter(eigenmode, 0.3)
Sglobal = Sref @ S @ Strn

# Incidence
delta = (modes.mx == 0)*(modes.my == 0)

# Set amplitudes for s and p polarizations
pol_angle = np.deg2rad(90)

amp_s = tf.sin(pol_angle)  # 90 degree
amp_p = tf.cos(pol_angle)  # 0 degree

# Calculate the polarization vector based on s and p amplitudes
pol = modes.pol_vec_p * amp_p + modes.pol_vec_s * amp_s

# Calculate the incident electric field components
Einc = tf.cast(
    tf.concat([delta*pol[0], delta*pol[1]], 0), tf.dtypes.complex128)
Einc_z = delta*pol[2]

# Calculate the incident intensity
Iinc = tf.reduce_sum(np.abs(Einc)**2)+tf.reduce_sum(np.abs(Einc_z)**2)

# Calculate the longitudinal wave vector components
kz_r = tf.sqrt((n1**2-modes.kx**2-modes.ky**2).astype('complex'))
kz_t = tf.sqrt((n2**2-modes.kx**2-modes.ky**2).astype('complex'))


def incidence(S):
    # Calculate the electric field components using the scattering matrix
    Eref = tf.reshape((S.value[0]@Einc[:, None]), [2, -1])
    Etrn = tf.reshape((S.value[2]@Einc[:, None]), [2, -1])

    # Calculate the longitudinal electric field components
    Eref_z = -(Eref[0]*modes.kx+Eref[1]*modes.ky)/kz_r
    Etrn_z = -(Etrn[0]*modes.kx+Etrn[1]*modes.ky)/kz_t

    # Calculate the reflected and transmitted intensities
    Iref = tf.reduce_sum(tf.abs(Eref)**2, 0)+tf.abs(Eref_z)**2
    Itrn = tf.reduce_sum(tf.abs(Etrn)**2, 0)+tf.abs(Etrn_z)**2

    # Calculate the reflection and transmission coefficients
    R = Iref*tf.math.real(kz_r)/modes.k0z/Iinc
    T = Itrn*tf.math.real(kz_t)/modes.k0z/Iinc

    return Eref, Etrn, R, T


Eref, Etrn, R, T = incidence(Sglobal)
# Reshape the reflection and transmission coefficients into 2D arrays
R_2d = tf.reshape(R, [modes.num_modes_y, modes.num_modes_x])
T_2d = tf.reshape(T, [modes.num_modes_y, modes.num_modes_x])

# Plot the reflection and transmission coefficients
plt.subplot(2, 1, 1)
plt.title("Reflection")
plt.imshow(R_2d, cmap='jet', vmin=0)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title("Transmission")
plt.imshow(T_2d, cmap='jet', vmin=0)
plt.colorbar()
plt.tight_layout()
plt.show()

# Calculate the sum of reflection and transmission coefficients
print(np.sum(R+T))
print("Error:", 1-np.sum(R+T))
