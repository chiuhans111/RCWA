import tensorflow as tf
from RCWA.Domain import Domain
from RCWA.Modes import Modes
from RCWA.EigenMode import EigenMode
from RCWA.ScatterMat import ScatterMatBuilder
from RCWA.Device import SlantGrating
from RCWA import Utils
import numpy as np
import matplotlib.pyplot as plt

# Reference:
# [1] M. A. Golub and A. A. Friesem,
# “Effective grating theory for resonance domain surface-relief diffraction gratings,”
# J. Opt. Soc. Am. A, vol. 22, no. 6, p. 1115, Jun. 2005, doi: 10.1364/JOSAA.22.001115.

period = 0.530

n1 = 1
n2 = 1.581

er1 = n1**2
er2 = n2**2
ur1 = 1
ur2 = 1

domain = Domain()
domain.set_period_centered(period, period)

modes = Modes(domain)
modes.set_harmonics(10, 0)


def run(depth):
    modes.set_incidence_AOI_POI(
        AOI=np.deg2rad(30),
        POI=np.deg2rad(0))
    modes.set_wavelength(0.530)

    # Default Matrix
    sbuilder = ScatterMatBuilder(modes)
    ref_mode = EigenMode(modes)
    trn_mode = EigenMode(modes)
    ref_mode.from_homogeneous(er1, ur1)
    trn_mode.from_homogeneous(er2, ur2)
    Sref = sbuilder.BuildScatterRef(ref_mode)
    Strn = sbuilder.BuildScatterTrn(trn_mode)

    Sglobal = Sref

    # Create Device
    if depth>0:
        S = SlantGrating(modes, sbuilder, n1=n1, n2=n2,
                        ff=0.5,
                        slant_angle=np.deg2rad(0),
                        depth=depth,
                        dff=-1,
                        dz=depth/20)

        Sglobal = Sglobal @ S

    Sglobal = Sglobal @ Strn

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
    # plt.subplot(2, 1, 1)
    # plt.title("Reflection")
    # plt.imshow(R_2d, cmap='jet', vmin=0)
    # plt.colorbar()
    # plt.subplot(2, 1, 2)
    # plt.title("Transmission")
    # plt.imshow(T_2d, cmap='jet', vmin=0)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # Calculate the sum of reflection and transmission coefficients
    print(np.sum(R+T))
    print("Error:", 1-np.sum(R+T))

    return R, T


plt.figure(figsize=(6, 5), dpi=150)
t0 = []
t1 = []
depths = np.linspace(0, 4*period, 61)
for depth in depths:
    R, T = run(depth)
    t0.append(np.sum(T * (modes.mx == 0) * (modes.my == 0)))
    t1.append(np.sum(T * (modes.mx == -1) * (modes.my == 0)))
plt.plot(depths/period, t0)
plt.plot(depths/period, t1)
plt.grid()
plt.show()
