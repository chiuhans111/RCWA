import tensorflow as tf
from RCWA.Domain import Domain
from RCWA.Modes import Modes
from RCWA.EigenMode import EigenMode
from RCWA.ScatterMat import ScatterMatBuilder
from RCWA.Device import SlantGrating
from RCWA import Utils
import numpy as np
import matplotlib.pyplot as plt
import time

# Reference:
# [1] M. A. Golub and A. A. Friesem,
# “Effective grating theory for resonance domain surface-relief diffraction gratings,”
# J. Opt. Soc. Am. A, vol. 22, no. 6, p. 1115, Jun. 2005, doi: 10.1364/JOSAA.22.001115.

period = 0.357

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


def run(AOI, dff):
    modes.set_incidence_AOI_POI(
        AOI=np.deg2rad(AOI),
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
    S = SlantGrating(modes, sbuilder, n1=n1, n2=n2,
                     ff=0.5,
                     slant_angle=np.deg2rad(35),
                     depth=0.8,
                     dff=dff,
                     dz=0.02)

    Sglobal = Sglobal @ S
    Sglobal = Sglobal @ Strn

    # Incidence
    delta = (modes.mx == 0)*(modes.my == 0)

    def incidence(S, pol_angle_deg):
        # Set amplitudes for s and p polarizations
        pol_angle = np.deg2rad(pol_angle_deg)

        amp_s = tf.sin(pol_angle)  # 90 degree, TE
        amp_p = tf.cos(pol_angle)  # 0 degree, TM

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

    Eref, Etrn, R_TM, T_TM = incidence(Sglobal, 0)
    Eref, Etrn, R_TE, T_TE = incidence(Sglobal, 90)

    print(np.sum(R_TM+T_TM))
    print("Error:", 1-np.sum(R_TM+T_TM))
    print(np.sum(R_TE+T_TE))
    print("Error:", 1-np.sum(R_TE+T_TE))

    return R_TM, T_TM, R_TE, T_TE


tick = time.time()
AOIs = np.linspace(-30, 30, 61)
tm = []
te = []
tm2 = []
te2 = []

simulations = 0
for AOI in AOIs:
    simulations+=1
    R_TM, T_TM, R_TE, T_TE = run(AOI, dff=-1)
    tm.append(np.sum(T_TM * (modes.mx == -1) * (modes.my == 0)))
    te.append(np.sum(T_TE * (modes.mx == -1) * (modes.my == 0)))
    R_TM, T_TM, R_TE, T_TE = run(AOI, dff=0)
    tm2.append(np.sum(T_TM * (modes.mx == -1) * (modes.my == 0)))
    te2.append(np.sum(T_TE * (modes.mx == -1) * (modes.my == 0)))

tock = time.time()
print("seconds", tock-tick)
print("simulations", simulations)

np.save(f'./save/in_couple_tm_fine', tm)
np.save(f'./save/in_couple_te_fine', te)
np.save(f'./save/in_couple_tm2_fine', tm2)
np.save(f'./save/in_couple_te2_fine', te2)

plt.figure(figsize=(5, 4), dpi=150)
plt.plot(AOIs, tm, label='Triangular')
plt.plot(AOIs, tm2, label='Parallelogram')
plt.grid()
plt.ylabel('Diffraction Efficiency, 1st order')
plt.xlabel('Incidence Angle')
plt.legend()
plt.tight_layout()
plt.savefig('./result/in_coupling_fine_TM.png')
plt.show()


plt.figure(figsize=(5, 4), dpi=150)
plt.plot(AOIs, te, label='Triangular')
plt.plot(AOIs, te2, label='Parallelogram')
plt.grid()
plt.ylabel('Diffraction Efficiency, 1st order')
plt.xlabel('Incidence Angle')
plt.legend()
plt.tight_layout()
plt.savefig('./result/in_coupling_fine_TE.png')
plt.show()
