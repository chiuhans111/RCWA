import numpy as np
from rcwa import tf, complex_type, calculate_modes, grating_vector, sampling_coordinate, conv_mat, conv_modes, mode_vectors, homogeneous_matrix, build_scatter, coefficient_to_field, get_center_mode, side_scatter_mat, star_product
import matplotlib.pyplot as plt
from material import silver
from tqdm import tqdm
# from rcwa import


def main(wavelength, aoi):
    n, k = silver(wavelength)
    er_silver = (n+1j*k)**2

    n_ref = 1.0
    n_tra = 1.0

    AOI = np.deg2rad(aoi)
    POI = 0
    kx0 = np.cos(POI) * np.sin(AOI) * n_ref
    ky0 = np.sin(POI) * np.sin(AOI) * n_ref
    kz0 = np.cos(AOI) * n_ref

    # definition
    period_x = 0.1
    period_y = 0.1
    # wavelength = 1

    k0 = np.pi*2/wavelength

    gx0, gy0 = grating_vector(wavelength, period_x, period_y)

    # incidence angle (normalized)

    # total harmonics
    harmonics_x = 9
    harmonics_y = 9

    # Modes
    (modes_x, modes_y, modes,
     mode_x, mode_y,
     mode_x_flat, mode_y_flat) = calculate_modes(harmonics_x, harmonics_y)
    conv_mode_x, conv_mode_y = conv_modes(mode_x_flat, mode_y_flat)

    kx, ky = mode_vectors(kx0, ky0, gx0, gy0, mode_x_flat, mode_y_flat)

    # build shape
    nx, ny, x, y = sampling_coordinate(modes_x, modes_y,
                                       period_x, period_y,
                                       over_sampling=4)

    x = x-period_x/2
    y = y-period_y/2
    FF = 0.94
    W = period_x*FF
    H = period_y*FF
    # H = period_y
    er0 = 1
    er1 = er_silver
    mask = tf.logical_and(tf.abs(x) <= W/2, tf.abs(y) <= H/2)
    er = tf.cast(mask, tf.dtypes.float32) * (er1-er0) + er0
    ur = tf.ones([ny, nx])

    er_conv = conv_mat(conv_mode_x, conv_mode_y, er)
    ur_conv = conv_mat(conv_mode_x, conv_mode_y, ur)

    # plt.imshow(np.abs(er_conv))
    # plt.show()

    LAM0, W0, V0 = homogeneous_matrix(1, 1, kx, ky)

    L = 0.4
    S = build_scatter(kx, ky, er_conv, ur_conv, W0, V0, k0*L)

    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(np.abs(S[i]))
    # plt.show()

    SR = side_scatter_mat(n_ref**2, 1, kx, ky, W0, V0)
    ST = side_scatter_mat(n_tra**2, 1, kx, ky, W0, V0,
                          transmission_side=True)

    SG = star_product(SR, S)
    SG = star_product(SG, ST)

    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(np.abs(SR[i]))
    # plt.show()

    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(np.abs(ST[i]))
    # plt.show()

    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(np.abs(SG[i]))
    # plt.show()

    s11, s12, s21, s22 = SG

    center_mode_x, center_mode_y = get_center_mode(harmonics_x, harmonics_y)

    sxr = s11[:, center_mode_x]
    sxt = s21[:, center_mode_x]
    syr = s11[:, center_mode_y]
    syt = s21[:, center_mode_y]

    vec_s = np.array([ky0, -kx0, 0])
    l = np.sqrt(np.sum(vec_s*vec_s))
    if l == 0:
        vec_s = np.array([1, 0, 0])
    else:
        vec_s /= l
    vec_p = np.cross(vec_s, [kx0, ky0, kz0])

    ssr = sxr * vec_s[0] + syr * vec_s[1]
    sst = sxt * vec_s[0] + syt * vec_s[1]
    spr = sxr * vec_p[0] + syr * vec_p[1]
    spt = sxt * vec_p[0] + syt * vec_p[1]

    kzr = tf.sqrt(tf.cast(n_ref**2-kx*kx-ky*ky, complex_type))
    kzt = tf.sqrt(tf.cast(n_tra**2-kx*kx-ky*ky, complex_type))
    R_ratio = tf.math.real(kzr)/kz0
    T_ratio = tf.math.real(kzt)/kz0

    # extent = [-harmonics_x-0.5, harmonics_x +
    #           0.5, -harmonics_y-0.5, harmonics_y+0.5]

    SER = coefficient_to_field(ssr, kx, ky, kzr)
    SET = coefficient_to_field(sst, kx, ky, kzt)
    SR = tf.reshape(tf.reduce_sum(tf.abs(SER)**2, 0)
                    * R_ratio, [modes_y, modes_x])
    ST = tf.reshape(tf.reduce_sum(tf.abs(SET)**2, 0)
                    * T_ratio, [modes_y, modes_x])

    # print(tf.reduce_sum(SR+ST))

    # plt.subplot(2, 2, 1)
    # plt.imshow(SR, vmin=0, vmax=1,
    #         cmap='jet', extent=extent, origin='lower')
    # plt.subplot(2, 2, 2)
    # plt.imshow(ST, vmin=0, vmax=1,
    #         cmap='jet', extent=extent, origin='lower')

    PER = coefficient_to_field(spr, kx, ky, kzr)
    PET = coefficient_to_field(spt, kx, ky, kzt)
    PR = tf.reshape(tf.reduce_sum(tf.abs(PER)**2, 0)
                    * R_ratio, [modes_y, modes_x])
    PT = tf.reshape(tf.reduce_sum(tf.abs(PET)**2, 0)
                    * T_ratio, [modes_y, modes_x])

    # print(tf.reduce_sum(PR+PT))

    # plt.subplot(2, 2, 3)
    # plt.imshow(PR, vmin=0, vmax=1,
    #         cmap='jet', extent=extent, origin='lower')
    # plt.subplot(2, 2, 4)
    # plt.imshow(PT, vmin=0, vmax=1,
    #         cmap='jet', extent=extent, origin='lower')
    # plt.show()

    return SR, ST, PR, PT


THZ = np.linspace(100, 600, 200)
aois = np.linspace(0, 90, 80)[:-1]
results = []

for thz in tqdm(THZ):
    wavelength = 2.99792/thz*1e2
    row = []
    for aoi in aois:
        print(aoi)
        SR, ST, PR, PT = main(wavelength, aoi)
        row.append([np.sum(SR), np.sum(ST), np.sum(PR), np.sum(PT)])
    results.append(row)

results = np.array(results)
np.save('./result2.npy', results)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(['SR', 'ST', 'PR', 'PT'][i])
    plt.pcolormesh(aois, THZ, results[:, :, i], cmap='jet')
    plt.colorbar()
    plt.grid()

plt.tight_layout()
plt.show()
