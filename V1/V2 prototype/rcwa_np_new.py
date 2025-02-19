import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# This is still prototype....

def block_matrix(arrays, axis1=1, axis2=0):
    # block matrix but with specific concatenate axis
    return np.concatenate([
        np.concatenate(sub_array, axis=axis1)
        for sub_array in arrays
    ], axis=axis2)


def get_modes(harmonics_x, harmonics_y):
    mx = np.arange(-harmonics_x, harmonics_x+1)
    my = np.arange(-harmonics_y, harmonics_y+1)
    mx, my = np.meshgrid(mx, my)
    modes = mx.shape
    mx = mx.flatten()
    my = my.flatten()
    return mx, my, modes


def convolution_matrix(u, mx, my, inv=False):
    # build convolution matrix for the last 2 axis of u
    uf = np.fft.fft2(u) / u.shape[-1] / u.shape[-2]
    ix = mx[:, None] - mx[None, :]
    iy = my[:, None] - my[None, :]
    cm = uf[..., iy, ix]
    return cm


def build_omega(er, ur, kx, ky, kz, mx, my):
    # build omega matrix (the main differential equation: d/dz phi = omega phi)
    # Not yet applied Li's factorization rule, still studying it.
    kxd = np.diag(kx)
    kyd = np.diag(ky)
    kzd = np.diag(np.concatenate([kz, kz]))

    k1term = 1j * np.concatenate([kxd, kyd], axis=0)  # T
    k2term = 1j * np.concatenate([-kyd, kxd], axis=1)

    e1term = np.array([er[1, 2], -er[0, 2]])  # T
    u1term = np.array([ur[1, 2], -ur[0, 2]])  # T

    e2term = np.array([er[2, 0], er[2, 1]])
    u2term = np.array([ur[2, 0], ur[2, 1]])

    erz = er[2, 2]
    urz = ur[2, 2]

    e_mat = block_matrix(convolution_matrix(np.array([
        [er[1, 0], er[1, 1]],
        [-er[0, 0], -er[0, 1]]
    ]), mx, my))

    u_mat = block_matrix(convolution_matrix(np.array([
        [ur[1, 0], ur[1, 1]],
        [-ur[0, 0], -ur[0, 1]]
    ]), mx, my))

    e1erz = np.concatenate(convolution_matrix(
        e1term/erz[None], mx, my), axis=0)
    u1urz = np.concatenate(convolution_matrix(
        u1term/urz[None], mx, my), axis=0)
    e2erz = np.concatenate(convolution_matrix(
        e2term/erz[None], mx, my), axis=1)
    u2urz = np.concatenate(convolution_matrix(
        u2term/urz[None], mx, my), axis=1)

    u1term = np.concatenate(convolution_matrix(u1term, mx, my), axis=0)
    e1term = np.concatenate(convolution_matrix(e1term, mx, my), axis=0)

    erzi = convolution_matrix(1/erz, mx, my)
    urzi = convolution_matrix(1/urz, mx, my)

    EE_mat = -1j * kzd - k1term @ e2erz + u1urz @ k2term
    EH_mat = k1term @ erzi @ k2term + u_mat - u1term @ e2erz
    HH_mat = -1j * kzd - k1term @ u2urz + e1erz @ k2term
    HE_mat = k1term @ urzi @ k2term + e_mat - e1term @ e2erz

    omega = block_matrix([
        [EE_mat, EH_mat],
        [HE_mat, HH_mat]
    ])

    return omega


def homogeneous_matrix(er, ur, kx, ky):
    n2 = er * ur
    W = np.eye(kx.shape[0]*2)
    kz = np.sqrt((n2-kx**2-ky**2).astype('complex'))
    LAM = np.concatenate([1j * kz, 1j * kz], axis=0)

    V11 = np.diag(kx*ky/kz)
    V12 = np.diag((n2-kx**2)/kz)
    V21 = np.diag((ky**2-n2)/kz)
    V22 = -V11

    V = -1j/ur * block_matrix([
        [V11, V12],
        [V21, V22]
    ])

    W = block_matrix([
        [W, W],
        [V, -V]
    ])

    return LAM, W


def build_scatter_from_AB(A, B):
    # build scatter matrix considering:  A c1 = B c2
    half = A.shape[0]//2
    a11 = A[:half, :half]
    a12 = A[:half, half:]
    a21 = A[half:, :half]
    a22 = A[half:, half:]

    b11 = B[:half, :half]
    b12 = B[:half, half:]
    b21 = B[half:, :half]
    b22 = B[half:, half:]

    S = np.linalg.solve(
        block_matrix([
            [-a12, b11],
            [-a22, b21]
        ]),
        block_matrix([
            [a11, -b12],
            [a21, -b22]
        ])
    )

    s11 = S[:half, :half]
    s12 = S[:half, half:]
    s21 = S[half:, :half]
    s22 = S[half:, half:]

    return (s11, s12, s21, s22)


def build_scatter(omega, W0, k0L):
    # build scatter matrix
    LAM, W = np.linalg.eig(omega)
    order = np.argsort(-np.imag(LAM)*100+np.real(LAM))
    LAM = LAM[order]
    W = W[:, order]
    WiW0 = np.linalg.solve(W, W0)
    A = np.diag(np.exp(LAM*k0L/2)) @ WiW0
    B = np.diag(np.exp(-LAM*k0L/2)) @ WiW0
    return build_scatter_from_AB(A, B), W, LAM


def build_scatter_side(er, ur, kx, ky, W0, transmission_side=False):
    # build scatter matrix for reflection and transmission side
    LAM, W = homogeneous_matrix(er, ur, kx, ky)
    if transmission_side:
        A = W0
        B = W
    else:
        A = W
        B = W0
    return build_scatter_from_AB(A, B), W


def star_product(A, B):
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(a11.shape[0], dtype=a11.dtype)
    Ib11a22 = I - np.matmul(b11, a22)
    Ia22b11 = I - np.matmul(a22, b11)
    s11 = a11 + np.matmul(a12, np.linalg.solve(Ib11a22, np.matmul(b11, a21)))
    s22 = b22 + np.matmul(b21, np.linalg.solve(Ia22b11, np.matmul(a22, b12)))
    s12 = np.matmul(a12, np.linalg.solve(Ib11a22, b12))
    s21 = np.matmul(b21, np.linalg.solve(Ia22b11, a21))
    return (s11, s12, s21, s22)


def get_kz(n, kx, ky):
    kz = np.sqrt((n**2-kx**2-ky**2).astype('complex'))
    return kz


def get_intensity(Ex, Ey, kx, ky, kz):
    kzr = np.real(kz)
    valid = np.abs(kzr) >= 1e-3
    Ez = -(Ex*kx + Ey*ky) / (kzr + (~valid))
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return I * valid



def break_field(c1p, c1m, c3p, c3m, A, B):
    # this function breaks the scatter matrix,
    # to get the mode value inside
    # c1p -> A -> c2p -> B -> c3p
    # c1m <- A <- c2m <- B <- c3m
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(*a11.shape)
    c2p = np.linalg.solve(I-a22@b11, a21@c1p+a22@b12@c3m)
    c2m = np.linalg.solve(I-b11@a22, b12@c3m+b11@a21@c1p)
    # Alternative formula, not working
    # c2p = np.linalg.solve(b21, c3p - b22@c3m)
    # c2m = np.linalg.solve(a12, c1m - a11@c1p)
    return c2p, c2m


def simulate_rcwa(
    n_ref=1,
    n_tra=1,
    pitch_x=1,
    pitch_y=1,
    mx=[],
    my=[],
    wavelength=0.532,
    k0x=0,
    k0y=0,
    Ex=0,
    Ey=1,
    layers=[],
):
    # normalize k vector
    k0 = np.pi*2 / wavelength

    # modes
    n_modes = mx.shape[0]

    # normalize g vectors and k vectors
    gx = wavelength / pitch_x
    gy = wavelength / pitch_y

    kx = k0x + gx * mx
    ky = k0y + gy * my
    kz = 0 * mx  # k0z + 0 * mx

    LAM0, W0 = homogeneous_matrix(1, 1, kx, ky)
    Sref, Wref = build_scatter_side(n_ref**2, 1, kx, ky, W0)
    Stra, Wtra = build_scatter_side(
        n_tra**2, 1, kx, ky, W0, transmission_side=True)

    Sglobal = Sref

    modes = []
    scatter_mats = []

    for er, ur, L in layers:
        omega = build_omega(er, ur, kx, ky, kz, mx, my)
        S, W, LAM = build_scatter(omega, W0, k0*L)
        scatter_mats.append((Sglobal, S))
        modes.append([W, LAM, L])
        Sglobal = star_product(Sglobal, S)

    scatter_mats.append([Sglobal, Stra])
    Sglobal = star_product(Sglobal, Stra)
    kz_ref = get_kz(n_ref, kx, ky)
    kz_tra = get_kz(n_tra, kx, ky)

    delta = (mx == 0)*(my == 0)

    C_inc = np.concatenate([Ex*delta, Ey*delta], axis=0)[:, None]
    C_ref = Sglobal[0] @ C_inc
    C_tra = Sglobal[2] @ C_inc

    # break modes (unwrap scatter matrix layer by layer to find the field inside)
    c1p = C_inc
    c1m = C_ref
    c3p = C_tra
    c3m = np.zeros_like(C_inc)
    c_modes = [(c3p, c3m)]
    for A, B in scatter_mats[::-1]:
        c2p, c2m = break_field(c1p, c1m, c3p, c3m, A, B)
        c_modes.append((c2p, c2m))
        c3p = c2p
        c3m = c2m
    c_modes.append((c1p, c1m))
    c_modes = c_modes[::-1]

    # Compute the field inside each layers
    fields = []
    zs = []
    dz = 0.05
    z0 = 0

    c_modes = np.array(c_modes)
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.abs(c_modes[:, 0]), cmap='jet')
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.abs(c_modes[:, 1]), cmap='jet')
    # plt.show()

    for i in range(len(c_modes)-3):
        W, LAM, L = modes[i]
        c1p, c1m = c_modes[i+1]
        c2p, c2m = c_modes[i+2]
        nz = np.ceil(L/dz)

        WiW0 = np.linalg.solve(W, W0)

        C1 = WiW0 @ np.concatenate([c1p, c1m], axis=0)
        C2 = WiW0 @ np.concatenate([c2p, c2m], axis=0)

        for z in np.arange(nz)/nz*L:
            modep = W[:, :n_modes * 2] @ (
                np.exp(LAM[:n_modes*2] * k0*z)[:, None] * C1[:n_modes*2])
            modem = W[:, n_modes*2:] @ (
                np.exp(LAM[n_modes*2:] * k0*(z-L))[:, None] * C2[n_modes*2:])
            mode = modep + modem
            fields.append(mode[:, 0])
            zs.append(z+z0)
        z0 += L

    # Compute diffraction efficiency
    E_inc = Wref[:n_modes*2, :n_modes*2] @ C_inc
    E_ref = Wref[:n_modes*2, n_modes*2:] @ C_ref
    E_tra = Wtra[:n_modes*2, :n_modes*2] @ C_tra

    I_inc = get_intensity(E_inc[:n_modes, 0],
                          E_inc[n_modes:, 0], kx, ky, kz_ref)
    I_ref = get_intensity(E_ref[:n_modes, 0],
                          E_ref[n_modes:, 0], kx, ky, kz_ref)
    I_tra = get_intensity(E_tra[:n_modes, 0],
                          E_tra[n_modes:, 0], kx, ky, kz_tra)

    I = np.sum(I_inc)
    R = I_ref / I
    T = I_tra / I * np.real(kz_tra) / np.real(kz_ref)
    return R, T, zs, fields


def get_k_from_angle(AOI, POI):
    k0x = np.cos(POI) * np.sin(AOI)
    k0y = np.sin(POI) * np.sin(AOI)
    k0z = np.cos(AOI)
    return k0x, k0y, k0z


def visualize_field(pitch_x, pitch_y, fields, n_modes_x, n_modes_y, points_x, points_y, zs, k0,  k0x, k0y, index_xs, index_zs, index_contour):
    pad_x = points_x-n_modes_x
    pad_y = points_y-n_modes_y

    fields = np.fft.ifftshift(np.reshape(
        fields, [len(fields), 4, n_modes_y, n_modes_x]), (2, 3))

    fields = block_matrix([
        [
            fields[:, :, :n_modes_y//2, :n_modes_x//2],
            np.zeros([len(fields), 4, n_modes_y//2, pad_x]),
            fields[:, :, :n_modes_y//2, n_modes_x//2:],
        ],
        [np.zeros([len(fields), 4, pad_y, points_x])],
        [
            fields[:, :, n_modes_y//2:, :n_modes_x//2],
            np.zeros([len(fields), 4, n_modes_y-n_modes_y//2, pad_x]),
            fields[:, :, n_modes_y//2:, n_modes_x//2:],
        ],
    ], axis1=3, axis2=2)

    px = (np.arange(points_x)/points_x)*pitch_x
    py = (np.arange(points_y)/points_y)*pitch_y
    px, py = np.meshgrid(px, py)
    EX = fields[:, 0]
    EX = np.fft.ifft2(EX) * np.exp(1j*((k0x*px+k0y*py) * k0)
                                   )[None] * points_x * points_y
    EY = fields[:, 1]
    EY = np.fft.ifft2(EY) * np.exp(1j*((k0x*px+k0y*py) * k0)
                                   )[None] * points_x * points_y

    plt.subplot(1, 2, 1)
    plt.title('EX')
    plt.pcolormesh(px[0], zs, np.real(
        EX[:, points_y//2, :]), vmin=-2, vmax=2, cmap='RdBu', shading='gouraud')
    plt.axis('equal')
    plt.contour(index_xs, index_zs, index_contour, colors=['k'])

    plt.subplot(1, 2, 2)
    plt.title('EY')
    plt.pcolormesh(px[0], zs, np.real(
        EY[:, points_y//2, :]), vmin=-2, vmax=2, cmap='RdBu', shading='gouraud')
    plt.axis('equal')

    plt.contour(index_xs, index_zs, index_contour, colors=['k'])


def get_index_contour(layers):
    index_contour = []
    index_zs = []

    z0 = 0
    for er, ur, L in layers:
        n = np.sqrt(er*ur)
        n1 = n[0, 0, n.shape[2]//2, :]
        n2 = n[1, 1, n.shape[2]//2, :]
        n3 = n[2, 2, n.shape[2]//2, :]
        n = np.maximum(n1, n2)
        n = np.maximum(n, n3)
        index_zs.append(z0)
        index_contour.append(n)
        z0 += L
        index_zs.append(z0)
        index_contour.append(n)
    return index_zs, index_contour


# Example usage
if __name__ == '__main__':

    depth = 2
    wavelength = 0.532
    AOI = np.deg2rad(20)
    POI = 0
    k0x, k0y, k0z = get_k_from_angle(AOI, POI)

    Ex = 1
    Ey = 1

    n_ref = 1.5
    n_tra = 1

    pitch_x = 4
    pitch_y = 1

    harmonics_x = 10
    harmonics_y = 0
    n_modes_x = harmonics_x*2+1
    n_modes_y = harmonics_y*2+1
    mx, my, modes = get_modes(harmonics_x, harmonics_y)

    # Construct high resolution grid
    samples_x = 256
    samples_y = 1

    x = np.arange(samples_x)/samples_x * pitch_x
    y = np.arange(samples_y)/samples_y * pitch_y

    x, y = np.meshgrid(x, y)

    I = np.ones([1, 1, samples_y, samples_x]) * np.eye(3)[:, :, None, None]

    # Build index profile
    layers = []
    layers.append([I*n_ref**2, I, 1])

    material_1 = np.eye(3) * (1.6)**2
    material_2 = np.eye(3) * (1.8)**2

    width = pitch_x/2
    mask = np.abs(x-pitch_x/2) > width/2
    mask = mask[None, None]
    er = material_1[:, :, None, None] * mask + \
        (1-mask)*material_2[:, :, None, None]
    layers.append([er, I, depth])
    layers.append([I*n_tra**2, I, 1])

    # Plot index profile
    index_zs, index_contour = get_index_contour(layers)
    index_xs = x[0]
    plt.pcolormesh(index_xs, index_zs, index_contour, cmap='jet')
    plt.colorbar()
    plt.contour(index_xs, index_zs, index_contour, colors=['k'])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Index profile')
    plt.tight_layout()
    plt.axis('equal')

    # Plot field distribution
    points_x = 100
    points_y = n_modes_y

    R, T, zs, fields = simulate_rcwa(n_ref, n_tra, pitch_x, pitch_y,
                                     mx, my, wavelength, k0x, k0y, Ex, Ey, layers)

    k0 = np.pi*2/wavelength

    plt.figure()
    plt.suptitle('Field distribution')
    visualize_field(pitch_x, pitch_y, fields, n_modes_x, n_modes_y,
                    points_x, points_y, zs,
                    k0,  k0x, k0y, index_xs, index_zs, index_contour)
    plt.tight_layout()

    # Plot diffraction efficiency
    plt.figure()
    gx = wavelength / pitch_x
    kx = k0x + gx * mx
    tx = np.degrees(np.arcsin(kx))
    plt.plot(tx, R, '-o', label='R')
    plt.plot(tx, T, '-o', label='T')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()
    plt.title('Diffraction Efficiency vs Angle')
    plt.xlabel('Angle')
    plt.ylabel('Diffraction Efficiency')
    plt.tight_layout()
    plt.show()
