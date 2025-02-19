import numpy as np
import matplotlib.pyplot as plt
from .utils import block_matrix
from .modes import Modes
from .layer import Layer
from .matrix_equation import homogeneous_isotropic_matrix
from .scattermatrix import star_product, get_field_incide


def get_kz(n, kx, ky):
    kz = np.sqrt((n**2-kx**2-ky**2).astype('complex'))
    return kz


def get_intensity(Ex, Ey, kx, ky, kz):
    kzr = np.real(kz)
    valid = np.abs(kzr) > 0
    Ez = -(Ex*kx + Ey*ky) / (kzr + (~valid))
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return I * valid


class Simulation:
    def __init__(self, modes: Modes, layers: list[Layer] = [], keep_modes=False):
        self.modes = modes

        # modes
        Sref, Wref, LAMref = layers[0].build_scatter_side(modes)
        Stra, Wtra, LAMtra = layers[-1].build_scatter_side(
            modes, transmission_side=True)

        Sglobal = Sref
        if keep_modes:
            mode_matrices = [[Wref, LAMref, 0]]
            scatter_mats = []

        for layer in layers[1:-1]:
            S, W, LAM = layer.build_scatter(modes)
            Sglobal = star_product(Sglobal, S)
            if keep_modes:
                scatter_mats.append((Sglobal, S))
                mode_matrices.append([W, LAM, layer.t])

        Sglobal = star_product(Sglobal, Stra)
        self.Sglobal = Sglobal

        if keep_modes:
            scatter_mats.append([Sglobal, Stra])
            mode_matrices.append([Wtra, LAMtra, 0])

            self.scatter_mats = scatter_mats
            self.mode_matrices = mode_matrices

        # kvectors
        kx = modes.kx
        ky = modes.ky

        n_ref = layers[0].n
        n_tra = layers[-1].n

        self.kz_ref = get_kz(n_ref, kx, ky)
        self.kz_tra = get_kz(n_tra, kx, ky)

    def run(self, Ex, Ey):
        modes = self.modes
        delta = (modes.mx == 0) * (modes.my == 0)
        self.C_inc = np.concatenate([Ex*delta, Ey*delta], axis=0)[:, None]
        self.C_ref = self.Sglobal[0] @ self.C_inc
        self.C_tra = self.Sglobal[2] @ self.C_inc

    def get_internal_field(self, dz=0.05):
        n_modes = self.modes.n_modes
        W0 = self.modes.W0
        k0 = self.modes.k0

        # break modes (unwrap scatter matrix layer by layer to find the field inside)
        c1p = self.C_inc
        c1m = self.C_ref
        c3p = self.C_tra
        c3m = np.zeros_like(c1p)
        c_modes = [(c3p, c3m)]
        for A, B in self.scatter_mats[::-1]:
            c2p, c2m = get_field_incide(c1p, c1m, c3p, c3m, A, B)
            c_modes.append((c2p, c2m))
            c3p = c2p
            c3m = c2m
        c_modes.append((c1p, c1m))
        c_modes = c_modes[::-1]

        # Compute the field inside each layers
        fields = []
        zs = []
        z0 = 0

        c_modes = np.array(c_modes)
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.abs(c_modes[:, 0]), cmap='jet')
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.abs(c_modes[:, 1]), cmap='jet')
        # plt.show()

        for i in range(len(c_modes)-3):
            W, LAM, L = self.mode_matrices[i+1]
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
        return zs, fields

    def render_fields(self, points_x, points_y, fields):
        n_modes_x = self.modes.shape[1]
        n_modes_y = self.modes.shape[0]
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

        px = (np.arange(points_x)/points_x)*self.modes.pitch_x
        py = (np.arange(points_y)/points_y)*self.modes.pitch_y
        px, py = np.meshgrid(px, py)

        kx0 = self.modes.kx0
        ky0 = self.modes.ky0
        k0 = self.modes.k0

        EX = fields[:, 0]
        EX = np.fft.ifft2(EX) * np.exp(1j*((kx0*px+ky0*py) * k0)
                                       )[None] * points_x * points_y
        EY = fields[:, 1]
        EY = np.fft.ifft2(EY) * np.exp(1j*((kx0*px+ky0*py) * k0)
                                       )[None] * points_x * points_y

        return EX, EY
    
    def get_efficiency(self):
        n_modes = self.modes.n_modes
        kx = self.modes.kx
        ky = self.modes.ky
        kz_ref = self.kz_ref
        kz_tra = self.kz_tra

        # Compute diffraction efficiency
        E_inc = self.Wref[:n_modes*2, :n_modes*2] @ self.C_inc
        E_ref = self.Wref[:n_modes*2, n_modes*2:] @ self.C_ref
        E_tra = self.Wtra[:n_modes*2, :n_modes*2] @ self.C_tra

        I_inc = get_intensity(E_inc[:n_modes, 0],
                              E_inc[n_modes:, 0], kx, ky, kz_ref)
        I_ref = get_intensity(E_ref[:n_modes, 0],
                              E_ref[n_modes:, 0], kx, ky, kz_ref)
        I_tra = get_intensity(E_tra[:n_modes, 0],
                              E_tra[n_modes:, 0], kx, ky, kz_tra)

        I = np.sum(I_inc)
        R = I_ref / I
        T = I_tra / I * np.real(kz_tra) / np.real(kz_ref)
        return R, T


def get_k_from_angle(AOI, POI):
    kx0 = np.cos(POI) * np.sin(AOI)
    ky0 = np.sin(POI) * np.sin(AOI)
    kz0 = np.cos(AOI)
    return kx0, ky0, kz0


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
