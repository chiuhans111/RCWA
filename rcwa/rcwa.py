import numpy as np
import matplotlib.pyplot as plt
from .utils import block_matrix
from .modes import Modes
from .layer import Layer
from .scattermatrix import star_product, get_field_incide
from .utils import logger

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
            mode_matrices = []
            scatter_mats = []

        for layer in layers[1:-1]:
            S, W, LAM = layer.build_scatter(modes)
            if keep_modes:
                scatter_mats.append((Sglobal, S))
                mode_matrices.append([W, LAM, layer.t])
            Sglobal = star_product(Sglobal, S)


        if keep_modes:
            scatter_mats.append([Sglobal, Stra])
            self.scatter_mats = scatter_mats
            self.mode_matrices = mode_matrices
        
        # sglobal should apply after mat push
        Sglobal = star_product(Sglobal, Stra)
        self.Sglobal = Sglobal

        # kvectors
        kx = modes.kx
        ky = modes.ky

        n_ref = layers[0].n
        n_tra = layers[-1].n

        self.kz_ref = get_kz(n_ref, kx, ky)
        self.kz_tra = get_kz(n_tra, kx, ky)

        self.Wref = Wref
        self.Wtra = Wtra

    def run(self, Ex, Ey):
        modes = self.modes
        delta = (modes.mx == 0) * (modes.my == 0)
        self.C_inc = np.concatenate([Ex*delta, Ey*delta], axis=0)[:, None]
        self.C_ref = self.Sglobal[0] @ self.C_inc
        self.C_tra = self.Sglobal[2] @ self.C_inc
        return self
    
    def compute_mode_coefficients(self):
        # unwrap scatter matrix layer by layer to find the field inside
        c1p = self.C_inc # incidence mode coefficient
        c1m = self.C_ref # reflectance mode coefficient
        c3p = self.C_tra # transmitance mode coefficient
        c3m = np.zeros_like(c1p) # no backward wave from transmission side
        # c_modes = [(c3p, c3m)] # record all internal mode coefficient
        c_modes = [] # record all internal mode coefficient
        
        for A, B in self.scatter_mats[::-1]:
            c2p, c2m = get_field_incide(c1p, c1m, c3p, c3m, A, B)
            c_modes.append((c2p, c2m))
            c3p = c2p
            c3m = c2m
        # c_modes.append((c1p, c1m))
        # c_modes[0] = c_modes[1]
        c_modes = c_modes[::-1]
        return c_modes

    def get_internal_field(self, dz=0.05, mode_mask = None):
        n_modes = self.modes.n_modes # number of modes
        W0 = self.modes.W0 # free space mode matrix
        k0 = self.modes.k0 # free space wavenumber

        c_modes = self.compute_mode_coefficients()

        # Compute the field inside each layers
        fields = []
        zs = []
        z0 = 0

        c_modes = np.array(c_modes)

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.angle(c_modes[:, 0]), cmap='jet')
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.angle(c_modes[:, 1]), cmap='jet')
        # plt.show()

        for i in range(len(c_modes)-1):
            W, LAM, L = self.mode_matrices[i]
            c1p, c1m = c_modes[i]
            c2p, c2m = c_modes[i+1]
            nz = np.ceil(L/dz)

            WiW0 = np.linalg.solve(W, W0)

            C1 = WiW0 @ np.concatenate([c1p, c1m], axis=0)
            C2 = WiW0 @ np.concatenate([c2p, c2m], axis=0)

            # plt.subplot(2, 1, 1)
            # plt.plot(np.abs(C1))
            # plt.subplot(2, 1, 2)
            # plt.plot(np.abs(C2))

            if mode_mask is not None:
                order1 = np.argsort(np.argsort(-np.abs(C1[:, 0][:n_modes*2])))
                order2 = np.argsort(np.argsort(-np.abs(C1[:, 0][n_modes*2:])))
                order = np.concatenate([order1, order2+n_modes*2])
                C1 = C1 * mode_mask[order][:,None]
                C2 = C2 * mode_mask[order][:,None]
                # C1 /= np.mean(np.abs(C1))
                # C2 /= np.mean(np.abs(C2))


            for z in (np.arange(nz)+0.5)/nz*L:
                modep = W[:, :n_modes*2] @ (np.exp(LAM[:n_modes*2] * k0*z)[:, None] * C1[:n_modes*2])
                modem = W[:, n_modes*2:] @ (np.exp(LAM[n_modes*2:] * k0*(z-L))[:, None] * C2[n_modes*2:])
                mode = modep + modem
                # mode = W @ (np.exp(LAM * k0*z)[:, None] * C1)
                # mode = W @ (np.exp(LAM * k0*(z-L))[:, None] * C2)
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

        half_y = (n_modes_y+1)//2
        half_x = (n_modes_x+1)//2

        fields = block_matrix([
            [
                fields[:, :, :half_y, :half_x],
                np.zeros([len(fields), 4, half_y, pad_x]),
                fields[:, :, :half_y, half_x:],
            ],
            [np.zeros([len(fields), 4, pad_y, points_x])],
            [
                fields[:, :, half_y:, :half_x],
                np.zeros([len(fields), 4, n_modes_y-half_y, pad_x]),
                fields[:, :, half_y:, half_x:],
            ],
        ], axis1=3, axis2=2)

        px = (np.arange(points_x)/points_x)*self.modes.period_x
        py = (np.arange(points_y)/points_y)*self.modes.period_y
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

        return px, py, EX, EY
    
    def get_efficiency(self):
        n_modes = self.modes.n_modes
        kx = self.modes.kx
        ky = self.modes.ky
        kz_ref = self.kz_ref
        kz_tra = self.kz_tra

        # Compute diffraction efficiency
        E_inc = self.C_inc # self.Wref[:n_modes*2, :n_modes*2] @ self.C_inc
        E_ref = self.C_ref # self.Wref[:n_modes*2, n_modes*2:] @ self.C_ref
        E_tra = self.C_tra # self.Wtra[:n_modes*2, :n_modes*2] @ self.C_tra

        I_inc = get_intensity(E_inc[:n_modes, 0],
                              E_inc[n_modes:, 0], kx, ky, kz_ref)
        I_ref = get_intensity(E_ref[:n_modes, 0],
                              E_ref[n_modes:, 0], kx, ky, kz_ref)
        I_tra = get_intensity(E_tra[:n_modes, 0],
                              E_tra[n_modes:, 0], kx, ky, kz_tra)

        I = np.sum(I_inc)
        R = I_ref / I

        denominator = np.real(kz_ref)
        valid = np.abs(denominator) > 0
        denominator[~valid] = 1
        T = I_tra / I * np.real(kz_tra) / denominator
        T[~valid] = 0

        logger.info(f'R = {np.sum(R)}')
        logger.info(f'T = {np.sum(T)}')
        logger.info(f'A = {1-np.sum(R)-np.sum(T)}')
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
