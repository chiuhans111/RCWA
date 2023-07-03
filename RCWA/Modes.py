import numpy as np
from RCWA.Domain import Domain


class Modes:
    def __init__(self, domain: Domain) -> None:
        self.domain = domain
        pass

    def ind2mode(self, id):
        # Ensure the mode ID is within the valid range
        id = id % self.num_modes

        # Calculate the mode indices in the x and y directions
        ix = id % self.num_modes_x
        iy = id // self.num_modes_x

        # Calculate the mode indices relative to the number of harmonics
        mx = ix - self.num_harmonics_x
        my = iy - self.num_harmonics_y

        # Return the mode indices as a tuple
        return mx, my

    def set_harmonics(self, num_harmonics_x, num_harmonics_y):
        # Harmonics and mode number
        self.num_harmonics_x = num_harmonics_x
        self.num_harmonics_y = num_harmonics_y

        self.num_modes_x = num_harmonics_x*2+1  # Number of x modes
        self.num_modes_y = num_harmonics_y*2+1  # Number of y modes

        self.num_modes = self.num_modes_x * self.num_modes_y  # Total number of modes

        # Ensure the mode ID is within the valid range
        self.mx, self.my = self.ind2mode(np.arange(self.num_modes))

    def set_incidence_AOI_POI(self, AOI, POI):
        self.set_incidence(
            k0x=np.sin(AOI) * np.cos(POI),  # x-component of the wave vector
            k0y=np.sin(AOI) * np.sin(POI),  # y-component of the wave vector
            k0z=np.cos(AOI)  # z-component of the wave vector
        )

    def set_incidence(self, k0x, k0y, k0z):
        self.k0x = k0x
        self.k0y = k0y
        self.k0z = k0z

        # Polarization related ------------------------------------------------
        k0v = np.array([k0x, k0y, k0z])  # Wave vector

        # Normal vector to the surface
        normal_vec = np.array([0, 0, 1], dtype='float')

        pol_vec_s = np.cross(normal_vec, k0v)  # Vector for s-polarization

        # If the incident wave is perpendicular to the surface
        if k0z == 1:
            pol_vec_s = np.array([0, 1, 0], dtype='float')

        pol_vec_p = np.cross(pol_vec_s, k0v)  # Vector for p-polarization

        # Normalize the s-polarization vector
        self.pol_vec_s = pol_vec_s/np.sqrt(np.sum(pol_vec_s**2))

        # Normalize the p-polarization vector
        self.pol_vec_p = pol_vec_p/np.sqrt(np.sum(pol_vec_p**2))

    def set_wavelength(self, wavelength):
        # Wavelength of the incident wave (in micrometers)
        self.wavelength = wavelength
        self.k0 = np.pi * 2 / wavelength  # Wave number

        # Spatial frequency in the x-direction
        self.gx = wavelength / self.domain.period_x
        # Spatial frequency in the y-direction
        self.gy = wavelength / self.domain.period_y

        # Wave vector in the x-direction
        self.kx = self.mx * self.gx + self.k0x
        # Wave vector in the y-direction
        self.ky = self.my * self.gy + self.k0y
