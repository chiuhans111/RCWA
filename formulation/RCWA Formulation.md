# Formulation of RCWA (V2)

> This document describes the mathematical formulation of the Rigorous Coupled Wave Analysis (RCWA) as implemented in the V2 engine.

Note: This document is updated by AI.

## 1. Maxwell's Equations and Normalization

We start with the monochromatic, source-free Maxwell equations in frequency space:
$$\nabla \times \vec{E} = i\omega\mu\vec{H}$$
$$\nabla \times \vec{H} = -i\omega\epsilon\vec{E}$$

To improve numerical stability, we normalize the magnetic field $\tilde{\vec{H}}$ such that:
$$\tilde{\vec{H}} = i\sqrt{\frac{\mu_0}{\epsilon_0}}\vec{H}$$

The normalized Maxwell equations become:
$$\nabla \times \vec{E} = k_0 \mu_r \tilde{\vec{H}}$$
$$\nabla \times \tilde{\vec{H}} = k_0 \epsilon_r \vec{E}$$
where $k_0 = 2\pi/\lambda_0$ is the free-space wavenumber.

## 2. Fourier Space Mapping (Li's Rules)

For a periodic structure, the permittivity $\epsilon_r(x,y)$ and the fields are expanded into Fourier series. When multiplying two Fourier series (e.g., $\vec{D} = \epsilon \vec{E}$), we must use **Li's Rules (1997)** to ensure convergence:

1. **Laurent's Rule**: If $\epsilon$ and $E$ have no concurrent discontinuities, the product is the convolution matrix $[[\epsilon]]$.
2. **Inverse Rule**: If $\epsilon$ and $E$ are concurrently discontinuous, the product is $[[1/\epsilon]]^{-1}$.

In our implementation (`modes.py`), we construct the convolution matrices for $\epsilon_x, \epsilon_y, \epsilon_z$ according to these rules to avoid the "Gibbs phenomenon" in grating simulations.

## 3. The State-Space Equation

Substituting the plane wave expansions into the normalized Maxwell equations, we eliminate the longitudinal components ($E_z, H_z$) to arrive at a set of coupled differential equations for the transverse components:

$$\frac{d}{d(k_0 z)} \begin{bmatrix} E_x \\ E_y \\ H_x \\ H_y \end{bmatrix} = \Omega \begin{bmatrix} E_x \\ E_y \\ H_x \\ H_y \end{bmatrix}$$

For an isotropic layer, the matrix $\Omega$ is:
$$\Omega = \begin{bmatrix} 0 & 0 & S_x \epsilon_z^{-1} S_y & \mu_r I - S_x \epsilon_z^{-1} S_x \\ 0 & 0 & S_y \epsilon_z^{-1} S_y - \mu_r I & -S_y \epsilon_z^{-1} S_x \\ S_x \mu_r^{-1} S_y & \epsilon_y - S_x \mu_r^{-1} S_x & 0 & 0 \\ S_y \mu_r^{-1} S_y - \epsilon_x & -S_y \mu_r^{-1} S_x & 0 & 0 \end{bmatrix}$$
where $S_x, S_y$ are diagonal matrices of normalized wave vectors.

## 4. Eigen-mode Solution

In each layer, we solve the eigenproblem of the $\Omega$ matrix:
$$\Omega = \begin{bmatrix} W & W \\ V & -V \end{bmatrix} \begin{bmatrix} \lambda & 0 \\ 0 & -\lambda \end{bmatrix} \begin{bmatrix} W & W \\ V & -V \end{bmatrix}^{-1}$$

- **$W$ Matrix**: Contains the electric field eigenvectors.
- **$V$ Matrix**: Contains the magnetic field eigenvectors.
- **$\lambda$ Vector**: Contains the propagation constants (eigenvalues).

The fields at any point $z$ within the layer are:
$$\psi(z) = \begin{bmatrix} W & W \\ V & -V \end{bmatrix} \begin{bmatrix} e^{\lambda k_0 z} & 0 \\ 0 & e^{-\lambda k_0 z} \end{bmatrix} \vec{c}$$

## 5. S-Matrix Cascading (Redheffer Star Product)

To connect multiple layers without numerical instability (the "$h$ problem"), we convert the boundary conditions into a **Scattering Matrix (S-Matrix)**. For two layers with matrices $S^{(A)}$ and $S^{(B)}$, the combined matrix $S^{(AB)} = S^{(A)} \otimes S^{(B)}$ is calculated via the Redheffer Star Product:

$$S_{11}^{(AB)} = S_{11}^{(A)} + S_{12}^{(A)} (I - S_{11}^{(B)} S_{22}^{(A)})^{-1} S_{11}^{(B)} S_{21}^{(A)}$$
$$S_{12}^{(AB)} = S_{12}^{(A)} (I - S_{11}^{(B)} S_{22}^{(A)})^{-1} S_{12}^{(B)}$$
$$S_{21}^{(AB)} = S_{21}^{(B)} (I - S_{22}^{(A)} S_{11}^{(B)})^{-1} S_{21}^{(A)}$$
$$S_{22}^{(AB)} = S_{22}^{(B)} + S_{21}^{(B)} (I - S_{22}^{(A)} S_{11}^{(B)})^{-1} S_{22}^{(A)} S_{12}^{(B)}$$

This approach is implemented in `scattermatrix.py` and ensures that decaying evanescent waves do not cause exponential growth in the calculation.

## 6. Efficiency Calculation

After calculating the global S-matrix for the entire stack, we apply the incident wave vector to find the reflected and transmitted coefficients. The diffraction efficiency for each mode is then:
$$DE_{refl, m} = \frac{\text{Re}(k_{z, m}^{inc})}{\text{Re}(k_{z, 0}^{inc})} |R_m|^2$$
$$DE_{tran, m} = \frac{\text{Re}(k_{z, m}^{sub})}{\text{Re}(k_{z, 0}^{inc})} |T_m|^2$$
