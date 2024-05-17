# Formulation of RCWA
Rigorous coupled wave analysis formulation.

This is a documentation to provide reference to important derivation, and help people understand the implementation in the code. Some variable will have (`Variable Name`) to denote the short-form variable naming convention as it's sometimes hard to understand.

## Maxwell Equation
$$\nabla\cdot\vec D=\rho$$

$$\nabla\cdot\vec B=0$$

$$\nabla\times\vec E=-{\partial\vec B\over\partial t}$$

$$\nabla\times\vec H=\vec J+{\partial\vec D\over\partial t}$$

- $\vec D$ : Electric displacement
- $\vec B$ : Magnetic flux density
- $\vec H$ : Magnetic field vector (`H`)
- $\vec E$ : Electric field vector (`E`)
- $\rho$ : Charge density
- $J$ : Current density
- $t$ : time (`t`)

### Linear material assumption 
$$\vec D=\varepsilon\vec E$$

$$\vec B=\mu\vec H$$

- $\varepsilon$ : Permittivity
- $\mu$ : Permeability

###  Source-free assumption
$$\rho=0$$

$$J=0$$

### Monochromatic assumption
$$\vec E\propto e^{-i\omega t}$$

$$\vec H\propto e^{-i\omega t}$$
- $\omega$ : Angular frequency



### Normalization of magnetic field
$$\tilde{\vec H}=i\sqrt{\mu_0\over\varepsilon_0}\vec H$$

- $\varepsilon_0$ Free-space permittivity
- $\mu_0$ : Free-space permeability

$$\nabla\times\vec E=k_0\mu_r\tilde{\vec H}$$

$$\nabla\times\tilde{\vec H}=k_0\varepsilon_r\vec E$$

- $k_0$ : Free-space wave number $2\pi\over\lambda_0$
    - $\lambda_0$ : Free-space wavelength (`wl`)
- $\varepsilon_r$ : Relative permittivity $\varepsilon\over\varepsilon_0$ (`er`)
- $\mu_r$ : Relative permeability $\mu\over\mu_0$ (`ur`)

> ### How does this normalization term determined?
> we can observe if we apply linear, source-free, monochromatic assumption, we get the following equation:
> $$\nabla\times\vec E=i\omega\mu\vec H$$
> $$\nabla\times\vec H=-i\omega\varepsilon\vec E$$
> Where the value of $\mu$ and $\varepsilon$ in free-space is:
> - $\varepsilon_0 = 8.8541878128(13)×10^{−12}$ F⋅m−1
> - $\mu_0 = 1.25663706212(19)×10^{−6}$ N⋅A−2
> 
> And the angular frequency of the visible light is typically $2.7×10^{15}$ rad/s to $4.7×10^{15}rad/s$, which is hard to compute.
>
> If we introduce a normalization term as $\tilde{\vec H}=\alpha\vec H$, the equation becomes:
> $$\nabla\times\vec E=({i\omega\mu\over\alpha})\tilde{\vec H}$$
> $$\nabla\times\tilde{\vec H}=(-i\omega\varepsilon\alpha)\vec E$$
> We can balance the coefficient in free-space condition as:
> $${i\omega\mu_0\over\alpha}=-i\omega\varepsilon_0\alpha$$
> Result in:
> $$\alpha^2=-{\mu_0\over\varepsilon_0}$$
> $$\alpha=i\sqrt{\mu_0\over\varepsilon_0}$$


## Plane wave decomposition
$$\vec E(x,y;z)=\sum_{mn}\vec E(m,n;z)e^{i(k_x(m,n)x+k_y(m,n)y)}$$

$$\tilde{\vec H}(x,y;z)=\sum_{mn}\vec H(m,n;z)e^{i(k_x(m,n)x+k_y(m,n)y)}$$

- $\vec E(m,n;z)$ : Electric field coefficients
- $\vec H(m,n;z)$ : Magnetic field coefficients
- $m,n$ : Mode number, integer

$$\varepsilon_r(x, y)=\sum_{mn}\varepsilon_r(m,n)e^{i(mG_xx+nG_yy)}$$

$$\mu_r(x,y)=\sum_{mn}\mu_r(m,n)e^{i(mG_xx+nG_yy)}$$

- $\varepsilon_r(m,n)$ : Permittivity coefficients
- $\mu_r(m,n)$ : Permeability coefficients