# RCWA (Rigorous Coupled Wave Analysis)

## Introduction
This repository contains the implementation of Rigorous Coupled Wave Analysis (RCWA), 
a numerical method used to analyze the scattering and diffraction of electromagnetic waves by periodic structures.
It includes a detailed formulation document, code, and examples.

### V2 Update Plan
- Provide efficient algorithm that is flexible for:
  - Simplify for homogeneous case
  - Stable isotropic case
  - Extended anisotropic case
- Provide advanced visualization capability
  - Calculate and display electric field inside structures

## Documentation (WIP)
- [RCWA Documentation](https://github.com/chiuhans111/RCWA/wiki)

## Directory Structure
- `formulation/`: 
  - Contains the formulation document in both docx and pdf formats, providing a detailed explanation of the RCWA method.
- `rcwa`: main package
- `test`: tests and examples
## Requirements

- Python 3.10
- Dependencies:
  - NumPy (for numerical computations)
  - Matplotlib (for plotting)
  - SciPy (for solving eigenproblems)

## Formulation
The document available in the `formulation/` directory provides derivations of the mathematical equations, and implementation details of the RCWA method.

## Code Implementation

The `notebook/rcwa_v1.ipynb` Jupyter Notebook file contains the code implementation of the RCWA method. 
The provided code achieves energy conservation (T+R=1). 
The notebook also includes visualization capabilities to explore and visualize some of the modes obtained through RCWA calculations.

Feel free to modify the code according to your specific requirements, 
such as changing the incident angle, wavelength, or the number of harmonics. 
Experiment with different parameters and explore the visualization of various modes.

## Known Issues
- Unstable, numerical error when using large harmonics, period=wavelength cases.

## Examples
*results obtained from RSoft DiffractMOD and our implementation of the RCWA method.*

### Traingular grating transmission diffraction efficiency


![image](https://github.com/chiuhans111/RCWA/assets/13620115/fbbcd056-4422-4563-a392-2c8f406a7507)

The simulations were conducted under the following conditions: 
a wavelength of 530 nm, incidence angle of 30 degrees,
grating period of 530 nm, groove depth ranges from 0 to 4 times the period (2120 nm), 
refractive index of 1.581 for both grating and substrate, symmetrical triangular groove shape, and TE polarization. 

Error comparison, RMSE: 0.276% 

![image](https://github.com/chiuhans111/RCWA/assets/13620115/07340c6b-d19c-4e62-9387-4a7a0160cb09)

### Slanted triangular grating transmission diffraction efficiency

![image](https://github.com/chiuhans111/RCWA/assets/13620115/2c12297b-7572-42fd-991f-648f9ec07520)

The simulations were conducted under the following conditions: 
a wavelength of 530 nm, incidence angle ranges from -20 to 50 degrees, grating period of 583 nm, 
refractive index of 1.46 for both grating and substrate, triangular groove shape, and TE polarization. 

Three grating shapes are simulated; 
in case a, the depth is 1.219 nm, with a slant angle of 23.289 degrees; 
in case b, the depth is 1.309 nm, with a slant angle of 12.55 degrees; 
in case c, the depth is 1.346 nm, with symmetric triangular groove.

RMSE: 0.262%

### TE/TM transmission diffraction efficiency

![image](https://github.com/chiuhans111/RCWA/assets/13620115/62487415-8b2e-4529-ab89-155ee33be86f)

The simulations were conducted under a wavelength of 530 nm, incidence angle ranging from -30 to 30 degrees, 
grating period of 357nm, groove depth of 800 nm, refractive index of 2.0 for both grating and substrate, triangular groove with 35 degrees of slant angle, and bot TM and TE polarization. 

RMSE: 1.279%


## Future Direction

The future direction of this project includes the integration of TensorFlow to improve performance 
and the development of a more user-friendly package with enhanced visualization capabilities. 

Collaboration and contributions from the community are highly encouraged. 
If you have ideas, suggestions, or would like to contribute to the project, please don't hesitate to reach out.

## About the Author

I am an electro-optical student who has a strong passion for optics. 
I developed this RCWA implementation as part of my research work.
If you have any questions, suggestions, or would like to collaborate on optical research projects, feel free to contact me.

## References
The RCWA formulation is based on the following references:

- EMPossible: The website [EMPossible](https://empossible.net/) provides a formulation document titled "Formulation of Rigorous Coupled-Wave Analysis (RCWA)" that offers insights into the theoretical foundations and mathematical equations of the RCWA method. You can find the document [here](https://empossible.net/wp-content/uploads/2019/08/Lecture-7a-RCWA-Formulation.pdf).

- R. C. Rumpf, "IMPROVED FORMULATION OF SCATTERING MATRICES FOR SEMI-ANALYTICAL METHODS THAT IS CONSISTENT WITH CONVENTION," PIER B, vol. 35, pp. 241–261, 2011. This research paper by R. C. Rumpf presents an improved formulation of scattering matrices for semi-analytical methods, including RCWA. The paper offers valuable insights into the consistent formulation of scattering matrices and can be accessed [here](https://doi.org/10.2528/PIERB11083107).

- EMPossible: The YouTube channel [EMPossible](https://www.youtube.com/@empossible1577) provides educational videos on electromagnetic principles, including a lecture titled "Lecture 19 (CEM) -- Formulation of Rigorous Coupled-Wave Analysis." This lecture provides a visual explanation of the RCWA formulation and can be viewed [here](https://www.youtube.com/watch?v=LEWTvwrYxiI&t=1s&ab_channel=EMPossible).


## License

The software is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.
