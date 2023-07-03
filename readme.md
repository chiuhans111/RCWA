# RCWA (Rigorous Coupled Wave Analysis)

This repository contains the implementation of Rigorous Coupled Wave Analysis (RCWA), 
a numerical method used to analyze the scattering and diffraction of electromagnetic waves by periodic structures.
It includes a detailed formulation document, code, and examples.

The RCWA formulation is based on the following references:

- EMPossible: The website [EMPossible](https://empossible.net/) provides a formulation document titled "Formulation of Rigorous Coupled-Wave Analysis (RCWA)" that offers insights into the theoretical foundations and mathematical equations of the RCWA method. You can find the document [here](https://empossible.net/wp-content/uploads/2019/08/Lecture-7a-RCWA-Formulation.pdf).

- R. C. Rumpf, "IMPROVED FORMULATION OF SCATTERING MATRICES FOR SEMI-ANALYTICAL METHODS THAT IS CONSISTENT WITH CONVENTION," PIER B, vol. 35, pp. 241–261, 2011. This research paper by R. C. Rumpf presents an improved formulation of scattering matrices for semi-analytical methods, including RCWA. The paper offers valuable insights into the consistent formulation of scattering matrices and can be accessed [here](https://doi.org/10.2528/PIERB11083107).

- EMPossible: The YouTube channel [EMPossible](https://www.youtube.com/@empossible1577) provides educational videos on electromagnetic principles, including a lecture titled "Lecture 19 (CEM) -- Formulation of Rigorous Coupled-Wave Analysis." This lecture provides a visual explanation of the RCWA formulation and can be viewed [here](https://www.youtube.com/watch?v=LEWTvwrYxiI&t=1s&ab_channel=EMPossible).

## Directory Structure

- `formulation/`: Contains the formulation document in both docx and pdf formats, providing a detailed explanation of the RCWA method.
- `notebook/rcwa_v1.ipynb`: Includes the code implementation of the RCWA method, achieving energy conservation (T+R=1) and providing visualization of some modes.

## Requirements

- Python 3.10
- Dependencies:
  - NumPy (for numerical computations)
  - Matplotlib (for plotting)
  - SciPy (for solving eigen problems)

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

## Future Direction

The future direction of this project includes the integration of TensorFlow to improve performance 
and the development of a more user-friendly package with enhanced visualization capabilities. 

Collaboration and contributions from the community are highly encouraged. 
If you have ideas, suggestions, or would like to contribute to the project, please don't hesitate to reach out.

## About the Author

I am an electro-optical student who has a strong passion for optics. 
I developed this RCWA implementation as part of my research work.
If you have any questions, suggestions, or would like to collaborate on optical research projects, feel free to contact me.
