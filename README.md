# METALTIP

License: GPL-3.0-or-later

METALTIP is an open-source Python framework for electrostatic simulations of a metallic tip--sample system in generalized prolate spheroidal coordinates. It is intended for modeling electrostatic force, frequency shift, and contact-potential-difference (CPD) profiles in Kelvin probe force microscopy (KPFM) and related AFM force-spectroscopy measurements.

The code follows the variable-ξ finite-difference strategy used in Feenstra's SEMITIP framework, while being independently re-implemented for metallic tip--metallic sample configurations. The 3D version extends the calculation to laterally inhomogeneous surface-potential distributions.

## Repository contents

This repository contains two standalone Python scripts:

- `metaltip-2d-v1.1.0.py`  
  Axisymmetric 2D calculation for laterally homogeneous metallic samples.

- `metaltip-3d-v1.1.0.py`  
  Three-dimensional calculation for laterally inhomogeneous surface-potential distributions.

Recommended auxiliary files:

- `README.md`
- `requirements.txt`
- `LICENSE`
- `CITATION.cff`
- `.gitignore`

## Main features

### METALTIP 2D

- Solves Laplace's equation in vacuum.
- Uses a grounded sample boundary at `eta = 0`.
- Uses a Dirichlet boundary condition on the metallic tip.
- Uses Gauss--Seidel / SOR relaxation with staged grid refinement.
- Computes electrostatic force from the Maxwell stress.
- Computes frequency shift, Δf, from the electrostatic force.
- Supports bias-dependent F--V and Δf--V calculations.

### METALTIP 3D

- Solves Laplace's equation in vacuum for laterally inhomogeneous surface-potential distributions.
- Allows the sample-surface potential to be specified at `eta = 0`.
- Uses a Dirichlet boundary condition on the metallic tip.
- Uses a periodic boundary condition in the azimuthal direction.
- Uses Gauss--Seidel / SOR relaxation with staged grid refinement.
- Computes electrostatic force and frequency shift.
- Supports bias-dependent F--V and Δf--V calculations.
- Supports laterally resolved CPD-profile calculations.

## Requirements

METALTIP requires Python 3 and the following Python packages:

```text
numpy
scipy
numba
tqdm
matplotlib
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Quick start

Clone the repository and install the required packages:

```bash
git clone https://github.com/nobuyuki-ishida-nims/metaltip.git
cd METALTIP
pip install -r requirements.txt
```

Then edit the `Params` dataclass in the script you want to run.

### 2D calculation

Run:

```bash
python metaltip-2d-v1.1.0.py
```

The calculation mode is selected by `Params.mode`:

```python
mode = "df-V"
```

Available modes are:

- `"F"`: electrostatic force at a single bias
- `"F-V_approx"`: approximate force--bias curve
- `"F-V"`: full force--bias curve
- `"df-V_approx"`: approximate frequency-shift--bias curve
- `"df-V"`: full frequency-shift--bias curve

### 3D calculation

Run:

```bash
python metaltip-3d-v1.1.0.py
```

The top-level calculation mode is selected by `Params.mode`:

```python
mode = "CPD_prof"
```

Available top-level modes are:

- `"spec"`: spectrum calculation at a fixed lateral position
- `"CPD_prof"`: lateral CPD-profile calculation

For `mode = "spec"`, the spectrum type is selected by `Params.spec_mode`:

- `"F"`: electrostatic force at a single bias
- `"F-V_approx"`: force--bias curve from a five-point quadratic approximation
- `"F-V"`: full force--bias curve
- `"df-V_approx"`: frequency-shift--bias curve from a five-point quadratic approximation
- `"df-V"`: full frequency-shift--bias curve

For `mode = "CPD_prof"`, the CPD extraction method is selected by `Params.CPD_mode`:

- `"F"`: CPD from force--bias curves
- `"df"`: CPD from finite-amplitude Δf--bias curves
- `"df_approx"`: CPD from the small-amplitude approximation, Δf ≈ -(f0 / 2k) dF/dz

## Output files

When `save_data = True`, calculated data are saved in the `data/` directory.

Each output file contains:

1. a text header with the calculation mode and effective parameters,
2. a list of all fields defined in the `Params` dataclass,
3. numerical data below the `[DATA]` section.

This makes it possible to reconstruct the calculation settings from the output file.

## Important parameters

Most user-adjustable settings are defined in the `Params` dataclass.

Common parameters include:

- `force_mode`: `"sample"` or `"tip"`
- `open_angle`: full opening angle of the hyperbolic tip in degrees
- `sep_STM_nm`: nominal static STM tip--sample separation in nm
- `rad`: tip radius in nm
- `bias`, `bias_start`, `bias_end`: bias settings in V
- `Nrin`, `Nvin`: initial grid sizes
- `scale`: grid-scaling factor
- `nstep`: number of staged grid-refinement steps
- `ep0`, `ep_final`: convergence tolerances
- `itmax`: maximum number of iterations
- `kappa`: tunneling-current decay constant in nm^-1
- `ampl`: oscillation amplitude in nm
- `freq`: resonance frequency in Hz
- `k`: spring constant in N/m
- `z_offset`: additional tip displacement in nm
- `CPD`: contact potential difference in V
- `df_offset`: frequency-shift offset in Hz

Additional 3D parameters include:

- `surface_model`: surface-potential model, currently `"step"` or `"double_step"`
- `CPD_diff`: potential difference between surface-potential regions in V
- `w_interface`: smoothing width of the interface in nm
- `double_step_gap`: width of the strip-like region for the double-step model in nm
- `x_shift`: lateral shift of the prescribed surface-potential pattern in nm
- `x_start`, `x_end`, `dx`: lateral scan range and step size in nm
- `Npin`: number of azimuthal grid points

## Notes

- The code is computationally intensive, especially for full 3D Δf calculations.
- Approximate modes use a small number of bias points and a quadratic fit. It is recommended to validate the approximation by comparing with full bias sweeps for representative parameter settings.
- In dynamic STM conditions, the effective oscillation-center separation is calculated from `sep_STM_nm`, `kappa`, and `ampl`.

## Citation

If you use METALTIP in your work, please cite both the software and the associated paper.

### Associated paper

N. Ishida, "An Open-Source Framework for Quantitative Electrostatic Simulations in Kelvin Probe Force Microscopy", Journal of Applied Physics, in press.

The final bibliographic information, including DOI, volume, and page numbers, will be updated after publication.

### Software

N. Ishida, METALTIP, version 1.1.0, 2026.

## License

METALTIP is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See the `LICENSE` file for details.

## Author

Nobuyuki Ishida  
National Institute for Materials Science (NIMS)

## Acknowledgment

This code follows concepts from Feenstra's SEMITIP framework, especially the variable-ξ finite-difference strategy. METALTIP is an independent implementation and is not an official SEMITIP distribution.
