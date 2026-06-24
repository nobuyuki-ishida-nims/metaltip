"""
METALTIP (3D)

Electrostatic simulation code for a metallic tip–sample system
in generalized prolate spheroidal coordinates.

Version 1.1.0 - written by Nobuyuki Ishida, Mar 2026

This code is based on Feenstra's SEMITIP framework and adopts its
variable-ξ finite-difference strategy, while being independently
re-implemented and extended here for three-dimensional metallic
tip–metallic sample configurations.

In addition to solving the vacuum electrostatic potential, the code
computes electrostatic force and frequency shift under experimentally
relevant AFM/KPFM conditions, including bias-dependent F–V and Δf–V
characteristics and laterally resolved CPD profiles above spatially
inhomogeneous surface-potential distributions.

Features
--------
- Solves Laplace's equation in vacuum
- Sample boundary: surface-potential boundary specified at eta = 0
- Tip boundary: Dirichlet condition (phi = bias at eta = etat)
- Periodic boundary condition in theta
- Gauss–Seidel / SOR relaxation with staged grid refinement
- Computes electrostatic force from the Maxwell stress
- Computes frequency shift (Δf) from the electrostatic force
- Supports bias-dependent F–V and Δf–V calculations
- Supports laterally resolved CPD-profile calculations in 3D
"""

import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from scipy.special import i0  # modified Bessel I0
from datetime import datetime
try:
    from numba import njit
except ImportError as e:
    raise ImportError(
        "METALTIP requires 'numba'. Install it with: pip install numba"
    ) from e
    
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

pi = np.pi
EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]

@dataclass(frozen=True)
class Params:
    """
    Parameters controlling the METALTIP 3D calculation.

    mode
        "spec"
            Compute a spectrum at a fixed lateral position.
        "CPD_prof"
            Compute a lateral CPD profile by sweeping the lateral shift
            of the prescribed surface-potential pattern.

    spec_mode
        "F"
            Compute the electrostatic force at a single bias value.
        "F-V_approx"
            Compute an approximate force–bias curve over the specified bias range.
            The force is evaluated explicitly only at a small number of bias
            points, and the remaining points are obtained from a parabolic fit.
        "F-V"
            Compute the full force–bias curve over the specified bias range by
            evaluating the force at all bias points.
        "df-V_approx"
            Same as "F-V_approx", but for frequency shift (Δf) instead of force.
        "df-V"
            Same as "F-V", but for frequency shift (Δf) instead of force.

    CPD_mode
        "F"
            Compute a CPD profile by extracting the parabola minimum from
            force–bias curves at each lateral position.
        "df"
            Compute a CPD profile from finite-amplitude Δf–V curves.
        "df_approx"
            Compute a CPD profile from the small-amplitude approximation
            Δf ≈ -(f0 / 2k) dF/dz.

    force_mode
        Surface used for Maxwell-stress force integration:
        "sample" 
            integrate electrostatic pressure on the sample surface.
            This is the recommended/default mode for the present geometry.
        "tip":    
            integrate electrostatic pressure on the metallic tip surface.
            Useful as a consistency check or when the tip-side surface force is desired.

    sep_STM_nm
        Nominal input tip–sample separation defined by the static STM
        setpoint, i.e., with the tip oscillation switched off. For
        simulations under oscillating conditions, this separation is
        converted to the mean tip–sample distance used in the electrostatic
        calculation by taking into account the oscillation amplitude and the
        tunneling-current decay constant.

        For force and F–V calculations, the electrostatic interaction is
        evaluated at the oscillation-center separation derived from
        `sep_STM_nm` (and `z_offset`, if specified). For Δf calculations,
        `ampl` must be nonzero; use `sep_STM_nm` and/or `z_offset` to tune
        the oscillation-center distance.

    surface_model
        Built-in analytic model used to define the sample-surface potential
        at eta = 0. Currently supported models are "step" and "double_step".

    x_shift
        Lateral shift of the prescribed surface-potential pattern. In
        CPD-profile calculations, the lateral coordinate is scanned by
        sweeping this quantity.
    """
    
    # --- mode ---
    mode:         str = "CPD_prof"    # "CPD_prof" or "spec"
    spec_mode:    str = "df-V"        # "F","F-V_approx", "F-V", "df-V_approx", or "df-V"
    CPD_mode:     str = "df"          # "F", "df", or "df_approx"
    dz:         float = 0.03          # dz (nm) used in CPD_prof mode and df_approx, recommend df_approx: 0.03
    force_mode:   str = "sample"      # Maxwell-stress integration surface: "sample" or "tip"
    save_data:   bool = True          # save calculated data to file
    plot_data:   bool = True          # generate plots

    # --- SOR relaxation parameters ---
    omega:      float = 1.85       # SOR relaxation factor used in the Gauss–Seidel iteration
    
    # --- geometry / physics ---
    open_angle:    float = 20.0    # full opening angle of the hyperbolic tip (degrees)
    sep_STM_nm:    float = 1.0     # tip–sample separation corresponding in static STM mode (nm); see the docstring for details
    rad:           float = 10.0    # tip radius (nm)
    rad2:          float = 0.0     # reserved for future use; keep 0.0 in v1

    bias:          float = 1.0     # bias used for single-bias calculations (mode="spec",spec_mode="F")
    bias_start:    float = -1.0    # start of the bias sweep range
    bias_end:      float = 1.0     # end of the bias sweep range
    n_bias_points: int   = 21      # number of bias points in the sweep

    # --- surface potential (eta = 0) ---
    surface_model: str = "step"    # "step", "double_step", ...
    
    # model-specific parameters
    CPD_diff:        float = 0.50  # potential difference V2 - V1 across the interface (V)
    w_interface:     float = 0.5   # smoothing width of the interface (nm)
    double_step_gap: float = 5.0   # spacing between the two interfaces (nm)
    
    # sweep (tip scan) parameters
    x_shift:     float = 0.0       # lateral shift used for a single F-V (or Δf-V) curve calculation (nm)
    x_start:     float = -15.0     # start of lateral-shift sweep for CPD-profile calculations (nm)
    x_end:       float = 15.0      # end of lateral-shift sweep for CPD-profile calculations (nm)
    dx:          float = 0.5       # lateral-shift step size (nm)
    
    # --- grid / solver ---
    Nrin:         int = 16        # initial number of radial grid points
    Nvin:         int = 8         # initial number of grid points into the vacuum
    Npin:         int = 16        # number of theta points (periodic) fixed, not scaled in each step
    scale:      float = 0.5       # grid-scaling factor used at each refinement step
    nstep:        int = 5         # number of staged grid-refinement steps
    ep0:        float = 1.0e-4    # convergence parameter of the first step
    ep_final:   float = 1.0e-6    # convergence parameter of the final step
    itmax:        int = 100000    # maximum number of iterations
    
    # --- average tunneling current ---
    kappa:      float = 11.0      # tunneling-current decay constant (1/nm), typically 10.0-11.0
    ampl:       float = 0.50      # oscillation amplitude of AFM tip (nm)
    freq:       float = 30.0e3    # resonance frequency of the force sensor (Hz)
    k:          float = 1825      # spring constant of the force sensor (N/m)
    z_offset:   float = 0.0       # additional tip displacement along z applied during spectroscopy (nm)

    # --- experiment ----
    CPD:        float =  0.0      # contact potential difference between the tip and region 1 (V)
    df_offset:  float =  0.0      # frequency-shift offset at the parabola vertex (Hz)

@dataclass
class Workspace:
    r: np.ndarray      # radial grid points on the sample surface
    dr: np.ndarray     # radial grid spacing
    tip: np.ndarray    # boolean mask indicating the tip region on the grid
    phi: np.ndarray    # work array for the electrostatic potential
    Vs:   np.ndarray   # sample surface potential at eta=0

def alloc_workspace(params: Params) -> Workspace:
    """
    Allocate workspace arrays at the maximum grid size required over all
    refinement steps.

    The arrays are preallocated once and then reused during the calculation
    to avoid repeated memory allocation. In the current implementation,
    only Nr and Nv are refined; Np is kept fixed at params.Npin.
    """
    Nr_max = params.Nrin * (2 ** (params.nstep - 1))
    Nv_max = params.Nvin * (2 ** (params.nstep - 1))
    Np_max = params.Npin
    return Workspace(
        r=np.zeros(Nr_max, dtype=np.float64),
        dr=np.zeros(Nr_max, dtype=np.float64),
        tip=np.zeros((Nr_max, Nv_max), dtype=np.bool_),
        phi=np.zeros((2, Nr_max, Nv_max, Np_max), dtype=np.float64),
        Vs=np.zeros((Nr_max, Np_max), dtype=np.float64),
    )

@dataclass(frozen=True)
class Derived:
    """
    Quantities derived from the user-specified input parameters.

    These variables are computed internally from the geometry,
    solver settings, and oscillation conditions, and are reused
    during the electrostatic calculation.
    """

    last_step_index: int   # final step index (= nstep - 1), used to define the convergence-tolerance schedule
    ep_scale: float        # factor controlling the stepwise decrease of the convergence tolerance
    slope: float           # slope of the tip shank
    etat: float            # eta value of the tip surface
    a: float               # focal-length parameter of the generalized prolate spheroidal coordinates
    sprime: float          # effective mean tip–sample separation used in the electrostatic calculation
    z0: float              # z coordinate of the center of the eta = etat surface
    c: float               # axial shift parameter of the coordinate system

def make_derived(params: Params, sep_nm: float) -> Derived:
    # ep schedule
    last_step_index = params.nstep - 1
    ep_scale = (params.ep_final / params.ep0) ** (1.0 / last_step_index) if last_step_index > 0 else 1.0

    # geometry
    slope = np.tan(np.deg2rad(90.0 - (params.open_angle / 2.0)))
    etat = 1.0 / np.sqrt(1.0 + 1.0 / (slope * slope))
    a = params.rad * slope * slope / etat
    sprime = a * etat
   
    sep_nm = float(sep_nm)
    z0 = sep_nm - sprime
    c = z0 / sprime
    return Derived(last_step_index=last_step_index, ep_scale=ep_scale,
                   slope=slope, etat=etat, a=a,
                   sprime=sprime, z0=z0, c=c)

@njit(cache=True, nogil=True)
def tip_profile(r, rad2):
    if rad2 <= 0.0:
        return 0.0
    if r >= rad2:
        return 0.0
    return rad2 - np.sqrt(rad2*rad2 - r*r)

@njit(cache=True, nogil=True)
def fill_surface_potential_step(Vs, r,
                                Nr: int, Np: int,
                                dp: float,
                                CPD_diff: float,
                                x_shift: float,
                                w_interface: float):
    """
    Fill Vs for a smooth single-step surface potential:
        V = 0          (x << x_shift)
        V = CPD_diff   (x >> x_shift)
    """
    w = w_interface
    if w <= 0.0:
        w = 1.0e-12

    for i in range(Nr):
        ri = r[i]
        for k in range(Np):
            theta = float(k) * dp
            x = ri * np.cos(theta)
            s = 0.5 * (1.0 + np.tanh((x - x_shift) / w))
            Vs[i, k] = CPD_diff * s

@njit(cache=True, nogil=True)
def fill_surface_potential_double_step(Vs, r,
                                       Nr: int, Np: int,
                                       dp: float,
                                       CPD_diff: float,
                                       x_shift: float,
                                       w_interface: float,
                                       double_step_gap: float):
    """
    Fill Vs for a smooth double-step / strip model:
        V = 0          outside the strip
        V = CPD_diff   inside the strip

    The strip is centered at x = x_shift and bounded by
    x = x_shift - gap/2 and x = x_shift + gap/2.
    """
    w = w_interface
    if w <= 0.0:
        w = 1.0e-12

    half_gap = 0.5 * double_step_gap

    for i in range(Nr):
        ri = r[i]
        for k in range(Np):
            theta = float(k) * dp
            x = ri * np.cos(theta)

            sL = 0.5 * (1.0 + np.tanh((x - (x_shift - half_gap)) / w))
            sR = 0.5 * (1.0 + np.tanh((x - (x_shift + half_gap)) / w))

            Vs[i, k] = CPD_diff * (sL - sR)

@dataclass
class StepCache3D:
    """Precomputed geometry- and mesh-dependent objects for one refinement step.
    Notes
    -----
    - Independent of x_shift (only Vs depends on x_shift).
    - Independent of bias (the tip mask and stencil coefficients do not depend on bias).
    """
    Nr: int
    Nv: int
    Np: int
    dr0: float
    deta: float
    dp: float
    r: np.ndarray          # (Nr,)
    dr: np.ndarray         # (Nr,)
    tip: np.ndarray        # (Nr, Nv) bool (axisymmetric)
    Ap: np.ndarray         # (Nr, Nv-1)
    Am: np.ndarray         # (Nr, Nv-1)
    Bp: np.ndarray         # (Nr, Nv-1)
    Bm: np.ndarray         # (Nr, Nv-1)
    Cc: np.ndarray         # (Nr, Nv-1)
    Pc: np.ndarray         # (Nr, Nv-1)
    inv_den: np.ndarray    # (Nr, Nv-1)

@dataclass
class SolverCache3D:
    """Cached objects reused across a full 3D solve for a fixed separation."""
    d: Derived
    sep_nm: float
    steps: list[StepCache3D]

def prepare_solver_cache_3d(params: Params, sep_nm: float) -> SolverCache3D:
    """
    Prepare a reusable cache for repeated 3D solves at fixed separation.

    Intended use
    ------------
    - CPD-profile sweeps with many surface-potential configurations at fixed geometry
    - Bias sweeps at fixed geometry

    This cache stores, for each staged refinement step:
    - radial grid and radial spacing
    - axisymmetric tip mask
    - precomputed stencil coefficients

    It does NOT store `Vs`, because the sample-surface potential may vary
    between solves. In the current implementation, only Nr and Nv are
    refined; Np is kept fixed.
    """
    d = make_derived(params, sep_nm)

    Nr = int(params.Nrin)
    Nv = int(params.Nvin)
    Np = int(params.Npin)
    nstep = int(params.nstep)

    dr0 = float(params.rad * params.scale)
    dp = 2.0 * pi / float(Np)

    steps: list[StepCache3D] = []
    for i_step in range(nstep):
        deta = d.etat / float(Nv)

        r = np.zeros(Nr, dtype=np.float64)
        dr = np.zeros(Nr, dtype=np.float64)
        tip = np.zeros((Nr, Nv), dtype=np.bool_)

        _build_r_dr_tipmask_3d(tip, r, dr, Nr, Nv, dr0, d.a, d.c, d.etat, params.rad2)

        xi, dxi = build_xi_dxi(r, d.a)
        Ap, Am, Bp, Bm, Cc, Pc, inv_den = precompute_stencil_coeffs_feenstra_vargrid_3d(
            xi, dxi, Nr, Nv, d.a, d.c, deta, dp
        )

        steps.append(
            StepCache3D(
                Nr=Nr, Nv=Nv, Np=Np,
                dr0=dr0, deta=deta, dp=dp,
                r=r, dr=dr, tip=tip,
                Ap=Ap, Am=Am, Bp=Bp, Bm=Bm, Cc=Cc, Pc=Pc, inv_den=inv_den,
            )
        )

        if i_step != (nstep - 1):
            Nr *= 2
            Nv *= 2
            dr0 *= 0.5

    return SolverCache3D(d=d, steps=steps, sep_nm=float(sep_nm))

def refine_grid_3d_cached(phi, tip, Vs, r, dr,
                          step_coarse: StepCache3D,
                          step_fine: StepCache3D,
                          params: Params,
                          *,
                          x_shift: Optional[float] = None,
                          bias: float):
    """
    Refine the solution from a coarse to a fine grid using cached geometry data.

    This mirrors `refine_grid_3d()` but does not rebuild the grid, tip mask,
    or stencil coefficients. It only:
    - copies the coarse solution into `phi[1]`
    - loads the fine-grid `r`, `dr`, and `tip` mask
    - updates `Vs` for the fine grid
    - interpolates the interior vacuum potential into `phi[0]`
    - reapplies the tip Dirichlet boundary condition
    """
    Nr_coarse = step_coarse.Nr
    Nv_coarse = step_coarse.Nv
    Np = step_coarse.Np  # fixed across refinement steps

    Nr = step_fine.Nr
    Nv = step_fine.Nv

    # save coarse solution
    phi[1, :Nr_coarse, :Nv_coarse, :Np] = phi[0, :Nr_coarse, :Nv_coarse, :Np]

    # load fine grid/mask
    r[:Nr] = step_fine.r
    dr[:Nr] = step_fine.dr
    tip[:Nr, :Nv] = step_fine.tip

    # update Vs for the fine grid
    update_Vs_3d(
        Vs, r,
        Nr, Np, step_fine.dp,
        params,
        x_shift=x_shift,
    )

    # enforce tip Dirichlet before interpolation
    _apply_tip_dirichlet_3d(phi, tip, Nr, Nv, Np, bias)

    # interpolate interior vacuum from coarse phi[1]
    for i in range(Nr):
        ic = i // 2
        icp = ic + 1
        if icp >= Nr_coarse:
            icp = Nr_coarse - 1
        is_mid_r = (i % 2) == 1

        for j in range(Nv - 1):
            if tip[i, j]:
                continue

            J = j + 1  # 1-based eta index
            if J == 1:
                jc0 = 0
                for k in range(Np):
                    if not is_mid_r:
                        vcoarse = phi[1, ic, jc0, k]
                    else:
                        vcoarse = 0.5 * (phi[1, ic, jc0, k] + phi[1, icp, jc0, k])
                    phi[0, i, j, k] = 0.5 * (vcoarse + Vs[i, k])

            elif (J % 2) == 0:
                Jc = J // 2
                jc0 = Jc - 1
                for k in range(Np):
                    if not is_mid_r:
                        phi[0, i, j, k] = phi[1, ic, jc0, k]
                    else:
                        phi[0, i, j, k] = 0.5 * (phi[1, ic, jc0, k] + phi[1, icp, jc0, k])

            else:
                J1 = J // 2
                J2 = (J + 1) // 2
                jc10 = J1 - 1
                jc20 = J2 - 1
                for k in range(Np):
                    if not is_mid_r:
                        phi[0, i, j, k] = 0.5 * (phi[1, ic, jc10, k] + phi[1, ic, jc20, k])
                    else:
                        v1 = 0.5 * (phi[1, ic, jc10, k] + phi[1, icp, jc10, k])
                        v2 = 0.5 * (phi[1, ic, jc20, k] + phi[1, icp, jc20, k])
                        phi[0, i, j, k] = 0.5 * (v1 + v2)

def solve_potential_3d_cached(
    params: Params,
    ws: Workspace,
    cache: SolverCache3D,
    *,
    bias: Optional[float] = None,
    x_shift: Optional[float] = None,
):
    """
    Solve the potential using a reusable cache of geometry and stencil data.

    Notes
    -----
    - The cache is valid only for a fixed separation (`cache.sep_nm`).
    - The sample-surface potential and the metal boundary value may vary
      between solves, so they are updated each time.
    """
    d = cache.d
    ep = params.ep0

    bias_use = float(params.bias) if bias is None else float(bias)
    x_use = float(params.x_shift) if x_shift is None else float(x_shift)

    # Step 0: load grid/mask
    step0 = cache.steps[0]
    Nr = step0.Nr
    Nv = step0.Nv
    Np = step0.Np

    ws.r[:Nr] = step0.r
    ws.dr[:Nr] = step0.dr
    ws.tip[:Nr, :Nv] = step0.tip

    # Update Vs for the current solve
    update_Vs_3d(
        ws.Vs, ws.r,
        Nr, Np, step0.dp,
        params,
        x_shift=x_use,
    )

    # Initial guess (also enforces tip boundary at this step)
    set_initial_guess_3d(ws.phi, ws.tip, ws.Vs, Nr, Nv, Np, step0.deta, d.etat, bias_use)

    for i_step, step in enumerate(cache.steps):
        Nr = step.Nr
        Nv = step.Nv
        Np = step.Np

        print("Iter step No.", i_step + 1)
        print(f"sep (nm), bias, rad (nm) = {cache.sep_nm:.3f}, {bias_use}, {params.rad}")
        print("Nr, Nv, Np =", Nr, Nv, Np)

        # Safety: enforce tip Dirichlet
        _apply_tip_dirichlet_3d(ws.phi, ws.tip, Nr, Nv, Np, bias_use)

        iterate_potential_feenstra_vargrid_precomp_3d(
            ws.phi, ws.tip, ws.Vs,
            step.Ap, step.Am, step.Bp, step.Bm, step.Cc, step.Pc, step.inv_den,
            Nr, Nv, Np,
            ep=ep, itmax=params.itmax,
            omega=float(params.omega),
        )

        if i_step == (len(cache.steps) - 1):
            break

        step_next = cache.steps[i_step + 1]
        refine_grid_3d_cached(
            ws.phi, ws.tip, ws.Vs, ws.r, ws.dr,
            step, step_next,
            params,
            x_shift=x_use,
            bias=bias_use,
        )
        ep *= d.ep_scale

    final = cache.steps[-1]
    return final.Nr, final.Nv, final.Np, final.deta, final.dp, d

@njit(cache=True, nogil=True)
def _build_r_dr_tipmask_3d(tip, r, dr,
                           Nr: int, Nv: int,
                           dr0: float,
                           a: float, c: float, etat: float,
                           rad2: float):
    """
    Build the radial grid (`r`, `dr`) and the axisymmetric tip mask.

    This function prepares only geometry-dependent arrays. It does not
    initialize `Vs` or `phi`.
    """
    deta = etat / float(Nv)
    for i in range(Nr):
        r_i = (2.0*Nr*dr0/pi) * np.tan(pi*(i+0.5)/(2.0*Nr))
        r[i] = r_i
        dr[i] = dr0 / (np.cos(pi*(i+0.5)/(2.0*Nr))**2)

        xi2m1 = (r_i / a) ** 2
        xi = np.sqrt(1.0 + xi2m1)

        # interior eta layers: j = 0..Nv-2 correspond to J = 1..Nv-1
        # j = 0 (J = 1) is the first vacuum grid point, not the sample surface.
        for j in range(Nv-1):
            J = j + 1
            eta = J * deta

            z  = a * eta * (xi + c)
            zp = z * (J + 0.5) / float(J)
            rp = a * np.sqrt(xi2m1 * (1.0 - eta*eta))

            # NOTE (v1.0):
            # For the current fixed geometry, the tip surface coincides with the
            # outermost eta-layer, so this inside-tip test mainly serves future
            # extensions in which the tip boundary may intersect interior layers.
            ztip = (a * etat *
                    (np.sqrt(1.0 + rp*rp/((1.0 - etat*etat)*a*a)) + c)
                    - tip_profile(rp, rad2))

            tip[i, j] = (zp > ztip)

        tip[i, Nv-1] = True  # outer boundary / tip Dirichlet layer

@njit(cache=True, nogil=True)
def _apply_tip_dirichlet_3d(phi, tip,
                            Nr: int, Nv: int, Np: int,
                            bias: float):
    """
    Enforce the tip Dirichlet boundary condition (phi = bias) in both
    buffers for all theta points.

    The tip mask is axisymmetric in (r, eta), so the same boundary value
    is applied to every azimuthal index k.
    """
    for i in range(Nr):
        for j in range(Nv):
            if tip[i, j]:
                for k in range(Np):
                    phi[0, i, j, k] = bias
                    phi[1, i, j, k] = bias

def update_Vs_3d(Vs, r,
                 Nr: int, Np: int,
                 dp: float,
                 params: Params,
                 *,
                 x_shift: Optional[float] = None):
    """
    Update the sample-surface potential array `Vs` on the current 3D grid.

    Parameters
    ----------
    Vs : ndarray
        Output array updated in place. The active region is `Vs[:Nr, :Np]`.
    r : ndarray
        Radial grid array.
    Nr, Np : int
        Active grid sizes in the radial and azimuthal directions.
    dp : float
        Angular grid spacing in radians.
    params : Params
        Calculation parameters including the selected built-in
        surface-potential model.
    x_shift : float, optional
        Lateral shift of the surface-potential pattern. If not given,
        `params.x_shift` is used.

    Notes
    -----
    - This function does not return a new array; it fills `Vs` in place.
    - The built-in analytic model is selected by `params.surface_model`.
    - Currently supported models are `"step"` and `"double_step"`.
    """
    x_use = float(params.x_shift) if x_shift is None else float(x_shift)

    if params.surface_model == "step":
        fill_surface_potential_step(
            Vs, r, Nr, Np, dp,
            params.CPD_diff, x_use, params.w_interface
        )

    elif params.surface_model == "double_step":
        fill_surface_potential_double_step(
            Vs, r, Nr, Np, dp,
            params.CPD_diff, x_use, params.w_interface, params.double_step_gap
        )

    else:
        raise ValueError(f"Unknown surface_model: {params.surface_model}")

@njit(cache=True, nogil=True)
def build_xi_dxi(r_nm: np.ndarray, a_nm: float):
    """
    Construct the xi grid and its spacing from the radial grid.

    Parameters
    ----------
    r_nm : ndarray
        Radial grid in nm.
    a_nm : float
        Prolate spheroidal parameter a in nm.

    Returns
    -------
    xi : ndarray
        Xi grid defined by sqrt(1 + (r / a)^2).
    dxi : ndarray
        Grid spacing in xi. For i = 0, dxi[0] is measured from xi = 1.
    """
    r_nm = np.asarray(r_nm, dtype=np.float64)
    a_nm = float(a_nm)
    xi = np.sqrt(1.0 + (r_nm / a_nm) ** 2)
    dxi = np.empty_like(xi)
    dxi[0] = xi[0] - 1.0
    dxi[1:] = xi[1:] - xi[:-1]
    return xi, dxi

@njit(cache=True, nogil=True)
def set_initial_guess_3d(phi, tip, Vs, Nr, Nv, Np, deta, etat, bias):
    """
    Initialize the vacuum potential using the analytical 1D solution
    along the eta direction, broadcast independently over theta.
    """
    cetat = np.log((1.0 + etat) / (1.0 - etat))

    for i in range(Nr):
        jsave = 0
        for j in range(Nv - 1, -1, -1):
            if not tip[i, j]:
                jsave = j
                break

        for j in range(jsave, -1, -1):
            eta = (j + 1) * deta * float(Nv) / float(jsave + 2)
            g = np.log((1.0 + eta) / (1.0 - eta)) / cetat
            for k in range(Np):
                v0 = Vs[i, k]
                val = v0 + (bias - v0) * g
                phi[0, i, j, k] = val
                phi[1, i, j, k] = val

@njit(cache=True, nogil=True)
def precompute_stencil_coeffs_feenstra_vargrid_3d(
    xi: np.ndarray,
    dxi: np.ndarray,
    Nr: int, Nv: int,
    a: float, c: float,
    deta: float,
    dp: float,
):
    """
    3D extension of `precompute_stencil_coeffs_feenstra_vargrid()`.

    Compared with the 2D version, this function adds the theta-term
    coefficient
        P = t3 / dp^2,
    where
        t3 = xi2me2c / (xi2m1 * (1 - eta^2)).
    The diagonal denominator is correspondingly updated as
        den3 = den2 + 2 * P.
    """
    xi = np.asarray(xi, dtype=np.float64)
    dxi = np.asarray(dxi, dtype=np.float64)

    Ap = np.zeros((Nr, Nv-1), dtype=np.float64)
    Am = np.zeros((Nr, Nv-1), dtype=np.float64)
    Bp = np.zeros((Nr, Nv-1), dtype=np.float64)
    Bm = np.zeros((Nr, Nv-1), dtype=np.float64)
    C  = np.zeros((Nr, Nv-1), dtype=np.float64)
    P  = np.zeros((Nr, Nv-1), dtype=np.float64)
    inv_den = np.zeros((Nr, Nv-1), dtype=np.float64)

    inv_deta2 = 1.0/(deta*deta)
    inv_2deta = 0.5/deta
    inv_dp2   = 1.0/(dp*dp)

    j_arr = np.arange(Nv-1, dtype=np.float64)
    eta = (j_arr + 1.0)*deta
    eta2 = eta*eta
    eta4 = eta2*eta2
    ome2 = 1.0 - eta2

    c2 = c*c
    c3 = c2*c
    c2p2 = c2 + 2.0
    c2p6 = c2 + 6.0
    tc2p2 = 3.0*c2 + 2.0
    c2p1 = c2 + 1.0
    c2m1 = c2 - 1.0

    for i in range(0, Nr):
        xi_i = xi[i]
        xi2 = xi_i*xi_i
        xi3 = xi2*xi_i
        xi4 = xi3*xi_i
        xi5 = xi4*xi_i
        xi2m1 = xi2 - 1.0

        dxm = dxi[i]
        dxp = dxi[i+1] if (i+1) < Nr else dxi[i]
        if dxm <= 0.0:
            dxm = 1.0e-12
        if dxp <= 0.0:
            dxp = 1.0e-12
        dxpm = dxm + dxp
        if dxpm <= 0.0:
            dxpm = 1.0e-12

        inv_dxpm = 1.0/dxpm
        inv_dxp = 1.0/dxp
        inv_dxm = 1.0/dxm

        for j in range(Nv-1):
            e2 = eta2[j]
            e4 = eta4[j]
            om = ome2[j]
            e  = eta[j]

            xi2me2   = xi2 - e2
            xi2me2c  = xi_i*(xi_i + c) - e2*(c*xi_i + 1.0)
            xi2me2c2 = xi2me2c*xi2me2c

            t1 = xi2m1 * ((xi_i + c) * (xi_i + c) - e2 * (2.0 * c * xi_i + c2p1)) / xi2me2c
            t2 = om * xi2me2 / xi2me2c
            # theta coefficient (SEMITIP3 ITER3)
            t3 = xi2me2c / (xi2m1 * om)
            t4 = -c * e * xi2m1 * om / xi2me2c

            t5 = (
                (c3 + 3.0*c2*xi_i + c*c2p2*xi2 + 3.0*c2*xi3 + 4.0*c*xi4 + 2.0*xi5 +
                 e4*(c3 + tc2p2*xi_i + c*c2p6*xi2 + 3.0*c2*xi3) -
                 2.0*e2*(c*c2m1 + 3.0*c2*xi_i + c*c2p6*xi2 + tc2p2*xi3 + c*xi4)
                ) / xi2me2c2
            )

            t6 = (
                -e*(c2 + 4.0*c*xi_i + c2*xi2 + 2.0*xi4 +
                    e4*(2.0 + c2 + 4.0*c*xi_i + c2*xi2) -
                    2.0*e2*(c2 + 4.0*c*xi_i + c2p2*xi2)
                   ) / xi2me2c2
            )

            ap = t1 * 2.0 * (inv_dxp * inv_dxpm) + t5 * (inv_dxpm)
            am = t1 * 2.0 * (inv_dxm * inv_dxpm) - t5 * (inv_dxpm)
            bp = t2 * inv_deta2 + t6 * inv_2deta
            bm = t2 * inv_deta2 - t6 * inv_2deta
            ccoef = t4 * (inv_dxpm / deta)
            pcoef = t3 * inv_dp2

            den2 = t1 * 2.0 * ((inv_dxp + inv_dxm) * inv_dxpm) + 2.0 * t2 * inv_deta2
            den3 = den2 + 2.0 * pcoef

            if den3 <= 0.0 or (not np.isfinite(den3)):
                den3 = 1.0e-30

            Ap[i, j] = ap
            Am[i, j] = am
            Bp[i, j] = bp
            Bm[i, j] = bm
            C[i, j]  = ccoef
            P[i, j]  = pcoef
            inv_den[i, j] = 1.0/den3

    return Ap, Am, Bp, Bm, C, P, inv_den

@njit(cache=True, nogil=True)
def iterate_potential_feenstra_vargrid_precomp_3d(
    phi, tip, Vs,
    Ap, Am, Bp, Bm, C, P, inv_den,
    Nr, Nv, Np,
    ep, itmax,
    omega=1.0,
    #enforce_rmax=False
):
    """
    3D extension of iterate_potential_feenstra_vargrid_precomp()
    - periodic theta: k±1 wraps around

    Boundary conditions
    -------------------
    Tip metal (Dirichlet):
        V = bias on the tip (tip_mask == True).
        (In this implementation, j = Nv-1 is also set to tip/bias.)

    Sample surface (prescribed potential, 2-region CPD):
        The sample surface is at eta = 0 and its potential is stored in Vs[i,k].
        IMPORTANT: eta = 0 is NOT stored in the phi array.
        The first vacuum layer is j = 0 (eta = deta).
        When updating j = 0, the solver uses Vs as a ghost value for the j = -1
        neighbor (eta = 0), e.g. v(j-1) = Vs[i,k] (and corresponding cross terms).

    Axis (r = 0):
        Symmetry boundary (∂V/∂r = 0) is handled by a one-sided/ghost relation.

    Outer boundary (r = rmax):
        If enforce_rmax=True, phi(Nr-1, j, k) is copied from phi(Nr-2, j, k)
        for non-tip points (Neumann-type condition).
        Otherwise, a one-sided update consistent with the rmax boundary treatment
        is applied.
    """
    for it in range(itmax):
        err_max = 0.0

        # --- rmax boundary (same as 2D, applied per k) ---
        i = Nr - 1
        imi = Nr - 2
        for k in range(Np):
            kp = k + 1 if (k + 1) < Np else 0
            km = k - 1 if (k - 1) >= 0 else (Np - 1)

            for j in range(Nv-1):
                if tip[i, j]:
                    continue

                old = phi[0, i, j, k]
                ap = Ap[i, j]
                am = Am[i, j]
                bp = Bp[i, j]
                bm = Bm[i, j]
                cc = C[i, j]
                pp = P[i, j]

                vim = phi[0, imi, j, k]
                vjp1 = phi[0, i, j+1, k]
                vip_jp1 = vjp1
                vim_jp1 = phi[0, imi, j+1, k]

                if j == 0:
                    vjm1 = Vs[i, k]
                    vip_jm1 = Vs[i, k]
                    vim_jm1 = Vs[imi, k]
                else:
                    vjm1 = phi[0, i, j-1, k]
                    vip_jm1 = vjm1
                    vim_jm1 = phi[0, imi, j-1, k]

                vkp = phi[0, i, j, kp]
                vkm = phi[0, i, j, km]

                rhs = (
                    am*vim
                    + bp*vjp1 + bm*vjm1
                    + cc*(vip_jp1 - vip_jm1 - vim_jp1 + vim_jm1)
                    + pp*(vkp + vkm)
                )

                den = 1.0 / inv_den[i, j]
                den_eff = den - ap
                if den_eff <= 1.0e-30 or (not np.isfinite(den_eff)):
                    den_eff = 1.0e-30

                new_gs = rhs / den_eff
                new = old + omega*(new_gs - old)

                phi[0, i, j, k] = new
                diff = abs(new - old)
                if diff > err_max:
                    err_max = diff

        # --- i=0 axis (ghost) ---
        i = 0
        ipi = 1
        for k in range(Np):
            kp = k + 1 if (k + 1) < Np else 0
            km = k - 1 if (k - 1) >= 0 else (Np - 1)

            for j in range(Nv-1):
                if tip[i, j]:
                    continue

                old = phi[0, i, j, k]

                vim = (9.0*phi[0, i, j, k] - phi[0, ipi, j, k]) * 0.125
                vip = phi[0, ipi, j, k]

                if j == 0:
                    vjm1 = Vs[i, k]
                    vip_jm1 = Vs[ipi, k]
                    vim_jm1 = Vs[i, k]
                else:
                    vjm1 = phi[0, i, j-1, k]
                    vip_jm1 = phi[0, ipi, j-1, k]
                    vim_jm1 = (9.0*phi[0, i, j-1, k] - phi[0, ipi, j-1, k]) * 0.125

                vjp1 = phi[0, i, j+1, k]
                vip_jp1 = phi[0, ipi, j+1, k]
                vim_jp1 = (9.0*phi[0, i, j+1, k] - phi[0, ipi, j+1, k]) * 0.125

                vkp = phi[0, i, j, kp]
                vkm = phi[0, i, j, km]

                ap = Ap[i, j]
                am = Am[i, j]
                bp = Bp[i, j]
                bm = Bm[i, j]
                cc = C[i, j]
                pp = P[i, j]

                temp = (
                    ap*vip + am*vim
                    + bp*vjp1 + bm*vjm1
                    + cc*(vip_jp1 - vip_jm1 - vim_jp1 + vim_jm1)
                    + pp*(vkp + vkm)
                )
                new_gs = temp * inv_den[i, j]
                new = old + omega*(new_gs - old)

                phi[0, i, j, k] = new
                diff = abs(new - old)
                if diff > err_max:
                    err_max = diff
                    
        # --- interior i=1..Nr-2 ---
        for i in range(1, Nr-1):
            ipi = i + 1
            imi = i - 1
            for k in range(Np):
                kp = k + 1 if (k + 1) < Np else 0
                km = k - 1 if (k - 1) >= 0 else (Np - 1)

                for j in range(Nv-1):
                    if tip[i, j]:
                        continue

                    old = phi[0, i, j, k]
                    ap = Ap[i, j]
                    am = Am[i, j]
                    bp = Bp[i, j]
                    bm = Bm[i, j]
                    cc = C[i, j]
                    pp = P[i, j]

                    vip = phi[0, ipi, j, k]
                    vim = phi[0, imi, j, k]

                    vjp1 = phi[0, i, j+1, k]
                    vip_jp1 = phi[0, ipi, j+1, k]
                    vim_jp1 = phi[0, imi, j+1, k]

                    if j == 0:
                        vjm1 = Vs[i, k]
                        vip_jm1 = Vs[ipi, k]
                        vim_jm1 = Vs[imi, k]
                    else:
                        vjm1 = phi[0, i, j-1, k]
                        vip_jm1 = phi[0, ipi, j-1, k]
                        vim_jm1 = phi[0, imi, j-1, k]

                    vkp = phi[0, i, j, kp]
                    vkm = phi[0, i, j, km]

                    temp = (
                        ap*vip + am*vim
                        + bp*vjp1 + bm*vjm1
                        + cc*(vip_jp1 - vip_jm1 - vim_jp1 + vim_jm1)
                        + pp*(vkp + vkm)
                    )
                    new_gs = temp * inv_den[i, j]
                    corr = new_gs - old
                    new = old + omega * corr
                    
                    phi[0, i, j, k] = new
                    diff = abs(new - old)

                    if diff > err_max:
                        err_max = diff

        if (it % 100) == 0 and it != 0:
            print("Iter, Res =", "\t", it, "\t", err_max)

        if err_max < ep:
            return err_max

    return err_max

#@njit(cache=True, nogil=True)
def refine_grid_3d(phi, tip, Vs, r, dr,
                   Nr, Nv, Np, dr0, dp,
                   a, c, etat, rad2, bias,
                   params: Params,
                   *,
                   x_shift: Optional[float] = None,
                   ):
    """
    Refine the 3D grid by doubling the resolution in r and eta.

    The coarse solution is first copied to `phi[1]`. The fine-grid geometry
    is then rebuilt, the sample-surface potential is updated, the tip
    Dirichlet boundary is reapplied, and the interior vacuum potential is
    interpolated into `phi[0]`.
    """
    Nr_coarse = Nr
    Nv_coarse = Nv

    # save coarse solution
    phi[1, :Nr_coarse, :Nv_coarse, :Np] = phi[0, :Nr_coarse, :Nv_coarse, :Np]

    # refine sizes
    Nr = Nr * 2
    Nv = Nv * 2
    dr0_fine = dr0 / 2.0
    deta = etat / float(Nv)

    # rebuild fine grid / tip mask
    _build_r_dr_tipmask_3d(
        tip, r, dr,
        Nr, Nv,
        dr0_fine,
        a, c, etat, rad2,
    )

    # update Vs for the fine grid
    update_Vs_3d(
        Vs, r,
        Nr, Np, dp,
        params,
        x_shift=x_shift,
    )

    # enforce tip Dirichlet before interpolation
    _apply_tip_dirichlet_3d(phi, tip, Nr, Nv, Np, bias)

    # interpolate interior vacuum from coarse phi[1]
    for i in range(Nr):
        ic = i // 2
        icp = ic + 1
        if icp >= Nr_coarse:
            icp = Nr_coarse - 1
        is_mid_r = (i % 2) == 1

        for j in range(Nv - 1):
            if tip[i, j]:
                continue

            J = j + 1
            if J == 1:
                jc0 = 0
                for k in range(Np):
                    if not is_mid_r:
                        vcoarse = phi[1, ic, jc0, k]
                    else:
                        vcoarse = 0.5 * (phi[1, ic, jc0, k] + phi[1, icp, jc0, k])
                    phi[0, i, j, k] = 0.5 * (vcoarse + Vs[i, k])

            elif (J % 2) == 0:
                Jc = J // 2
                jc0 = Jc - 1
                for k in range(Np):
                    if not is_mid_r:
                        phi[0, i, j, k] = phi[1, ic, jc0, k]
                    else:
                        phi[0, i, j, k] = 0.5 * (phi[1, ic, jc0, k] + phi[1, icp, jc0, k])

            else:
                J1 = J // 2
                J2 = (J + 1) // 2
                jc10 = J1 - 1
                jc20 = J2 - 1
                for k in range(Np):
                    if not is_mid_r:
                        phi[0, i, j, k] = 0.5 * (phi[1, ic, jc10, k] + phi[1, ic, jc20, k])
                    else:
                        v1 = 0.5 * (phi[1, ic, jc10, k] + phi[1, icp, jc10, k])
                        v2 = 0.5 * (phi[1, ic, jc20, k] + phi[1, icp, jc20, k])
                        phi[0, i, j, k] = 0.5 * (v1 + v2)

    return Nr, Nv, deta

def solve_potential_3d_base(
    params: Params,
    ws: Workspace,
    sep_nm: float,
    *,
    bias: Optional[float] = None,
    x_shift: Optional[float] = None,
):
    """
    Solve the 3D vacuum electrostatic potential without using a prebuilt cache.

    The grid is initialized from scratch at the coarsest step, then refined
    step by step while rebuilding the geometry and updating the sample-surface
    potential as needed.
    """
    nstep = int(params.nstep)
    sep_nm = float(sep_nm)
    d = make_derived(params, sep_nm)

    Nr = int(params.Nrin)
    Nv = int(params.Nvin)
    Np = int(params.Npin)

    dr0 = float(params.rad * params.scale)
    deta = d.etat / float(Nv)
    dp = 2.0 * pi / float(Np)

    ep = params.ep0
    bias_use = float(params.bias) if bias is None else float(bias)
    x_use = float(params.x_shift) if x_shift is None else float(x_shift)

    # --- step 0: build grid / tip mask / Vs / initial guess ---
    _build_r_dr_tipmask_3d(
        ws.tip, ws.r, ws.dr,
        Nr, Nv,
        dr0,
        d.a, d.c, d.etat,
        params.rad2,
    )

    update_Vs_3d(
        ws.Vs, ws.r,
        Nr, Np, dp,
        params,
        x_shift=x_use,
    )

    _apply_tip_dirichlet_3d(
        ws.phi, ws.tip,
        Nr, Nv, Np,
        bias_use,
    )

    set_initial_guess_3d(
        ws.phi, ws.tip, ws.Vs,
        Nr, Nv, Np,
        deta, d.etat,
        bias_use,
    )

    for i_step in range(nstep):
        print("Iter step No.", i_step + 1)
        print(f"sep (nm), bias, rad (nm) = {sep_nm:.3f}, {bias_use}, {params.rad}")
        print("Nr, Nv, Np =", Nr, Nv, Np)

        xi, dxi = build_xi_dxi(ws.r[:Nr], d.a)
        Ap, Am, Bp, Bm, Cc, Pc, inv_den = precompute_stencil_coeffs_feenstra_vargrid_3d(
            xi, dxi, Nr, Nv, d.a, d.c, deta, dp
        )

        _apply_tip_dirichlet_3d(ws.phi, ws.tip, Nr, Nv, Np, bias_use)

        iterate_potential_feenstra_vargrid_precomp_3d(
            ws.phi, ws.tip, ws.Vs,
            Ap, Am, Bp, Bm, Cc, Pc, inv_den,
            Nr, Nv, Np,
            ep=ep, itmax=params.itmax,
            omega=float(params.omega),
        )

        if i_step == (nstep - 1):
            break

        Nr, Nv, deta = refine_grid_3d(
            ws.phi, ws.tip, ws.Vs, ws.r, ws.dr,
            Nr, Nv, Np, dr0, dp,
            d.a, d.c, d.etat, params.rad2, bias_use,
            params,
            x_shift=x_use,
        )
        dr0 *= 0.5
        ep *= d.ep_scale

    return Nr, Nv, Np, deta, dp, d

def _fd_weights_fornberg(x0: float, x: np.ndarray, m: int = 1) -> np.ndarray:
    """
    Compute finite-difference weights using the Fornberg algorithm.

    The returned weights approximate the m-th derivative at x0 from
    function values sampled at the 1D nodes x.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= m:
        raise ValueError("need n > m")
    c = np.zeros((n, m + 1), dtype=float)
    c1 = 1.0
    c4 = x[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            if c3 == 0.0:
                raise ValueError("x nodes must be distinct")
            c2 *= c3
            for k in range(mn, 0, -1):
                c[i, k] = (c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k])) / c2
            c[i, 0] = (-c1 * c5 * c[i - 1, 0]) / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = (c4 * c[j, 0]) / c3
        c1 = c2
    return c[:, m].copy()

def _d_dr_nonuniform_2nd(V: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute the first radial derivative dV/dr on a nonuniform 1D grid
    using second-order finite differences.

    Parameters
    ----------
    V : ndarray
        Array of shape (Nr, Np).
    r : ndarray
        Monotonically increasing radial grid of shape (Nr,), in meters.

    Returns
    -------
    ndarray
        Array of shape (Nr, Np) containing dV/dr.
    """
    V = np.asarray(V, dtype=float)
    r = np.asarray(r, dtype=float)
    Nr, Np = V.shape
    if Nr < 3:
        raise ValueError("Need Nr >= 3 for 2nd-order radial derivative")

    dVdr = np.zeros_like(V)

    # interior: i = 1..Nr-2 (2nd-order on non-uniform grid)
    r_im1 = r[:-2]
    r_i   = r[1:-1]
    r_ip1 = r[2:]
    h1 = r_i - r_im1
    h2 = r_ip1 - r_i

    a = -h2 / (h1 * (h1 + h2))
    b = (h2 - h1) / (h1 * h2)
    c =  h1 / (h2 * (h1 + h2))

    dVdr[1:-1, :] = (a[:, None] * V[:-2, :]
                     + b[:, None] * V[1:-1, :]
                     + c[:, None] * V[2:, :])

    # left boundary i=0: 3-point one-sided (2nd-order, non-uniform)
    h0 = r[1] - r[0]
    h1b = r[2] - r[1]
    dVdr[0, :] = (-(2*h0 + h1b) / (h0 * (h0 + h1b)) * V[0, :]
                  + (h0 + h1b) / (h0 * h1b) * V[1, :]
                  - h0 / (h1b * (h0 + h1b)) * V[2, :])

    # right boundary i=Nr-1: 3-point one-sided (2nd-order, non-uniform)
    h0 = r[-2] - r[-3]
    h1b = r[-1] - r[-2]
    dVdr[-1, :] = (h1b / (h0 * (h0 + h1b)) * V[-3, :]
                   - (h0 + h1b) / (h0 * h1b) * V[-2, :]
                   + (2*h1b + h0) / (h1b * (h0 + h1b)) * V[-1, :])

    return dVdr

def compute_force_via_sample_surface_pressure_fd_stencil_3d(
    phi: np.ndarray,
    Vs: np.ndarray,             # (Nr, Np) sample surface potential at eta=0
    r0_nm: np.ndarray,          # (Nr,) in nm
    a: float, c: float,
    deta: float,                # eta step (dimensionless) times? (as in your code)
    dp: float,                  # dtheta [rad]
    npts: int = 7,
):
    """
    Compute the sample-side electrostatic force from the Maxwell-stress
    traction on the z = 0 plane.

    The normal field Ez is obtained from a one-sided finite-difference
    derivative at eta = 0 using Vs as the boundary value and phi in the
    vacuum region. The tangential fields Er and Etheta are obtained from
    derivatives of Vs(r, theta) on the sample surface.

    The stress component is evaluated as

        Tzz = eps0 / 2 * (Ez^2 - Er^2 - Etheta^2),

    and integrated over the plane to obtain the total force.
    """
    nm_to_m = 1e-9
    phi = np.asarray(phi, dtype=float)
    Vs = np.asarray(Vs, dtype=float)
    r0_nm = np.asarray(r0_nm, dtype=float)

    Nr, Nv, Np = phi.shape
    if Vs.shape != (Nr, Np):
        raise ValueError(f"Vs must have shape (Nr,Np)={(Nr,Np)}, got {Vs.shape}")

    npts = int(npts)
    if npts < 3:
        raise ValueError("npts must be >= 3")
    if Nv < (npts - 1):
        raise ValueError(f"Nv too small for npts={npts} (need Nv >= {npts-1})")

    # --- tangential fields from Vs on z=0 plane ---
    r_m = r0_nm * nm_to_m

    # dVs/dtheta (periodic central diff)
    dVs_dtheta = (np.roll(Vs, -1, axis=1) - np.roll(Vs, 1, axis=1)) / (2.0 * dp)
    Etheta = -(dVs_dtheta / r_m[:, None])  # V/m

    # dVs/dr on non-uniform r grid
    dVs_dr = _d_dr_nonuniform_2nd(Vs, r_m)  # V/m
    Er = -dVs_dr

    # --- normal field Ez from one-sided eta derivative at eta=0 ---
    # Nodes in u = eta/deta: [0,1,2,...]
    u_nodes = np.arange(npts, dtype=float)
    w_du = _fd_weights_fornberg(0.0, u_nodes, m=1)  # derivative w.r.t. u at u=0

    Ez = np.zeros((Nr, Np), dtype=float)  # V/m
    for i in range(Nr):
        xi = np.sqrt(1.0 + (r0_nm[i] / a) ** 2)
        dzdeta_nm = a * (xi + c)  # dz/deta at eta=0, in nm (since a in nm)
        for k in range(Np):
            V = np.empty(npts, dtype=float)
            V[0] = Vs[i, k]
            # phi[...,0] corresponds to eta=Delta eta
            for kk in range(1, npts):
                V[kk] = phi[i, kk-1, k]

            dV_du_0 = float(np.dot(w_du, V))
            dV_deta0 = dV_du_0 / deta          # V per eta
            dVdz0_V_per_nm = dV_deta0 / dzdeta_nm
            Ez[i, k] = -(dVdz0_V_per_nm / nm_to_m)  # V/m

    # --- Maxwell-stress traction on z=0 plane ---
    # Tzz = eps0/2 (Ez^2 - Er^2 - Etheta^2)
    Tzz_ik = 0.5 * EPS0 * (Ez*Ez - Er*Er - Etheta*Etheta)  # N/m^2

    # --- integrate F = ∫∫ p(r,theta) r dr dtheta ---
    dr = np.diff(r_m)
    r_mid = 0.5 * (r_m[:-1] + r_m[1:])

    F = 0.0
    for k in range(Np):
        p = Tzz_ik[:, k]
        p_mid = 0.5 * (p[:-1] + p[1:])
        F += np.sum(p_mid * r_mid * dr) * dp

    # disk correction for 0 <= r <= r0 (since r=0 not included)
    sum_p0 = float(np.sum(Tzz_ik[0, :] * dp))
    r0 = r_m[0]
    F += 0.5 * (r0 * r0) * sum_p0

    return Tzz_ik, -F #, Ez, Er, Etheta

def compute_force_via_tip_surface_pressure_fd_stencil_Fz_3d(
    phi: np.ndarray,            # (Nr, Nv, Np)
    r0_nm: np.ndarray,          # (Nr,)
    a: float, c: float,
    etat: float,
    deta: float,
    dp: float,
    bias: float,
    tip_mask: np.ndarray,       # (Nr, Nv) True inside metal (surface included)
    npts: int = 7,
):
    """
    Compute the tip-side force Fz from the Maxwell pressure on the tip surface.

    Notes
    -----
    - The tip surface is assumed to coincide with j = Nv - 1 for all radial columns.
    - The npts - 1 points immediately below the surface in -eta must be vacuum.
    - The usable radial columns are assumed to form a contiguous block starting
      from the smallest radius, so that the radial integration and central-disk
      correction are valid.

    Method
    ------
    - Estimate dV/deta at the tip surface using a one-sided FD stencil.
    - Convert (dV/dxi, dV/deta) to (dV/dr, dV/dz) via (J^{-1})^T.
    - On the equipotential tip surface, dV/dxi = 0.
    - Since the gradient is normal to the equipotential tip surface,
      |En| = |grad V| = sqrt((dV/dr)^2 + (dV/dz)^2).
    - Pressure p = (eps0/2) * En^2, then integrate Fz over the tip surface.
    """
    nm_to_m = 1e-9

    phi = np.asarray(phi, dtype=float)
    r0_nm = np.asarray(r0_nm, dtype=float)
    tip_mask = np.asarray(tip_mask, dtype=bool)

    Nr, Nv, Np = phi.shape
    npts = int(npts)

    if npts < 3:
        raise ValueError("npts must be >= 3")
    if Nv < npts:
        raise ValueError(f"Nv too small for npts={npts} (need Nv >= {npts})")
    if tip_mask.shape != (Nr, Nv):
        raise ValueError("tip_mask shape must match (Nr, Nv)")

    # FD weights for d/du at u=0 with nodes u=[0,-1,-2,...]
    u_nodes = -np.arange(npts, dtype=float)  # [0, -1, -2, ...]
    w_du = _fd_weights_fornberg(0.0, u_nodes, m=1)  # (npts,)

    # fixed surface index
    j_surf = Nv - 1
    j0 = Nv - npts   # inclusive
    j1 = Nv - 1      # exclusive -> covers Nv-npts ... Nv-2

    # usable columns: surface is metal and the contiguous vacuum band exists
    used = np.ones(Nr, dtype=bool)
    used &= tip_mask[:, j_surf]                  # surface must be metal
    used &= ~np.any(tip_mask[:, j0:j1], axis=1)  # Nv-npts ... Nv-2 must be vacuum

    idx_used = np.where(used)[0]
    if idx_used.size < 4:
        raise RuntimeError("Too few usable r-columns for tip force integration.")

    # xi label for each i-column (from r at eta=0)
    xi = np.sqrt(1.0 + (r0_nm / float(a)) ** 2)
    Xi_used = xi[idx_used].astype(float)

    # --- dV/deta on surface for all (i,k) at once ---
    # vacuum samples at Nv-2, Nv-3, ..., Nv-npts
    # shape: (N_used, npts-1, Np)
    vac = phi[idx_used, Nv-2 : Nv-npts-1 : -1, :]

    # build V nodes for stencil: V[0]=bias, V[1:]=vac
    # shape: (N_used, npts, Np)
    Vnodes = np.empty((idx_used.size, npts, Np), dtype=float)
    Vnodes[:, 0, :] = float(bias)
    Vnodes[:, 1:, :] = vac

    # dV/du at u=0: dot over stencil axis (npts)
    # result shape: (N_used, Np)
    dV_du0 = np.tensordot(Vnodes, w_du, axes=([1], [0]))
    dV_deta_used = dV_du0 / float(deta)  # (N_used, Np)

    # ------------------------------------------------------------------
    # Inverse-Jacobian route:
    #   [dV/dr, dV/dz]^T = (J^{-1})^T [dV/dxi, dV/deta]^T
    #
    # J = [[r_xi, r_eta],
    #      [z_xi, z_eta]]
    #
    # On the tip surface (equipotential), dV/dxi = 0.
    # ------------------------------------------------------------------
    Eta = float(etat)
    sqrt_1mEta2 = np.sqrt(1.0 - Eta * Eta)
    sqrt_Xi2m1 = np.sqrt(Xi_used * Xi_used - 1.0)

    # Jacobian elements on eta = etat
    r_xi  = a * Xi_used * sqrt_1mEta2 / sqrt_Xi2m1
    r_eta = -a * Eta * sqrt_Xi2m1 / sqrt_1mEta2
    z_xi  = a * Eta
    z_eta = a * (Xi_used + c)

    detJ = r_xi * z_eta - r_eta * z_xi
    if np.any(np.abs(detJ) < 1.0e-30):
        raise RuntimeError("Jacobian determinant too small on the tip surface.")

    # Components of J^{-1}
    xi_r  =  z_eta / detJ
    xi_z  = -r_eta / detJ
    eta_r = -z_xi  / detJ
    eta_z =  r_xi  / detJ

    # On the equipotential tip surface, tangential derivative vanishes
    dV_dxi_used = np.zeros_like(dV_deta_used)

    # Transform to physical-space derivatives [V/nm]
    dV_dr_used = xi_r[:, None] * dV_dxi_used + eta_r[:, None] * dV_deta_used
    dV_dz_used = xi_z[:, None] * dV_dxi_used + eta_z[:, None] * dV_deta_used

    # Since grad(V) is normal to the equipotential tip surface, |En| = |grad V|
    dV_dn_used = np.sqrt(dV_dr_used * dV_dr_used + dV_dz_used * dV_dz_used)  # [V/nm]
    En_used = -dV_dn_used / nm_to_m                                            # [V/m]
    p_used = 0.5 * EPS0 * En_used * En_used                                    # [Pa]

    # scatter back to full (Nr, Np)
    p_ik = np.zeros((Nr, Np), dtype=float)
    p_ik[idx_used, :] = p_used

    # radius on tip surface curve r(eta=etat) for used i
    r_tip_nm_used = float(a) * np.sqrt((Xi_used * Xi_used - 1.0) * (1.0 - Eta * Eta))
    r_m_used = r_tip_nm_used * nm_to_m

    # integrate: F = ∫∫ p(r,phi) r dr dphi  (then return -F to match your sign)
    dr_seg = np.diff(r_m_used)                        # (N_used-1,)
    r_mid = 0.5 * (r_m_used[:-1] + r_m_used[1:])     # (N_used-1,)

    p_mid = 0.5 * (p_used[:-1, :] + p_used[1:, :])   # (N_used-1, Np)
    F = np.sum(p_mid * r_mid[:, None] * dr_seg[:, None]) * float(dp)

    # central disk (0 -> r_m_used[0]) using p at smallest r
    sum_p0 = np.sum(p_used[0, :] * float(dp))
    F0 = 0.5 * (r_m_used[0] ** 2) * sum_p0
    F += F0

    return p_ik, -F

def sep_center_from_sep_stm(params: Params) -> float:
    """
    Return the oscillation-center separation under dynamic STM
    with average-current feedback.
    """
    sep_center_nm = (
        params.sep_STM_nm
        + (1.0/(2.0*params.kappa)) * np.log(i0(2.0*params.kappa*params.ampl))
        + params.z_offset
    )
    return float(sep_center_nm)

def sep_for_quasistatic_force_nm(params: Params) -> float:
    """
    Return the effective tip-sample separation used for quasi-static
    force and bias-sweep calculations.

    In the current implementation, this is taken as the oscillation-center
    separation under dynamic STM conditions.
    """
    return float(sep_center_from_sep_stm(params))

def chebyshev_nodes(N: int):
    """
    Return the Gauss-Chebyshev nodes u in [-1, 1].
    """
    i = np.arange(1, N + 1)
    return np.cos((2*i - 1) * np.pi / (2*N))

def delta_f_from_force(
    params: Params,
    bias: float,
    x_shift: float,
    N: int = 6,
    ws: Optional[Workspace] = None,
    sep_center_nm: Optional[float] = None,
):
    """
    Convert force into FM-AFM frequency shift Δf [Hz] using Gauss–Chebyshev sampling.

    Parameters
    ----------
    sep_center_nm : float, optional
        If provided, overrides the oscillation center separation [nm].
        Chebyshev sample positions are defined as sep = sep_center_nm + A*u.
    """
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    A_nm = float(params.ampl)
    A_m  = A_nm * 1e-9
    if A_m <= 0.0:
        raise ValueError("params.ampl must be > 0 for Δf calculations")
    k    = float(params.k)
    f0   = float(params.freq)

    sep0_nm = float(sep_center_from_sep_stm(params)) if sep_center_nm is None else float(sep_center_nm)
    sep0_m  = sep0_nm * 1e-9

    u = chebyshev_nodes(N)
    sep_nm_samples = (sep0_m + A_m * u) * 1e9  # [nm]

    F_tip = np.zeros(N)
    F_sample = np.zeros(N)

    for idx, sep_nm in enumerate(sep_nm_samples):
        F_tip[idx], F_sample[idx] = calc_force_single_condition(
            params,
            sep_nm=sep_nm,
            bias=bias,
            x_shift=x_shift,
            ws=ws,
            cache=None,   # Chebyshev => sep varies => no cache
        )

    df_tip    = -(f0 / (k * A_m * N)) * np.sum(F_tip * u)
    df_sample = -(f0 / (k * A_m * N)) * np.sum(F_sample * u)
    return df_tip, df_sample

def calc_force_single_condition(
    params: Params,
    sep_nm: float,
    bias: Optional[float] = None,
    x_shift: Optional[float] = None,
    ws: Optional[Workspace] = None,
    cache: Optional[SolverCache3D] = None,
):
    """Solve the potential for one condition and return (F_tip, F_sample) [N].

    Parameters
    ----------
    sep_nm : float
        Instantaneous tip-sample separation used by the solver [nm].
    bias : float, optional
        Effective bias applied to the tip Dirichlet boundary.
    x_shift : float, optional
        Interface position for the built-in CPD-step model.
    cache : SolverCache3D, optional
        Reusable cache for repeated solves at fixed separation. If provided,
        `sep_nm` must match the separation used to build the cache.
    """
    sep_nm = float(sep_nm)
    bias_use = float(params.bias) if bias is None else float(bias)
    x_use    = float(params.x_shift) if x_shift is None else float(x_shift)

    if ws is None:
        ws = alloc_workspace(params)

    if cache is not None:
        sep_cache_nm = float(cache.d.z0 + cache.d.sprime) # reconstruct sep_nm from cached geometry
        if abs(sep_nm - sep_cache_nm) > 1e-12:
            raise ValueError(
                f"cache sep mismatch: sep_nm={sep_nm:.6f} nm vs cache={sep_cache_nm:.6f} nm. "
                "Rebuild cache for this sep."
            )
        Nr, Nv, Np, deta, dp, d = solve_potential_3d_cached(
            params, ws, cache,
            bias=bias_use,
            x_shift=x_use,
        )
    else:
        Nr, Nv, Np, deta, dp, d = solve_potential_3d_base(
            params, ws, sep_nm,
            bias=bias_use,
            x_shift=x_use,
        )

    # force at sample surface
    _, F_sample = compute_force_via_sample_surface_pressure_fd_stencil_3d(
        phi=ws.phi[0, :Nr, :Nv, :Np],
        Vs=ws.Vs[:Nr, :Np],
        r0_nm=ws.r[:Nr],
        a=d.a, c=d.c, deta=deta, dp=dp
    )

    # force at tip surface
    _, F_tip = compute_force_via_tip_surface_pressure_fd_stencil_Fz_3d(
        phi=ws.phi[0, :Nr, :Nv, :Np],
        r0_nm=ws.r[:Nr],
        a=d.a, c=d.c, etat=d.etat, deta=deta, dp=dp,
        bias=bias_use,
        tip_mask=ws.tip[:Nr, :Nv]
    )

    return F_tip, F_sample

def calc_force_bias_curve_approx_3d(
    params: Params,
    points: int,
    *,
    sep_nm: Optional[float] = None,
    x_shift: Optional[float] = None,
    ws: Optional[Workspace] = None,
    cache: Optional[SolverCache3D] = None,
):
    """
    Approximate force–bias curve using a quadratic fit (5 samples).
    Returns: (bias_app, F_fit, V_cpd_app)
    """
    # decide fixed separation for this sweep
    sep0 = sep_center_from_sep_stm(params) if sep_nm is None else float(sep_nm)

    # bias axis shown to user
    Nsub = 5
    bias_app_sub = np.linspace(params.bias_start, params.bias_end, Nsub)

    # internal bias fed to solver (your current convention)
    bias_eff_sub = bias_app_sub - params.CPD

    F_tip_sub = np.zeros(Nsub)
    F_sample_sub = np.zeros(Nsub)

    for i, b_eff in enumerate(bias_eff_sub):
        F_tip_sub[i], F_sample_sub[i] = calc_force_single_condition(
            params,
            sep_nm=sep0,
            bias=b_eff,
            x_shift=x_shift,
            ws=ws,
            cache=cache,
        )

    # choose which force to fit
    y_sub = F_tip_sub if params.force_mode == "tip" else F_sample_sub

    a, b, c = np.polyfit(bias_app_sub, y_sub, 2)

    bias_app = np.linspace(params.bias_start, params.bias_end, points)
    F_fit = a * bias_app**2 + b * bias_app + c
    V_cpd_app = -b / (2.0 * a)

    return bias_app, F_fit, V_cpd_app

def calc_force_bias_curve(
    params: Params,
    *,
    sep_nm: Optional[float] = None,
    x_shift: Optional[float] = None,
    ws: Optional[Workspace] = None,
    cache: Optional[SolverCache3D] = None,
):
    """
    Return force vs applied bias curve.
    Returns: (bias_app, F)
    """
    sep0 = sep_center_from_sep_stm(params) if sep_nm is None else float(sep_nm)

    bias_app = np.linspace(params.bias_start, params.bias_end, params.n_bias_points)
    bias_eff = bias_app - params.CPD

    F_tip = np.empty(params.n_bias_points)
    F_sample = np.empty(params.n_bias_points)

    #for i, b_eff in enumerate(bias_eff):
    for i, b_eff in enumerate(tqdm(bias_eff, desc="F-V")):
        F_tip[i], F_sample[i] = calc_force_single_condition(
            params,
            sep_nm=sep0,
            bias=b_eff,
            x_shift=x_shift,
            ws=ws,
            cache=cache,
        )

    F = F_tip if params.force_mode == "tip" else F_sample
    return bias_app, F

def calc_delta_f_V_curve_approx_3d(
    params: Params,
    points: int,
    *,
    x_shift: Optional[float] = None,
    ws: Optional[Workspace] = None,
    sep_center_nm: Optional[float] = None,
):
    """
    Approximate Δf–V curve using a parabolic model around CPD (5 bias samples).
    Returns: (bias_app, df_fit, V_cpd_app)

    Notes
    -----
    - bias_app is the displayed bias axis.
    - bias fed to solver is bias_eff = bias_app - params.CPD (kept for backward compatibility).
    - sep_center_nm overrides the oscillation center used for Chebyshev sampling.
    """

    Nsub = 5
    bias_app_sub = np.linspace(params.bias_start, params.bias_end, Nsub)
    bias_eff_sub = bias_app_sub - params.CPD

    df_tip_sub = np.zeros(Nsub)
    df_sample_sub = np.zeros(Nsub)

    for i, b_eff in enumerate(bias_eff_sub):
        df_tip_sub[i], df_sample_sub[i] = delta_f_from_force(
            params,
            bias=b_eff,
            x_shift=x_shift,
            N=6,
            ws=ws,
            sep_center_nm=sep_center_nm,
        )

    y_sub = df_tip_sub if params.force_mode == "tip" else df_sample_sub

    a, b, c = np.polyfit(bias_app_sub, y_sub, 2)

    bias_app = np.linspace(params.bias_start, params.bias_end, points)
    df_fit = a * bias_app**2 + b * bias_app + c
    V_cpd_app = -b / (2.0 * a)

    return bias_app, df_fit, V_cpd_app

def calc_delta_f_V_curve(
    params: Params,
    *,
    x_shift: Optional[float] = None,
    ws: Optional[Workspace] = None,
    sep_center_nm: Optional[float] = None,
):
    """
    Compute the Δf-V curve over the applied-bias range.

    Returns
    -------
    bias_app : ndarray
        Applied-bias axis.
    df : ndarray
        Frequency-shift curve.

    Notes
    -----
    - The bias passed to the solver is the effective bias
      `bias_eff = bias_app - params.CPD`.
    - `sep_center_nm` overrides the oscillation center used for
      Gauss-Chebyshev sampling.
    """
    bias_app = np.linspace(params.bias_start, params.bias_end, params.n_bias_points)
    bias_eff = bias_app - params.CPD

    df_tip = np.empty(params.n_bias_points)
    df_sample = np.empty(params.n_bias_points)

    #for i, b_eff in enumerate(bias_eff):
    for i, b_eff in enumerate(tqdm(bias_eff, desc="F-V")):
        df_tip[i], df_sample[i] = delta_f_from_force(
            params,
            bias=b_eff,
            x_shift=x_shift,
            N=6,
            ws=ws,
            sep_center_nm=sep_center_nm,
        )

    df = df_tip if params.force_mode == "tip" else df_sample
    return bias_app, df

def calc_CPD_profile_from_force(params: Params):
    """
    Compute the CPD profile extracted from approximate force-bias curves
    while sweeping the interface position.

    Notes
    -----
    - The separation is kept fixed during the sweep.
    - A single solver cache is reused because only `x_shift` changes.
    - The returned x-axis is `-x_arr`, following the current sign convention
      for converting interface position to tip lateral coordinate.
    """
    if params.dx <= 0:
        raise ValueError("params.dx must be > 0")
    
    x_arr = np.arange(params.x_start, params.x_end + 0.5*params.dx, params.dx)
    CPD_at_x = np.zeros(len(x_arr))

    ws = alloc_workspace(params)

    # Cache reuse assumes a fixed separation: determine the separation used here and build the cache for it.
    sep0 = sep_center_from_sep_stm(params)
    cache = prepare_solver_cache_3d(params, sep_nm=sep0)
    
    n_fit_points = 128
    #for i, x in enumerate(x_arr):
    for i, x in enumerate(tqdm(x_arr, desc="CPD profile (force)")):
        _, _, CPD_at_x[i] = calc_force_bias_curve_approx_3d(
            params,
            n_fit_points,
            sep_nm=sep0,
            x_shift=x,
            ws=ws,
            cache=cache,
        )
    # Convert swept interface positions to the corresponding tip lateral coordinates.
    return -x_arr, CPD_at_x

def calc_CPD_profile_from_delta_f(params: Params):
    if params.dx <= 0:
        raise ValueError("params.dx must be > 0")
    
    x_arr = np.arange(params.x_start, params.x_end + 0.5*params.dx, params.dx)
    CPD_at_x = np.zeros(len(x_arr))

    ws = alloc_workspace(params)
    
    n_fit_points = 128
    #for i, x in enumerate(x_arr):
    for i, x in enumerate(tqdm(x_arr, desc="CPD profile (Δf)")):
        _, _, CPD_at_x[i] = calc_delta_f_V_curve_approx_3d(
            params,
            n_fit_points,
            x_shift=x,
            ws=ws,
        )
    # Convert swept interface positions to the corresponding tip lateral coordinates.
    return -x_arr, CPD_at_x

def calc_CPD_profile_from_delta_f_approx(params: Params):
    """CPD profile using the small-amplitude approximation Δf ∝ dF/dz (central difference on F).

    Implementation
    --------------
    - Compute F–V at (sep0+dz) and (sep0-dz) by parabolic approximation (5 samples in bias).
    - Form dF/dz and extract CPD from quadratic fit of dF/dz vs bias_app.

    Notes
    -----
    - Uses two caches (plus/minus) because cache is only valid for a fixed sep.
    - Reuses one workspace across the whole x sweep.
    """
    if params.dx <= 0:
        raise ValueError("params.dx must be > 0")
    if params.dz <= 0:
        raise ValueError("params.dz must be > 0")
    
    x_arr = np.arange(params.x_start, params.x_end + 0.5 * params.dx, params.dx)
    CPD_at_x = np.zeros(len(x_arr))

    dz = params.dz  # [nm] central difference step

    # choose a single reference separation for this profile
    sep0 = sep_center_from_sep_stm(params)
    if (sep0 - dz) <= 0.0:
        raise ValueError(
            f"sep0 - dz must be > 0 (got sep0={sep0:.6f} nm, dz={dz:.6f} nm)"
        )

    ws = alloc_workspace(params)

    # build caches for fixed sep0±dz
    cache_plus  = prepare_solver_cache_3d(params, sep_nm=sep0 + dz)
    cache_minus = prepare_solver_cache_3d(params, sep_nm=sep0 - dz)

    n_fit_points = 128

    #for i, x in enumerate(x_arr):
    for i, x in enumerate(tqdm(x_arr, desc="CPD profile (Δf approx)")):
        # F–V at sep0+dz
        bias_app, F_plus, _ = calc_force_bias_curve_approx_3d(
            params,
            n_fit_points,
            sep_nm=sep0 + dz,
            x_shift=x,
            ws=ws,
            cache=cache_plus
        )
        # F–V at sep0-dz
        _, F_minus, _ = calc_force_bias_curve_approx_3d(
            params,
            n_fit_points,
            sep_nm=sep0 - dz,
            x_shift=x,
            ws=ws,
            cache=cache_minus
        )

        dFdz = (F_plus - F_minus) / (2.0 * dz)

        a, b, c = np.polyfit(bias_app, dFdz, 2)
        CPD_at_x[i] = -b / (2.0 * a)

    return -x_arr, CPD_at_x

def make_fname_header(params, sep_nm: Optional[float] = None):
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    if params.mode == "spec":
        spec_mode = params.spec_mode
        if spec_mode == "F-V":
            prefix = "F-V_"
            title = "Force vs bias"
            mode = spec_mode
            variable_name = "Bias [V]"
            channel_name = "Electrostatic force [N]"
        elif spec_mode == "F-V_approx":
            prefix = "F-V_approx_"
            title = "Force vs bias"
            mode = spec_mode
            variable_name = "Bias [V]"
            channel_name = "Electrostatic force [N]"
        elif spec_mode == "df-V":
            prefix = "df-V_"
            title = "Frequency shift vs bias"
            mode = spec_mode
            variable_name = "Bias [V]"
            channel_name = "Frequency shift [Hz]"
        elif spec_mode == "df-V_approx":
            prefix = "df-V_approx_"
            title = "Frequency shift vs bias"
            mode = spec_mode
            variable_name = "Bias [V]"
            channel_name = "Frequency shift [Hz]"
        else:
            raise ValueError(f"Unknown params.spec_mode: {spec_mode}")

    elif params.mode == "CPD_prof":
        CPD_mode = params.CPD_mode
        if CPD_mode == "F":
            prefix = "CPD-profile_from_F_"
            title = "CPD profile"
            mode = CPD_mode
            variable_name = "Distance [nm]"
            channel_name = "CPD [V]"
        elif CPD_mode in ("df", "df_approx"):
            prefix = "CPD-profile_from_df_"
            title = "CPD profile"
            mode = CPD_mode
            variable_name = "Distance [nm]"
            channel_name = "CPD [V]"
        else:
            raise ValueError(f"Unknown params.CPD_mode: {CPD_mode}")

    else:
        raise ValueError(f"Unknown params.mode: {params.mode}")

    def fnum(x: float, nd: int) -> str:
        return f"{x:.{nd}f}"

    def fmt_value(value) -> str:
        """Format parameter values in a reproducible, copy-friendly form."""
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)

    sep0 = float(sep_for_quasistatic_force_nm(params)) if sep_nm is None else float(sep_nm)
    sep_center = float(sep_center_from_sep_stm(params))

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")

    fname = data_dir / (
        f"{prefix}"
        f"sep{fnum(sep0, 2)}nm_"
        f"rad{fnum(params.rad, 1)}nm_"
        f"angle{fnum(params.open_angle, 1)}deg_"
        f"{ts}.dat"
    )

    header_lines = [
        f"METALTIP output: {title}",
        f"datetime: {now.isoformat(timespec='seconds')}",
        f"output mode: {params.mode}",
        f"output submode: {mode}",
        "",
        "[DERIVED / EFFECTIVE VALUES]",
        f"sep_center from dynamic STM (nm): {sep_center:.12g}",
        #f"separation used in this calculation (nm): {sep0:.12g}",
        "",
        "[INPUT PARAMETERS]",
    ]

    # Save every field defined in Params.  This avoids missing newly added
    # parameters in the output header when Params is extended in the future.
    for name in params.__dataclass_fields__:
        header_lines.append(f"{name}: {fmt_value(getattr(params, name))}")

    header_lines.extend([
        "",
        "[DATA]",
        f"{variable_name}   {channel_name}",
    ])
    header = "\n".join(header_lines) + "\n"

    return fname, header

def save_xy(fname, header, x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    with open(fname, "w", encoding="utf-8") as f:
        f.write(header)
        for xi, yi in zip(x, y):
            f.write(f"{xi:.12g}\t{yi:.12e}\n")

def plot_xy(params, x, y, xlabel="Bias (V)", ylabel=""):
    import matplotlib.pyplot as plt

    if params.mode == "CPD_prof":
        xlabel = "Distance (nm)"
        ylabel = "CPD (V)"

    plt.plot(x, y, ".")
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    params = Params()
    # ============================================================
    # Usage notes
    # ============================================================
    # This script supports two top-level modes:
    #
    #   1) mode = "spec"
    #      Compute a spectrum at a fixed lateral position.
    #      Available spectrum types are:
    #          - "F"          : electrostatic force at a single bias
    #          - "F-V_approx" : force–bias curve from a 5-point quadratic fit
    #          - "F-V"        : full force–bias curve
    #          - "df-V_approx": Δf–V curve from a 5-point quadratic fit
    #          - "df-V"       : full Δf–V curve
    #
    #   2) mode = "CPD_prof"
    #      Compute a lateral CPD profile by sweeping x_shift and extracting
    #      the local CPD at each lateral position.
    #      Available CPD extraction modes are:
    #          - "F"         : CPD from force values evaluated at 5 bias points
    #                          and fitted with a parabola
    #          - "df"        : CPD from finite-amplitude Δf values evaluated at
    #                          5 bias points and fitted with a parabola
    #          - "df_approx" : CPD from the small-amplitude approximation
    #                          Δf ≈ -(f0 / 2k) dF/dz
    #
    # General notes
    # -------------
    # - For force and F–V calculations, the tip–sample separation is taken as
    #   the oscillation-center separation under dynamic STM conditions
    #   (average-current feedback), derived from params.sep_STM_nm.
    # - In CPD_prof mode, both "F" and "df" use a 5-point parabolic
    #   approximation for CPD extraction rather than a full bias sweep.
    # - In CPD-profile calculations, the lateral coordinate is scanned by
    #   sweeping x_shift, which shifts the prescribed surface-potential pattern
    #   relative to the tip.
    # - The "df_approx" mode requires a finite central-difference step `dz`
    #   to evaluate dF/dz. If `dz` is chosen too small, the resulting CPD
    #   profile may become noisy. A value of 0.02 nm or larger is recommended
    #   in most cases.
    # ============================================================
    
    if params.mode == "spec":

        if params.spec_mode == "F":
            sep0 = sep_center_from_sep_stm(params)
            bias_eff = params.bias - params.CPD
            F_tip, F_sample = calc_force_single_condition(params, sep_nm=sep0, bias=bias_eff, x_shift=params.x_shift)
            print("\n==============================")
            print("F_tip =", F_tip)
            print("F_sample =", F_sample)
            print("==============================")

        elif params.spec_mode == "F-V_approx":
            bias, F, _ = calc_force_bias_curve_approx_3d(params,128)

            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, bias, F)
            if params.plot_data:
                plot_xy(params, bias, F, ylabel="Force (N)")

        #===================
        # Force–V curve (full calculation)
        #===================
        # In this mode, the electrostatic force is computed *directly* at every bias point
        # in the specified bias range (bias_start → bias_end). No quadratic approximation
        # is used here:

        elif params.spec_mode == "F-V":
            bias, F = calc_force_bias_curve(params)
            
            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, bias, F)
            if params.plot_data:
                plot_xy(params, bias, F, ylabel="Force (N)")

        elif params.spec_mode == "df-V_approx":
            if params.ampl==0:
                raise ValueError("Amplitude is 0. Finite-amplitude Δf formula is undefined at A=0.")
            bias, df, _ = calc_delta_f_V_curve_approx_3d(params,128)
            
            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, bias, df + params.df_offset)
            if params.plot_data:
                plot_xy(params, bias, df + params.df_offset, ylabel="Frequency shift (Hz)")

        elif params.spec_mode == "df-V":
            if params.ampl==0:
                raise ValueError("Amplitude is 0. Finite-amplitude Δf formula is undefined at A=0.")
            bias, df = calc_delta_f_V_curve(params)
            
            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, bias, df + params.df_offset)
            if params.plot_data:
                plot_xy(params, bias, df + params.df_offset, ylabel="Frequency shift (Hz)")
        
        else:
            #print(f"Unknown spec_mode: {params.spec_mode}")
            raise ValueError(f"Unknown spec_mode: {params.spec_mode}")

    elif params.mode == "CPD_prof":

        # IMPORTANT:
        # - For speed, the F–V curve is obtained using the *5-point approximation*:
        #   the force is evaluated at only 5 bias points and then fitted with a quadratic
        #   function to reconstruct F(V) and determine CPD.
        # - Because this is an approximation, it is recommended to validate it beforehand:
        #   run the corresponding spectrum calculation (spec_mode) at a few representative
        #   positions and compare the 5-point approximated curve against the full (exact)
        #   calculation to confirm that the approximation is accurate for your settings
        #   (bias range, distance, geometry, etc.).

        if params.CPD_mode == "F":
            x, V_cpd = calc_CPD_profile_from_force(params)
            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, x, V_cpd)
            if params.plot_data:
                plot_xy(params, x, V_cpd, ylabel="CPD (V)")

        #===================
        # CPD profile from Δf-based CPD extraction (CPD_mode == "df")
        #===================
        # IMPORTANT:
        # - Because Δf is computed via Gauss–Chebyshev sampling (force evaluated at multiple
        #   separations over an oscillation cycle), this mode is more computationally
        #   expensive than the force-based CPD extraction.
        # - As with the force-based mode, it is recommended to validate the 5-point
        #   approximation beforehand: run the spectrum calculation (spec_mode) at a few
        #   representative positions.
        
        elif params.CPD_mode == "df":
            if params.ampl==0:
                raise ValueError("Amplitude is 0. Finite-amplitude Δf formula is undefined at A=0.")

            x, V_cpd = calc_CPD_profile_from_delta_f(params)
            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, x, V_cpd)
            if params.plot_data:
                plot_xy(params, x, V_cpd, ylabel="CPD (V)")

        #===================
        # CPD profile from Δf (small-amplitude approximation) (CPD_mode == "df_approx")
        #===================
        #
        # - Instead of evaluating Δf via full Gauss–Chebyshev sampling over the oscillation,
        #   we approximate the FM-AFM frequency shift by the force gradient at the oscillation
        #   center:
        #
        #     Δf ≈ -(f0 / (2k)) * dF/dz   (small-amplitude limit)
        #
        # - Practically, dF/dz is estimated by a central finite difference around the
        #   oscillation center separation z_center:
        #
        #     dF/dz ≈ [F(z_center + dz) - F(z_center - dz)] / (2dz)
        #
        
        elif params.CPD_mode == "df_approx":
            x, V_cpd = calc_CPD_profile_from_delta_f_approx(params)
            if params.save_data:
                fname, header = make_fname_header(params)
                save_xy(fname, header, x, V_cpd)
            if params.plot_data:
                plot_xy(params, x, V_cpd, ylabel="CPD (V)")

    else:
        raise ValueError(f"Unknown computation mode: {params.mode}")
        #print(f"Unknown mode: {params.mode}")
       
