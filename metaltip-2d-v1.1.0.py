"""
METALTIP (2D)

Electrostatic simulation code for a metallic tip–sample system
in generalized prolate spheroidal coordinates.

Version 1.1.0 - written by Nobuyuki Ishida, Mar 2026

This code is based on Feenstra's SEMITIP framework and adopts its
variable-ξ finite-difference strategy, while being independently
re-implemented here for metallic tip–metallic sample configurations.

In addition to solving the vacuum electrostatic potential, the code
computes electrostatic force and frequency shift under experimentally
relevant AFM/KPFM conditions, including bias-dependent F–V and Δf–V
characteristics.

Features
--------
- Solves Laplace's equation in vacuum
- Sample boundary: grounded Dirichlet condition (phi = 0 at eta = 0)
- Tip boundary: Dirichlet condition (phi = bias at eta = etat)
- Gauss–Seidel / SOR relaxation with staged grid refinement
- Computes electrostatic force from the Maxwell stress
- Computes frequency shift (Δf) from the force
- Supports bias-dependent F–V and Δf–V calculations
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
        
#define physical constants
pi = np.pi
EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]

@dataclass
class Params:
    """
    Parameters controlling the METALTIP 2D calculation.
    mode
        "F"
            Compute the electrostatic force at a single bias value.
        "F-V_approx"
            Compute an approximate force–bias curve over the specified bias range.
            The force is evaluated explicitly only at single bias value, and the
            remaining points are obtained from a parabolic fit.
        "F-V"
            Compute the full force–bias curve over the specified bias range by
            evaluating the force at all bias points defined by `npoints`.
        "df-V_approx"
            Same as "F-V_approx", but for frequency shift (Δf) instead of force.
        "df-V"
            Same as "F-V", but for frequency shift (Δf) instead of force.

    force_mode
        Surface used for Maxwell-stress force integration:
        "sample" 
            integrate electrostatic pressure on the grounded sample surface.
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

        For force calculations, a fixed distance can be imposed by setting
        `ampl = 0` and adjusting `sep_STM_nm` directly (plus `z_offset`, if
        needed). For `Δf` calculations, `ampl` must be nonzero; use
        `sep_STM_nm` and/or `z_offset` to tune the center distance instead.
    """

    # --- mode ---
    mode:         str = "df-V"    # calculation mode: "F", "F-V_approx", F-V, "df-V", or "df-V_approx"
    force_mode:   str = "sample"  # Maxwell-stress integration surface: "sample" or "tip"
    save_data:   bool = True      # save calculated data to file
    plot_data:   bool = True      # generate plots
    
    # --- SOR relaxation parameters ---
    omega:      float = 1.85     # SOR relaxation factor used in the Gauss–Seidel iteration

    # --- geometry / physics ---
    open_angle: float = 20.0     # full opening angle of the hyperbolic tip (degrees)
    sep_STM_nm: float = 1.0      # tip–sample separation corresponding in static STM mode (nm); see the docstring for details
    rad:        float = 10.0     # tip radius (nm)
    rad2:       float = 0.0      # reserved for future use; keep 0.0 in v1

    bias:       float = 1.0      # bias used for single-bias calculations (mode="F")
    bias_start: float = -1.0     # start of the bias sweep range
    bias_end:   float = 1.0      # end of the bias sweep range
    npoints:    int   = 21       # number of bias points in the sweep

    # --- grid / solver ---
    Nrin:     int   = 16        # initial number of radial grid points
    Nvin:     int   = 8         # initial number of grid points into the vacuum
    scale:    float = 0.5       # grid-scaling factor used at each refinement step
    nstep:    int   = 6         # number of staged grid-refinement steps
    ep0:      float = 1.0e-4    # convergence tolerance for the first step
    ep_final: float = 1.0e-6    # convergence tolerance for the final step
    itmax:    int   = 100000    # maximum number of iterations

    # --- average tunneling current / dynamic STM parameters ---
    kappa:    float = 11.0      # tunneling-current decay constant (1/nm)
    ampl:     float = 0.5       # oscillation amplitude of the AFM tip (nm)
    freq:     float = 30.0e3    # resonance frequency of the force sensor (Hz)
    k:        float = 1825      # spring constant of the force sensor (N/m)
    z_offset: float = 0.0       # additional tip displacement along z applied during spectroscopy (nm)

    # --- experiment ---
    CPD:       float = 0.0      # contact potential difference (V)
    df_offset: float = 0.0      # frequency-shift offset at the parabola vertex (Hz)

@dataclass
class Workspace:
    r: np.ndarray    # radial grid points on the sample surface
    dr: np.ndarray   # radial grid spacing
    tip: np.ndarray  # boolean mask indicating the tip region on the grid
    phi: np.ndarray  # work array for the electrostatic potential
    
def alloc_workspace(params: Params) -> Workspace:
    """
    Allocate workspace arrays at the maximum grid size required over all
    refinement steps.
    The arrays are preallocated once and then reused during the calculation
    to avoid repeated memory allocation.
    """
    Nr_max = params.Nrin * (2 ** (params.nstep - 1))
    Nv_max = params.Nvin * (2 ** (params.nstep - 1))
    return Workspace(
        r=np.zeros(Nr_max),
        dr=np.zeros(Nr_max),
        tip=np.zeros((Nr_max, Nv_max), dtype=np.bool_),
        phi=np.zeros((2, Nr_max, Nv_max), dtype=np.float64),
    )

@dataclass
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
def build_grid_and_tip(phi, tip, r, dr, Nr, Nv, dr0,
                       a, c, etat, rad2, bias):
    """
    Build the radial grid (`r` and `dr`), set the tip mask, and apply
    the Dirichlet boundary condition (`phi = bias`) on the tip region
    and the top boundary.
    """

    deta = etat / float(Nv)
    for i in range(Nr):
        r[i] = (2.0 * Nr * dr0 / pi) * np.tan(pi * (i + 0.5) / (2.0 * Nr))
        dr[i] = dr0 / (np.cos(pi * (i + 0.5) / (2.0 * Nr)) ** 2)

        x2m1 = (r[i] / a) ** 2
        xi = np.sqrt(1.0 + x2m1)

        # interior eta layers: j = 0..Nv-2 correspond to J = 1..Nv-1
        # j = 0 (J = 1) is the first vacuum grid point, not the sample surface.
        for j in range(Nv - 1):
            J = j + 1
            eta = J * deta

            z = a * eta * (xi + c)
            zp = z * (J + 0.5) / float(J)
            rp = a * np.sqrt(x2m1 * (1.0 - eta * eta))

            # NOTE (v1.0):
            # For the current fixed geometry, the tip surface coincides with the
            # outermost eta-layer (j = Nv - 1), so this inside-tip test is
            # effectively redundant. It is retained for possible future extensions
            # in which the tip boundary may intersect interior layers.
            ztip = (a * etat *
                    (np.sqrt(1.0 + rp * rp / ((1.0 - etat * etat) * a * a)) + c)
                    - tip_profile(rp, rad2))
            if zp > ztip:
                tip[i, j] = True
                phi[0, i, j] = bias
                phi[1, i, j] = bias
            else:
                tip[i, j] = False

        # top boundary: j = Nv - 1
        tip[i, Nv - 1] = True
        phi[0, i, Nv - 1] = bias
        phi[1, i, Nv - 1] = bias

@njit(cache=True, nogil=True)
def set_initial_guess(phi, tip, Nr, Nv, deta, etat, bias):
    """
    Initialize the potential in the vacuum region below the tip using an
    analytical 1D solution along the eta direction.
    """
    cetat = np.log((1.0 + etat) / (1.0 - etat))

    for i in range(Nr):
        # find the highest vacuum index below the tip
        jsave = 0
        for j in range(Nv - 1, -1, -1):
            if not tip[i, j]:
                jsave = j
                break
        for j in range(jsave, -1, -1):
            eta = (j + 1) * deta * float(Nv) / float(jsave + 2)
            val = bias * np.log((1.0 + eta) / (1.0 - eta)) / cetat
            phi[0, i, j] = val
            phi[1, i, j] = val

def build_xi_dxi(r_nm: np.ndarray, a_nm: float):
    """
    Construct the Feenstra/SEMITIP-style xi grid and its spacing.

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
    
def precompute_stencil_coeffs_feenstra_vargrid(xi: np.ndarray,
                                               dxi: np.ndarray,
                                               Nr: int, Nv: int,
                                               a: float, c: float,
                                               deta: float):
    """
    Precompute stencil coefficients for the Gauss–Seidel update of the
    vacuum Laplace equation on the variable-xi grid.

    The discretization follows the Feenstra/SEMITIP-style treatment of
    xi derivatives on a nonuniform grid (SEMITIP2-6.1).

    Notes
    -----
    - The axis point (i = 0) is handled separately during the iteration.
      The corresponding coefficient row is filled with zeros here.
    - Coefficients are precomputed for j = 0..Nv-2; the top boundary
      j = Nv-1 is treated as a Dirichlet boundary.
    - At the outer radial boundary (i = Nr-1), the forward xi spacing is
      taken equal to the local spacing.

    Returns
    -------
    Ap, Am : ndarray
        Coefficients for the radial neighbors (i+1, j) and (i-1, j).
    Bp, Bm : ndarray
        Coefficients for the eta neighbors (i, j+1) and (i, j-1).
    C : ndarray
        Coefficient of the mixed derivative term.
    inv_den : ndarray
        Inverse of the diagonal denominator used in the GS update.
    """
    xi = np.asarray(xi, dtype=np.float64)
    dxi = np.asarray(dxi, dtype=np.float64)

    Ap = np.zeros((Nr, Nv-1), dtype=np.float64)  # coeff for (i+1, j)
    Am = np.zeros((Nr, Nv-1), dtype=np.float64)  # coeff for (i-1, j)
    Bp = np.zeros((Nr, Nv-1), dtype=np.float64)  # coeff for (i, j+1)
    Bm = np.zeros((Nr, Nv-1), dtype=np.float64)  # coeff for (i, j-1)
    C  = np.zeros((Nr, Nv-1), dtype=np.float64)  # mixed term coeff
    inv_den = np.zeros((Nr, Nv-1), dtype=np.float64)

    inv_deta2 = 1.0/(deta*deta)
    inv_2deta = 0.5/deta

    # eta = (j+1)*deta (j=0..Nv-2); eta=0 plane is treated via ghost at j=0 boundary handling
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

    # NOTE: now includes i=Nr-1
    for i in range(0, Nr):  # precompute for i=0..Nr-1 (Nr-1 is outer boundary)
        xi_i = xi[i]
        xi2  = xi_i*xi_i
        xi3  = xi2*xi_i
        xi4  = xi3*xi_i
        xi5  = xi4*xi_i
        xi2m1 = xi2 - 1.0

        # variable grid spacings
        # dxm: xi[i] - xi[i-1]
        # dxp: xi[i+1] - xi[i]
        dxm = dxi[i]
        dxp = dxi[i+1] if (i+1) < Nr else dxi[i]  # rmax: dxp = dxm
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

            den = t1 * 2.0 * ((inv_dxp + inv_dxm) * inv_dxpm) + 2.0 * t2 * inv_deta2
            if den <= 0.0 or (not np.isfinite(den)):
                den = 1.0e-30

            Ap[i, j] = ap
            Am[i, j] = am
            Bp[i, j] = bp
            Bm[i, j] = bm
            C[i, j]  = ccoef
            inv_den[i, j] = 1.0/den

    return Ap, Am, Bp, Bm, C, inv_den

@njit(cache=True, nogil=True)
def iterate_potential_feenstra_vargrid_precomp(phi, tip,
                                               Ap, Am, Bp, Bm, C, inv_den,
                                               Nr, Nv,
                                               ep, itmax,
                                               omega=1.0,
                                               enforce_rmax=False):
    """
    Gauss–Seidel / SOR relaxation for the vacuum potential using precomputed
    stencil coefficients.

    The update is applied only to non-tip vacuum nodes. The symmetry axis
    (i = 0) is handled using the SEMITIP ghost formula, the grounded sample
    boundary at eta = 0 is imposed through the j = 0 ghost treatment, and
    the outer radial boundary is handled by a ghost-point update at i = Nr-1.

    Notes
    -----
    `enforce_rmax=True` is retained only for internal testing of a simpler
    outer-boundary closure and is not recommended for routine calculations.
    """

    err_prev = 1.0e30

    for it in range(itmax):
        err_max = 0.0

        if enforce_rmax:
            for j in range(Nv):
                phi[0, Nr-1, j] = phi[0, Nr-2, j]
        else:
            i = Nr - 1
            imi = Nr - 2
            for j in range(Nv-1):
                if tip[i, j]:
                    continue

                old = phi[0, i, j]
                ap = Ap[i, j]
                am = Am[i, j]
                bp = Bp[i, j]
                bm = Bm[i, j]
                cc = C[i, j]

                vim = phi[0, imi, j]
                vjp1 = phi[0, i, j+1]
                vip_jp1 = vjp1              # ghost (i+1)->i
                vim_jp1 = phi[0, imi, j+1]

                if j == 0:
                    vjm1 = 0.0
                    vip_jm1 = 0.0
                    vim_jm1 = 0.0
                else:
                    vjm1 = phi[0, i, j-1]
                    vip_jm1 = vjm1           # ghost (i+1)->i
                    vim_jm1 = phi[0, imi, j-1]

                rhs = (
                    am*vim
                    + bp*vjp1 + bm*vjm1
                    + cc*(vip_jp1 - vip_jm1 - vim_jp1 + vim_jm1)
                )

                den = 1.0 / inv_den[i, j]
                den_eff = den - ap
                if den_eff <= 1.0e-30 or (not np.isfinite(den_eff)):
                    den_eff = 1.0e-30

                new_gs = rhs / den_eff
                new = old + omega*(new_gs - old)

                phi[0, i, j] = new
                diff = abs(new - old)
                if diff > err_max:
                    err_max = diff

        # ---- i = 0 (axis) : do on-the-fly SEMITIP ghost (cheap) ----
        i = 0
        ipi = 1
        for j in range(Nv-1):
            if tip[i, j]:
                continue

            old = phi[0, i, j]

            vim   = (9.0*phi[0, i, j] - phi[0, ipi, j]) * 0.125  # /8
            vip   = phi[0, ipi, j]

            if j == 0:
                vjm1 = 0.0
                vip_jm1 = 0.0
                vim_jm1 = 0.0
            else:
                vjm1 = phi[0, i, j-1]
                vip_jm1 = phi[0, ipi, j-1]
                vim_jm1 = (9.0*phi[0, i, j-1] - phi[0, ipi, j-1]) * 0.125

            vjp1 = phi[0, i, j+1]
            vip_jp1 = phi[0, ipi, j+1]
            vim_jp1 = (9.0*phi[0, i, j+1] - phi[0, ipi, j+1]) * 0.125

            ap = Ap[i, j]
            am = Am[i, j]
            bp = Bp[i, j]
            bm = Bm[i, j]
            cc = C[i, j]

            temp = (
                ap*vip + am*vim
                + bp*vjp1 + bm*vjm1
                + cc*(vip_jp1 - vip_jm1 - vim_jp1 + vim_jm1)
            )
            new_gs = temp * inv_den[i, j]
            new = old + omega*(new_gs - old)

            phi[0, i, j] = new
            diff = abs(new - old)
            if diff > err_max:
                err_max = diff

        # ---- interior i = 1..Nr-2 ----
        for i in range(1, Nr-1):
            ipi = i + 1
            imi = i - 1
            for j in range(Nv-1):
                if tip[i, j]:
                    continue

                old = phi[0, i, j]
                ap = Ap[i, j]
                am = Am[i, j]
                bp = Bp[i, j]
                bm = Bm[i, j]
                cc = C[i, j]

                vip = phi[0, ipi, j]
                vim = phi[0, imi, j]

                vjp1 = phi[0, i, j+1]
                vip_jp1 = phi[0, ipi, j+1]
                vim_jp1 = phi[0, imi, j+1]

                if j == 0:
                    vjm1 = 0.0
                    vip_jm1 = 0.0
                    vim_jm1 = 0.0
                else:
                    vjm1 = phi[0, i, j-1]
                    vip_jm1 = phi[0, ipi, j-1]
                    vim_jm1 = phi[0, imi, j-1]

                temp = (
                    ap*vip + am*vim
                    + bp*vjp1 + bm*vjm1
                    + cc*(vip_jp1 - vip_jm1 - vim_jp1 + vim_jm1)
                )
                new_gs = temp * inv_den[i, j]
                corr   = new_gs - old
                new    = old + omega * corr

                phi[0, i, j] = new
                diff = abs(new - old)
               
                if diff > err_max:
                    err_max = diff

        if it % 500 == 0 and it != 0:
            print("Iter, Res =", "\t", it, "\t", err_max)

        if err_max < ep:
            return err_max

    return err_max

@njit(cache=True, nogil=True)
def refine_grid(phi, tip, r, dr,
                Nr, Nv, dr0, deta,
                a, c, etat, rad2, bias):
    """
    Refine the grid by doubling the resolution in both r and eta.

    Notes
    -----
    - The coarse solution is assumed to be stored in `phi[1, :Nr, :Nv]`.
    - The tip geometry and Dirichlet boundary conditions are rebuilt on
      the refined grid using `build_grid_and_tip()`.
    - Interior vacuum nodes on the refined grid are initialized by
      interpolation from the coarse solution.
    - Tip and boundary nodes are not interpolated here, because their
      values are already set when rebuilding the geometry.

    Returns
    -------
    Nr, Nv, dr0, deta
        Updated grid sizes and spacings for the refined grid.
    """

    Nr_coarse = Nr
    Nv_coarse = Nv

    # --- save coarse solution (only needed region) ---
    phi[1, :Nr_coarse, :Nv_coarse] = phi[0, :Nr_coarse, :Nv_coarse]

    # --- refine sizes ---
    Nr = Nr * 2
    Nv = Nv * 2
    dr0 = dr0 / 2.0
    deta = etat / float(Nv)

    # --- rebuild geometry and BC on the refined grid ---
    build_grid_and_tip(phi, tip, r, dr,
                       Nr, Nv, dr0,
                       a, c, etat, rad2, bias)

    # --- interpolate interior vacuum from coarse phi[1] ---
    for i in range(Nr):
        ic  = i // 2
        icp = ic + 1
        if icp >= Nr_coarse:
            icp = Nr_coarse - 1

        is_mid_r = (i % 2) == 1

        for j in range(Nv-1):
            if tip[i, j]:
                continue  # tip/boundary value already assigned

            J = j + 1  # 1-based eta index
            if J == 1:
                # between surface (0.0) and coarse J=1 (jc0=0)
                jc0 = 0
                if not is_mid_r:
                    vcoarse = phi[1, ic, jc0]
                else:
                    vcoarse = 0.5 * (phi[1, ic, jc0] + phi[1, icp, jc0])
                temp = 0.5 * (vcoarse + 0.0)

            elif (J % 2) == 0:
                # even J -> inherit the corresponding coarse eta point
                Jc = J // 2
                jc0 = Jc - 1
                if not is_mid_r:
                    temp = phi[1, ic, jc0]
                else:
                    temp = 0.5 * (phi[1, ic, jc0] + phi[1, icp, jc0])

            else:
                # odd J>1 -> average neighboring coarse eta points
                J1 = J // 2
                J2 = (J + 1) // 2
                jc10 = J1 - 1
                jc20 = J2 - 1

                if not is_mid_r:
                    temp = 0.5 * (phi[1, ic, jc10] + phi[1, ic, jc20])
                else:
                    v1 = 0.5 * (phi[1, ic, jc10] + phi[1, icp, jc10])
                    v2 = 0.5 * (phi[1, ic, jc20] + phi[1, icp, jc20])
                    temp = 0.5 * (v1 + v2)

            phi[0, i, j] = temp

    return Nr, Nv, dr0, deta

def solve_potential(params: Params, ws: Workspace, sep_nm: float):
    """
    Solve the vacuum electrostatic potential by staged grid refinement.

    Starting from the coarsest grid, the function builds the tip geometry,
    initializes the potential, performs Gauss–Seidel / SOR relaxation to
    convergence at each refinement step, and then refines the grid until
    the final resolution is reached.

    Parameters
    ----------
    params : Params
        User-specified calculation parameters.
    ws : Workspace
        Preallocated workspace arrays reused during the calculation.
    sep_nm : float
        Tip–sample separation used for the electrostatic calculation.

    Returns
    -------
    Nr, Nv, deta, d
        Final grid sizes, final eta spacing, and the derived parameters.
    """
    sep_nm = float(sep_nm)
    d = make_derived(params, sep_nm)
    # expand to local
    Nrin, Nvin, nstep = params.Nrin, params.Nvin, params.nstep
    ep0, itmax = params.ep0, params.itmax
    rad, rad2, scale = params.rad, params.rad2, params.scale
    angle = params.open_angle
    bias = params.bias
    
    ep_scale = d.ep_scale
    a, c, etat = d.a, d.c, d.etat

    # Initial grid
    Nr = Nrin
    Nv = Nvin
    dr0 = rad * scale
    deta = etat / float(Nv)
    
    # initialize ws.r, ws.dr, ws.tip, ws.phi
    build_grid_and_tip(ws.phi, ws.tip, ws.r, ws.dr,
                       Nr, Nv, dr0,
                       a, c, etat, rad2, bias)
    set_initial_guess(ws.phi, ws.tip, Nr, Nv, deta, etat, bias)
    ep = ep0
    for i_step in range(nstep):
        dz_eta0 = sep_nm/Nv   # eta-layer spacing converted to z spacing at r = 0
        xi, dxi = build_xi_dxi(ws.r[:Nr], a)
        
        print(f"Refinement step {i_step + 1}/{nstep}")
        print(f"sep = {sep_nm:.3f} nm, bias = {bias}, rad = {rad} nm, open angle = {angle} deg")
        print(f"Nr, Nv = {Nr}, {Nv}")
        print(f"dr0 = {dr0}, dz_eta0 = {dz_eta0} nm")        

        Ap, Am, Bp, Bm, Cc, inv_den = precompute_stencil_coeffs_feenstra_vargrid(
            xi, dxi, Nr, Nv, a, c, deta
        )

        omega_used = float(params.omega)
        err = iterate_potential_feenstra_vargrid_precomp(
            ws.phi, ws.tip,
            Ap, Am, Bp, Bm, Cc, inv_den,
            Nr, Nv,
            ep=ep, itmax=itmax,
            omega=omega_used,
        )

        if i_step == nstep-1:
            break

        Nr, Nv, dr0, deta = refine_grid(ws.phi, ws.tip, ws.r, ws.dr,
                    Nr, Nv, dr0, deta,
                    a, c, etat, rad2, bias)
        ep *= ep_scale
    return Nr, Nv, deta, d

def sep_center_from_sep_stm(params: Params) -> float:
    """Center (mean) separation in *dynamic STM*.

    The user specifies `sep_STM_nm` as the static STM separation when oscillation
    is OFF. When the tip oscillates with amplitude A, keeping the *average* tunnel
    current constant shifts the center position away from the surface by

        Δz = (1/(2κ)) * ln(I0(2κA))

    where κ is the tunneling decay constant and I0 is the modified Bessel
    function. We also add `z_offset` (used for spectroscopy offsets).
    """
    z_center_nm = params.sep_STM_nm
    z_center_nm += (1.0 / (2.0 * params.kappa)) * np.log(i0(2.0 * params.kappa * params.ampl))
    z_center_nm += params.z_offset
    return float(z_center_nm)

def sep_for_quasistatic_force_nm(params: Params) -> float:
    """Separation used for quasi-static force calculations.

    In the default workflow, we evaluate the electrostatic force at the center
    position corresponding to the STM setpoint in dynamic STM.
    """
    return sep_center_from_sep_stm(params)

def _fd_weights_fornberg(x0: float, x: np.ndarray, m: int = 1):
    """
    Compute finite-difference weights using the Fornberg (1988) algorithm.
    The returned weights approximate the m-th derivative at x0 from
    function values sampled at the 1D nodes x.

    Parameters
    ----------
    x0 : float
        Location where the derivative is approximated.
    x : ndarray
        Node locations.
    m : int, default=1
        Derivative order.

    Returns
    -------
    ndarray
        Weights `w` such that
        f^(m)(x0) ≈ sum_k w[k] * f(x[k]).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= m:
        raise ValueError("Need at least m+1 points")

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

def compute_force_via_sample_surface_pressure_fd_stencil(
    phi: np.ndarray,
    r0_nm: np.ndarray,
    a: float, c: float,
    deta: float,
    npts: int = 7,
):
    """
    Compute the electrostatic force from the Maxwell pressure on the
    grounded sample surface at eta = 0.

    The surface pressure is evaluated as

        p = 0.5 * eps0 * Ez^2

    and the total force is obtained by integrating over the z = 0 plane,

        Fz = - ∫ p dA,   dA = 2π r dr.

    The normal electric field at eta = 0 is obtained from a one-sided
    finite-difference stencil in eta using `npts` points:

        dV/deta|0 ≈ (1/deta) * sum_k w_k V(u_k),

    where u = eta / deta and u_nodes = [0, 1, 2, ..., npts-1].

    Using z = a (xi + c) eta, the derivative is converted as

        dV/dz|0 = (1 / (a (xi + c))) dV/deta|0.

    Parameters
    ----------
    phi : ndarray
        Electrostatic potential array of shape (Nr, Nv).
    r0_nm : ndarray
        Radial grid array on the sample surface, in nm.
    a, c : float
        Geometric parameters of the generalized prolate spheroidal coordinates.
    deta : float
        Eta grid spacing.
    npts : int, default=7
        Number of stencil points used for the one-sided derivative.

    Returns
    -------
    p : ndarray
        Electrostatic pressure on the sample surface (Pa).
    sigma : ndarray
        Surface charge density inferred from Ez (C/m^2).
    Fi : ndarray
        Node-resolved force contributions (N), returned with negative sign
        for attractive force toward the sample.
    F_total : float
        Total electrostatic force (N), returned with negative sign for
        attractive force toward the sample.
    """

    nm_to_m = 1e-9
    phi = np.asarray(phi, dtype=float)
    r0_nm = np.asarray(r0_nm, dtype=float)
    
    Nr = r0_nm.size

    Nr_phi, Nv = phi.shape
    if Nr_phi < Nr:
        raise ValueError("phi has fewer r-columns than Nr")
    if deta <= 0:
        raise ValueError("deta must be > 0")
    npts = int(npts)
    if npts < 3:
        raise ValueError("npts must be >= 3")

    # Node positions in u = eta/deta: [0,1,2,...]
    u_nodes = np.arange(npts, dtype=float)

    # Weights for d/du at u0=0
    w_du = _fd_weights_fornberg(0.0, u_nodes, m=1)

    # Availability check: if eta=0 is not included in phi, we need (npts-1) interior points.
    if Nv < (npts - 1):
        raise ValueError(f"Nv too small: need Nv >= {npts-1}")

    # Compute pressure (r-dependence)
    p = np.zeros(Nr, dtype=float)
    sigma = np.zeros(Nr, dtype=float)
    
    for i in range(Nr):
        # Assemble V at u_nodes
        V = np.empty(npts, dtype=float)
        V[0] = 0.0  # grounded boundary at eta=0

        # phi[i,0] corresponds to eta = 1*deta
        for k in range(1, npts):
            V[k] = phi[i, k-1]         # eta = k*deta

        dV_du_0 = float(np.dot(w_du, V))   # dV/du at u=0
        dV_deta0 = dV_du_0 / deta       # dV/deta at eta=0

        xi = np.sqrt(1.0 + (r0_nm[i] / a) ** 2)
        dz_etadeta_nm = a * (xi + c)   # nm  (exact at eta=0)
        dVdz_eta0 = dV_deta0 / dz_etadeta_nm              # V/nm
        
        Ez = -dVdz_eta0 / nm_to_m                     # V/m
        sigma[i] = EPS0 * Ez                      # C/m^2
        p[i] = 0.5 * EPS0 * Ez * Ez               # Pa

    # Integrate the pressure over the surface with the midpoint rule.
    # The force sign is applied at return.    
    r_m = r0_nm * nm_to_m
    dr_seg = np.diff(r_m)
    r_mid = 0.5 * (r_m[:-1] + r_m[1:])
    p_mid = 0.5 * (p[:-1] + p[1:])
    seg = 2.0 * pi * p_mid * r_mid * dr_seg      # N, segment contributions
    F_total = float(np.sum(seg))
    #add missing central disk contribution
    r0 = r_m[0]
    F0 = pi * p[0] * (r0 * r0)
    F_total += F0

    # Keep Fi shape (Nr,) for compatibility: distribute segment forces to nodes
    Fi = np.zeros(Nr, dtype=float)
    Fi[:-1] += 0.5 * seg
    Fi[1:]  += 0.5 * seg
    Fi[0] += F0

    return p, sigma, -Fi, -F_total

def compute_force_via_tip_surface_pressure_fd_stencil(
    phi: np.ndarray,      # (Nr, Nv) potential on the computational grid
    r0_nm: np.ndarray,    # (Nr,)   radial nodes on eta=0 plane (nm)
    a: float, c: float,
    etat: float,
    deta: float,
    bias: float,
    tip_mask: np.ndarray, # (Nr, Nv) True inside metal (surface included)
    npts: int = 7,        # 5,7,9 recommended (odd preferred)
    debug: bool = False,
):
    """
    Tip force from Maxwell pressure on the tip surface using a one-sided FD stencil.

    Assumption (METALTIP-2D v1):
      - Tip surface is exactly eta=etat, i.e. j_surf = Nv-1 for all i.
      - tip_mask[i, Nv-1] is True (metal surface).
      - The npts-1 points just outside the surface along -eta direction are vacuum:
            j = Nv-2, Nv-3, ..., Nv-npts  must be False.

    Method:
      - Estimate dV/deta at the surface using Fornberg FD weights for nodes u=[0,-1,-2,...]
        with Dirichlet at u=0: V=bias.
      - Use the inverse Jacobian to convert (dV/dxi, dV/deta) to (dV/dr, dV/dz).
      - On the equipotential tip surface, dV/dxi = 0 is imposed.
      - Since the gradient is normal to the equipotential tip surface, the normal field
        magnitude is |En| = |grad V| = sqrt((dV/dr)^2 + (dV/dz)^2).
      - Pressure p = (eps0/2) * En^2, integrate F = -2π ∫ p(r) r dr.
    """

    nm_to_m = 1e-9
    phi = np.asarray(phi, dtype=float)
    r0_nm = np.asarray(r0_nm, dtype=float)
    tip_mask = np.asarray(tip_mask, dtype=bool)

    Nr, Nv = phi.shape
    npts = int(npts)

    if npts < 3:
        raise ValueError("npts must be >= 3")
    if Nv < npts:
        raise ValueError(f"Nv too small for npts={npts} (need Nv >= {npts})")
    if tip_mask.shape != (Nr, Nv):
        raise ValueError("tip_mask shape must match phi shape (Nr, Nv).")

    # FD weights for d/du at u=0 with nodes u=[0,-1,-2,...]
    u_nodes = -np.arange(npts, dtype=float)  # [0, -1, -2, ...]
    w_du = _fd_weights_fornberg(0.0, u_nodes, m=1)

    # Determine which columns are usable (have required vacuum points)
    used = np.ones(Nr, dtype=bool)
    Veta = np.full(Nr, np.nan, dtype=float)  # dV/deta at surface (eta=etat)

    j_surf = Nv - 1
    j0 = Nv - npts      # inclusive start for vacuum slice
    j1 = Nv - 1         # exclusive end (up to Nv-2)

    n_drop = 0
    for i in range(Nr):
        # surface must be metal
        if not tip_mask[i, j_surf]:
            used[i] = False
            n_drop += 1
            continue

        # required contiguous vacuum points: Nv-2 .. Nv-npts
        if np.any(tip_mask[i, j0:j1]):
            used[i] = False
            n_drop += 1
            continue

        # Assemble V at nodes u=[0,-1,-2,...]
        V = np.empty(npts, dtype=float)
        V[0] = bias
        V[1:] = phi[i, Nv-2 : Nv-npts-1 : -1]

        dV_du_0 = float(np.dot(w_du, V))
        Veta[i] = dV_du_0 / float(deta)

    idx_used = np.where(used)[0]
    if idx_used.size < 4:
        raise RuntimeError("Too few usable columns for force integration (tip surface derivative).")

    # xi label is constant along an eta-line, so it can be reconstructed from r on eta=0
    xi = np.sqrt(1.0 + (r0_nm / float(a)) ** 2)

    Xi_used = xi[idx_used]
    Veta_used = Veta[idx_used]
    Eta = float(etat)

    # ------------------------------------------------------------------
    # Inverse-Jacobian route:
    #   [dV/dr, dV/dz]^T = (J^{-1})^T [dV/dxi, dV/deta]^T
    #
    # Here J = [[r_xi, r_eta],
    #           [z_xi, z_eta]]
    #
    # On the tip surface (equipotential), dV/dxi = 0.
    # ------------------------------------------------------------------

    # Jacobian elements on eta = etat
    sqrt_1mEta2 = np.sqrt(1.0 - Eta * Eta)
    sqrt_Xi2m1 = np.sqrt(Xi_used * Xi_used - 1.0)

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
    Vxi_used = np.zeros_like(Veta_used)

    # Transform to physical-space derivatives [V/nm]
    V_r_used = xi_r * Vxi_used + eta_r * Veta_used
    V_z_used = xi_z * Vxi_used + eta_z * Veta_used

    # Since grad(V) is normal to the equipotential tip surface, |En| = |grad V|
    dVdn_V_per_nm_used = np.sqrt(V_r_used * V_r_used + V_z_used * V_z_used)
    En_used = -dVdn_V_per_nm_used / nm_to_m   # V/m
    p_used = 0.5 * EPS0 * En_used * En_used   # Pa
    sigma_used = EPS0 * En_used               # C/m^2

    # radius on tip surface curve r(eta=etat)
    r_tip_nm_used = a * np.sqrt((Xi_used * Xi_used - 1.0) * (1.0 - Eta * Eta))
    r_tip_m_used = r_tip_nm_used * nm_to_m

    # integrate along r (midpoint rule)
    dr_seg = np.diff(r_tip_m_used)
    r_mid = 0.5 * (r_tip_m_used[:-1] + r_tip_m_used[1:])
    p_mid = 0.5 * (p_used[:-1] + p_used[1:])
    seg = 2.0 * pi * p_mid * r_mid * dr_seg
    F_tip = float(np.sum(seg))

    # missing central disk (0 -> r_tip_m_used[0])
    r0 = r_tip_m_used[0]
    F0 = pi * p_used[0] * (r0 * r0)
    F_tip += F0

    # scatter back to full-length arrays (Nr)
    p = np.zeros(Nr, dtype=float)
    sigma = np.zeros(Nr, dtype=float)
    Fi = np.zeros(Nr, dtype=float)

    p[idx_used] = p_used
    sigma[idx_used] = sigma_used

    Fi_used = np.zeros(idx_used.size, dtype=float)
    Fi_used[:-1] += 0.5 * seg
    Fi_used[1:]  += 0.5 * seg
    Fi_used[0] += F0
    Fi[idx_used] = Fi_used

    if debug:
        tailN = min(10, seg.size)
        tail = float(np.sum(seg[-tailN:])) if tailN > 0 else 0.0
        frac = tail / F_tip if F_tip != 0.0 else np.nan
        print(f"[tip-pressure] Nr={Nr} Nv={Nv} used={idx_used.size}/{Nr} dropped={n_drop}  npts={npts}")
        print(f"[tip-pressure] F_tip={F_tip:.6e} N  p_max={p_used.max():.3e} Pa  r_max={r_tip_m_used[-1]/nm_to_m:.3f} nm")
        print(f"[tip-pressure] tail(last {tailN})={tail:.3e} N  frac={frac:.3e}")

    return p, sigma, -Fi, -F_tip

def calc_force_single_condition(
    params: Params,
    sep_nm: Optional[float] = None,
    bias: Optional[float] = None,
    ws: Optional[Workspace] = None,
):
    """Solve potential and return electrostatic force (F_tip, F_sample) [N].

    Parameters
    ----------
    sep_nm:
        *Effective* tip–sample separation (nm) to pass to the potential solver.
        If omitted, we use `sep_for_quasistatic_force_nm(params)`.
    bias:
        Effective bias voltage (V) used for the Dirichlet boundary on the tip.
        If omitted, `params.bias` is used.
    ws:
        If provided, reuse the preallocated workspace to avoid repeated large
        allocations during sweeps (bias/sep/Chebyshev sampling).
    """
    sep_nm_use = sep_for_quasistatic_force_nm(params) if sep_nm is None else float(sep_nm)
    bias_use = float(params.bias) if bias is None else float(bias)

    # Create a local snapshot only when needed (no mutation).
    p = replace(params, bias=bias_use) if bias is not None else params
    
    if ws is None:
        ws = alloc_workspace(p)
    else:
        # Safety: reallocate if the provided workspace is too small.
        Nr_max = p.Nrin * (2 ** (p.nstep - 1))
        Nv_max = p.Nvin * (2 ** (p.nstep - 1))
        if (
            ws.r.shape[0] < Nr_max
            or ws.dr.shape[0] < Nr_max
            or ws.tip.shape[0] < Nr_max
            or ws.tip.shape[1] < Nv_max
            or ws.phi.shape[1] < Nr_max
            or ws.phi.shape[2] < Nv_max
        ):
            ws = alloc_workspace(p)

    Nr, Nv, deta, d = solve_potential(p, ws, sep_nm_use)
    
    # force at sample surface
    _, _, Fi_sample, F_sample = compute_force_via_sample_surface_pressure_fd_stencil(
        phi=ws.phi[0, :Nr, :Nv],
        r0_nm=ws.r[:Nr],
        a=d.a, c=d.c, deta=deta
    )
    
    # force at tip surface
    _, _, Fi_tip, F_tip = compute_force_via_tip_surface_pressure_fd_stencil(
        phi=ws.phi[0, :Nr, :Nv],
        r0_nm=ws.r[:Nr],
        a=d.a, c=d.c, etat=d.etat, deta=deta,
        bias=bias_use, tip_mask=ws.tip[:Nr, :Nv]
    )
    
    return F_tip, F_sample

def delta_f_from_force(
    params: Params,
    N: int = 6,
    ws: Optional[Workspace] = None,
    sep_center_nm: Optional[float] = None,
    bias: Optional[float] = None,
):
    """
    Convert force into the FM-AFM frequency shift Δf [Hz] using
    Gauss-Chebyshev quadrature over the oscillation cycle.

    Workspace reuse:
        If ws is provided, it is reused across all Chebyshev nodes to avoid
        repeated large allocations. If ws is None, one workspace is allocated
        and reused internally.

    Parameters
    ----------
    N : int, default=6
        Number of Gauss-Chebyshev sampling points.
    sep_center_nm : float, optional
        Center position of the tip oscillation in nm. If omitted, the value
        derived from the static STM setpoint is used.

    Returns
    -------
    df_tip, df_sample : float
        Frequency shifts derived from the tip-side and sample-side forces, in Hz.
    """
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    
    A_nm = float(params.ampl)
    A_m = A_nm * 1e-9
    if A_m <= 0.0:
        raise ValueError("params.ampl must be > 0 for Δf calculations")
    k = float(params.k)
    f0 = float(params.freq)

    bias_use = float(params.bias) if bias is None else float(bias)
    sep_center_nm_use = (
        sep_center_from_sep_stm(params) if sep_center_nm is None else float(sep_center_nm)
    )
    z0_m = sep_center_nm_use * 1e-9

    # Chebyshev nodes u in [-1, 1]
    i = np.arange(1, N + 1)
    u = np.cos((2 * i - 1) * pi / (2 * N))
    
    # Chebyshev node positions along the oscillation trajectory
    z_m = z0_m + A_m * u
    z_nm = z_m * 1e9

    # Allocate once (if needed) and reuse for all nodes
    if ws is None:
        ws = alloc_workspace(params)

    F_tip = np.zeros(N, dtype=float)
    F_sample = np.zeros(N, dtype=float)

    for idx, zi in enumerate(z_nm):
        F_tip[idx], F_sample[idx] = calc_force_single_condition(
            params,
            sep_nm=float(zi),
            bias=bias_use,
            ws=ws,
        )

    df_tip = -(f0 / (k * A_m * N)) * np.sum(F_tip * u)
    df_sample = -(f0 / (k * A_m * N)) * np.sum(F_sample * u)

    return df_tip, df_sample

def make_fname_header(params: Params, sep_nm: Optional[float] = None):
    """
    Create the output filename and text header for bias-sweep calculations.

    The header records all fields defined in Params so that the complete
    calculation settings can be reconstructed from the output file.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    mode = params.mode
    if mode == "F-V":
        prefix = "F-V_"
        title = "Force vs bias"
        variable_name = "Bias [V]"
        channel_name = "Electrostatic force [N]"
    elif mode == "F-V_approx":
        prefix = "F-V_approx_"
        title = "Approximate force vs bias"
        variable_name = "Bias [V]"
        channel_name = "Electrostatic force [N]"
    elif mode == "df-V":
        prefix = "df-V_"
        title = "Frequency shift vs bias"
        variable_name = "Bias [V]"
        channel_name = "Frequency shift [Hz]"
    elif mode == "df-V_approx":
        prefix = "df-V_approx_"
        title = "Approximate frequency shift vs bias"
        variable_name = "Bias [V]"
        channel_name = "Frequency shift [Hz]"
    else:
        raise ValueError(f"Unknown params.mode: {mode}")

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
        f"output mode: {mode}",
        "",
        "[DERIVED / EFFECTIVE VALUES]",
        f"sep_center from dynamic STM (nm): {sep_center:.12g}",
        #f"separation used in this calculation (nm): {sep0:.12g}",
        "",
        "[INPUT PARAMETERS]",
    ]

    # Save every field defined in Params. This avoids missing newly added
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
    """Save x-y data to a text file with a header."""
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)

    with open(fname, "w", encoding="utf-8") as f:
        f.write(header)
        for xi, yi in zip(x, y):
            f.write(f"{xi:.12g}\t{yi:.12e}\n")

def plot_xy(x, y, xlabel="bias [V]", ylabel=""):
    """Plot x-y data."""
    import matplotlib.pyplot as plt

    plt.plot(x, y, ".")
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def calc_force_bias_curve_approx(params: Params, npoints: int):
    """
    Approximate the force-bias curve by assuming a parabolic dependence
    centered at the CPD.

    The force is evaluated explicitly at one nonzero effective bias, and
    the full F-V curve is reconstructed as F = a (V - CPD)^2.
    """
    npoints = int(npoints)
    if npoints < 1:
        raise ValueError("npoints must be >= 1")

    V0_candidates = [params.bias_start, params.bias_end]
    Vapp0 = max(V0_candidates, key=lambda v: abs(v - params.CPD))
    Veff0 = Vapp0 - params.CPD
    if abs(Veff0) < 1e-15:
        raise ValueError("bias_start and bias_end are both equal to CPD")

    bias_arr = np.linspace(params.bias_start, params.bias_end, npoints)

    # Evaluate the force at one nonzero effective bias.
    sep0_nm = sep_for_quasistatic_force_nm(params)
    F_tip0, F_sample0 = calc_force_single_condition(params, sep_nm=sep0_nm, bias=Veff0)

    a_tip = F_tip0 / (Veff0 ** 2)
    a_sample = F_sample0 / (Veff0 ** 2)

    F_tip = a_tip * (bias_arr - params.CPD) ** 2
    F_sample = a_sample * (bias_arr - params.CPD) ** 2

    F = F_tip if params.force_mode == "tip" else F_sample
    return bias_arr, F

def calc_force_bias_curve(params: Params):
    """
    Compute the force-bias curve over the applied-bias range.

    The electrostatic potential is solved at each applied bias `V`, using
    the effective bias `V - CPD` for the tip Dirichlet boundary condition.
    A single Workspace is reused across the sweep to avoid repeated large
    allocations.
    """
    npoints = int(params.npoints)
    if npoints < 1:
        raise ValueError("params.npoints must be >= 1")

    bias_app = np.linspace(params.bias_start, params.bias_end, npoints)
    bias_eff = bias_app - params.CPD
    F_tip = np.empty(npoints)
    F_sample = np.empty(npoints)

    ws = alloc_workspace(params)
    sep0_nm = sep_for_quasistatic_force_nm(params)

    iterable = tqdm(bias_eff, total=npoints, desc="bias", leave=False)
    for i, bias in enumerate(iterable):
        F_tip[i], F_sample[i] = calc_force_single_condition(
            params,
            sep_nm=sep0_nm,
            bias=float(bias),
            ws=ws,
        )

    F = F_tip if params.force_mode == "tip" else F_sample
    return bias_app, F

def calc_delta_f_V_curve_approx(params: Params, npoints: int):
    """
    Approximate the Δf-V curve by assuming a parabolic dependence
    centered at the CPD.

    The frequency shift is evaluated explicitly at one nonzero effective
    bias, and the full Δf-V curve is reconstructed as
    Δf = a (V - CPD)^2.
    """
    npoints = int(npoints)
    if npoints < 1:
        raise ValueError("npoints must be >= 1")

    V0_candidates = [params.bias_start, params.bias_end]
    Vapp0 = max(V0_candidates, key=lambda v: abs(v - params.CPD))
    Veff0 = Vapp0 - params.CPD
    if abs(Veff0) < 1e-15:
        raise ValueError("bias_start and bias_end are both equal to CPD")

    bias_arr = np.linspace(params.bias_start, params.bias_end, npoints)

    # Calculate Δf at one nonzero effective bias.
    df_tip0, df_sample0 = delta_f_from_force(params, N=6, bias=float(Veff0))

    a_tip = df_tip0 / (Veff0 ** 2)
    a_sample = df_sample0 / (Veff0 ** 2)

    df_tip = a_tip * (bias_arr - params.CPD) ** 2
    df_sample = a_sample * (bias_arr - params.CPD) ** 2

    df = df_tip if params.force_mode == "tip" else df_sample
    return bias_arr, df

def calc_delta_f_V_curve(params: Params):
    """
    Compute the Δf-V curve over the applied-bias range.

    At each applied bias V, the electrostatic calculation uses the effective
    bias V - CPD for the tip Dirichlet boundary condition. A single Workspace
    is reused across the bias sweep, and the same Workspace is also reused
    within each Δf evaluation over the Gauss-Chebyshev nodes.
    """
    npoints = int(params.npoints)
    if npoints < 1:
        raise ValueError("params.npoints must be >= 1")

    bias_app = np.linspace(params.bias_start, params.bias_end, npoints)
    bias_eff = bias_app - params.CPD
    df_tip = np.empty(npoints)
    df_sample = np.empty(npoints)

    ws = alloc_workspace(params)
    sep_center_nm = sep_center_from_sep_stm(params)

    iterable = tqdm(bias_eff, total=npoints, desc="bias", leave=False)
    for i, bias in enumerate(iterable):
        df_tip[i], df_sample[i] = delta_f_from_force(
            params,
            N=6,
            ws=ws,
            sep_center_nm=sep_center_nm,
            bias=float(bias),
        )

    df = df_tip if params.force_mode == "tip" else df_sample
    return bias_app, df

def main():
    params = Params()

    if params.mode == "F":
        bias_eff = params.bias - params.CPD
        F_tip, F_sample = calc_force_single_condition(params, bias=bias_eff)
        print("\n==============================")
        print("F_tip =", F_tip)
        print("F_sample =", F_sample)
        print("==============================")

    elif params.mode == "F-V_approx":
        bias, F = calc_force_bias_curve_approx(params, 128)
        if params.save_data:
            fname, header = make_fname_header(params)
            save_xy(fname, header, bias, F)
        if params.plot_data:
            plot_xy(bias, F, ylabel="Force (N)")

    elif params.mode == "F-V":
        bias, F = calc_force_bias_curve(params)
        if params.save_data:
            fname, header = make_fname_header(params)
            save_xy(fname, header, bias, F)
        if params.plot_data:
            plot_xy(bias, F, ylabel="Force (N)")

    elif params.mode == "df-V_approx":
        bias, df = calc_delta_f_V_curve_approx(params, 128)
        if params.save_data:
            fname, header = make_fname_header(params)
            save_xy(fname, header, bias, df + params.df_offset)
        if params.plot_data:
            plot_xy(bias, df + params.df_offset, ylabel="Frequency shift (Hz)")

    elif params.mode == "df-V":
        bias, df = calc_delta_f_V_curve(params)
        if params.save_data:
            fname, header = make_fname_header(params)
            save_xy(fname, header, bias, df + params.df_offset)
        if params.plot_data:
            plot_xy(bias, df + params.df_offset, ylabel="Frequency shift (Hz)")

    else:
        raise ValueError(f"Unknown computation mode: {params.mode}")


if __name__ == "__main__":
    main()

