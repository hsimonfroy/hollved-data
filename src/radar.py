"""
Compute cosmological radar and HUD lookup table.

Outputs hollved-data/aux/radar.json with:
  - ring:   [{name, radius}]  radii in comoving Mpc for BAO, Hubble, Event, Particle horizons
  - sphere: [{name, radius}]  comoving radius of the CMB sphere (surface of last scattering)
  - hud:    {chi_Mpc, z, lookback_Myr}  128-point lookup table for spaceship HUD interpolation

Cosmology: Planck 2018 VI, Table 2 best fit (same as utils.py).
Uses jax_cosmo for all expansion-rate integrals (via background.dchioverda),
mirroring the a2chi pattern from utils.py.
BAO sound horizon uses a numpy integral with radiation (T_CMB-derived Omega_gamma),
since jax_cosmo's Planck18 object excludes the radiation density at z~1000.
"""
import json
import sys
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax_cosmo import background
from jax_cosmo.scipy.ode import odeint

sys.path.insert(0, str(Path(__file__).parent))
from utils import Planck18, a2chi, ROOT

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
Mpc_in_km = 3.085677581e19           # km per Mpc (IAU 2012)
Myr_in_s  = 1e6 * 365.2425 * 24 * 3600  # Gregorian Myr in seconds
c_km_s    = 2.99792458e5             # exact speed of light in km/s

# ---------------------------------------------------------------------------
# Radiation constants (not stored in jax_cosmo's Cosmology object)
# ---------------------------------------------------------------------------
T_CMB = 2.7255   # K (Fixsen 2009)
N_eff = 3.046

# ---------------------------------------------------------------------------
# a2lookback_Myr — mirrors a2chi from utils.py
# ---------------------------------------------------------------------------
def a2lookback_Myr(cosmo, a, log10_amin=-6, steps=256):
    """
    Lookback time in Myr from scale factor a to today (a=1).

    Derivation:
      dchi/da = c / (a^2 H(a))  [Mpc/h in jax_cosmo, where H0_jax = 100 km/s/(Mpc/h)]
      → E(a) = (c_km_s/100) / (a^2 x dchioverda)
      → dt/dlna = (1/H0) / E(a) = H0_inv_Myr x a^2 x dchioverda x (100/c_km_s)  [Myr]
      Verify at a=1, E=1: dchioverda(1) = c_km_s/100  → dtoverdlna = H0_inv_Myr ✓
    """
    if "background.lookback_time" not in cosmo._workspace:
        h = float(cosmo.h)
        H0_inv_Myr = Mpc_in_km / (100.0 * h * Myr_in_s)
        atab = jnp.logspace(log10_amin, 0.0, steps)
        def dtoverdlna(y, x):
            xa = jnp.exp(x)
            return H0_inv_Myr * xa**2 * background.dchioverda(cosmo, xa) * (100.0 / c_km_s)
        ttab = odeint(dtoverdlna, 0.0, jnp.log(atab))
        ttab = ttab[-1] - ttab   # lookback from a to 1; 0 at a=1
        cosmo._workspace["background.lookback_time"] = {"a": atab, "t": ttab}
    cache = cosmo._workspace["background.lookback_time"]
    return float(jnp.interp(jnp.asarray(a, dtype=float), cache["a"], cache["t"]))

# ---------------------------------------------------------------------------
# Radar computations
# ---------------------------------------------------------------------------
def compute_particle_horizon(cosmo):
    """Comoving distance to a=1e-6 (≈ particle horizon) in Mpc."""
    return float(a2chi(cosmo, 1e-6, log10_amin=-6, steps=256)) / float(cosmo.h)

def compute_event_horizon(cosmo):
    """Comoving future light-cone limit (event horizon) in Mpc.

    Integrates dchi/dlna from a=1 forward to a=10000 (effectively ∞ for ΛCDM),
    using the same dchioverdlna integrand as a2chi.  Extending beyond a=100
    recovers the ~53 Mpc tail that truncation at a=100 would miss.
    """
    atab_future = jnp.logspace(0, 4, 512)   # a: 1 → 10000
    def dchioverdlna(y, x):
        xa = jnp.exp(x)
        return background.dchioverda(cosmo, xa) * xa
    eh_raw = odeint(dchioverdlna, 0.0, jnp.log(atab_future))
    return float(eh_raw[-1]) / float(cosmo.h)   # Mpc

def compute_hubble_radius(cosmo):
    """c/H0 in Mpc."""
    return c_km_s / (100.0 * float(cosmo.h))

def compute_cmb_sphere(cosmo, z_cmb=1089.0):
    """Comoving distance to the CMB surface of last scattering in Mpc."""
    a_cmb = 1.0 / (1.0 + z_cmb)
    return float(a2chi(cosmo, a_cmb, log10_amin=-6, steps=256)) / float(cosmo.h)

def compute_bao_sound_horizon(cosmo):
    """
    Comoving sound horizon at the baryon drag epoch, in Mpc.

    Uses Eisenstein & Hu (1998) eq. 4 for z_drag.
    Integrates c_s(z)/H(z) dz from z_drag to 1e7.

    jax_cosmo's Planck18 object excludes the radiation density, so H(z) at
    z~1000 (the drag epoch) is severely underestimated if derived from
    dchioverda.  Instead, H(z) is computed analytically from the full
    Friedmann equation including radiation (Omega_r from T_CMB).
    """
    h       = float(cosmo.h)
    Omega_b = float(cosmo.Omega_b)
    Omega_m = float(cosmo.Omega_c + cosmo.Omega_b)

    Ogamma_h2 = 2.47282e-5 * (T_CMB / 2.725)**4
    Or_h2     = (1 + N_eff * (7./8.) * (4./11.)**(4./3.)) * Ogamma_h2
    Omega_gamma = Ogamma_h2 / h**2
    Omega_r     = Or_h2 / h**2
    Omega_L     = 1.0 - Omega_m - Omega_r   # flat

    H0 = 100.0 * h   # km/s/Mpc

    def E_of_a(a):
        return np.sqrt(Omega_m * a**-3 + Omega_r * a**-4 + Omega_L)

    # Eisenstein & Hu (1998) drag epoch
    Omh2 = Omega_m * h**2
    Obh2 = Omega_b * h**2
    b1 = 0.313 * Omh2**-0.419 * (1.0 + 0.607 * Omh2**0.674)
    b2 = 0.238 * Omh2**0.223
    z_drag = 1291.0 * Omh2**0.251 / (1.0 + 0.659 * Omh2**0.828) * (1.0 + b1 * Obh2**b2)

    # Dense z grid: log-spaced from z_drag to 1e7
    z_arr = np.concatenate([
        np.logspace(np.log10(z_drag), 4, 200_000),
        np.logspace(4, 7, 50_000),
    ])
    a_arr = 1.0 / (1.0 + z_arr)

    H_arr = H0 * E_of_a(a_arr)   # km/s/Mpc (full Friedmann with radiation)

    # Photon-baryon sound speed
    R_b = 0.75 * Omega_b / (Omega_gamma * (1.0 + z_arr))
    cs  = c_km_s / np.sqrt(3.0 * (1.0 + R_b))   # km/s

    return float(np.trapz(cs / H_arr, z_arr))   # Mpc

# ---------------------------------------------------------------------------
# HUD lookup table
# ---------------------------------------------------------------------------
def compute_hud(cosmo, particle_horizon, n=128):
    """
    n-point lookup table mapping comoving distance chi (Mpc) to redshift z
    and lookback time (Myr), for spaceship HUD interpolation.

    Extracts the cached a-table built by a2chi and a2lookback_Myr (both use
    log10_amin=-6, steps=256), flips to ascending chi order, then samples
    at n evenly-spaced chi values from 0 to particle_horizon.
    """
    chi_cache = cosmo._workspace["background.radial_comoving_distance"]
    lb_cache  = cosmo._workspace["background.lookback_time"]

    atab  = np.array(chi_cache["a"])                           # 256 pts, 1e-6 → 1
    chi_h = np.array(chi_cache["chi"]) / float(cosmo.h)       # Mpc; large at a_min, 0 at a=1
    lb    = np.array(lb_cache["t"])                            # Myr; large at a_min, 0 at a=1

    # Flip to ascending order (chi: 0 → particle_horizon)
    chi_flip = chi_h[::-1]
    lb_flip  = lb[::-1]
    a_flip   = atab[::-1]

    chi_s = np.linspace(0.0, particle_horizon, n)
    z_s   = 1.0 / np.interp(chi_s, chi_flip, a_flip) - 1.0
    lb_s  = np.interp(chi_s, chi_flip, lb_flip)

    return chi_s, z_s, lb_s

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cosmo = Planck18()

    print("Computing radar...")

    # Particle horizon also warms the a2chi cache with the extended range
    particle_horizon = compute_particle_horizon(cosmo)
    event_horizon    = compute_event_horizon(cosmo)
    hubble_radius    = compute_hubble_radius(cosmo)
    cmb              = compute_cmb_sphere(cosmo)

    print("  Computing BAO sound horizon (slow: ~250k jax_cosmo evaluations)...")
    bao = compute_bao_sound_horizon(cosmo)

    print("  Building HUD lookup table...")
    a2lookback_Myr(cosmo, 1e-6, log10_amin=-6, steps=256)  # warms lookback cache
    chi_s, z_s, lb_s = compute_hud(cosmo, particle_horizon)

    # Print for verification
    print(f"\nBAO sound horizon:  {bao:.2f} Mpc")
    print(f"Hubble radius:      {hubble_radius:.2f} Mpc")
    print(f"Event horizon:      {event_horizon:.2f} Mpc")
    print(f"Particle horizon:   {particle_horizon:.2f} Mpc")
    print(f"CMB sphere:         {cmb:.2f} Mpc")
    print(f"\nHUD table: {len(chi_s)} points, chi 0 → {particle_horizon:.1f} Mpc")
    print(f"  z range:          {z_s[0]:.3f} → {z_s[-1]:.2e}")
    print(f"  lookback range:   {lb_s[0]:.1f} → {lb_s[-1]:.1f} Myr  "
          f"(age of universe ≈ {lb_s[-1]/1e3:.2f} Gyr)")

    data = {
        "ring": [
            {"name": "BAO",              "radius": round(bao, 3)},
            {"name": "Hubble radius",    "radius": round(hubble_radius, 3)},
            {"name": "Event horizon",    "radius": round(event_horizon, 3)},
            {"name": "Particle horizon", "radius": round(particle_horizon, 3)},
        ],
        "sphere": [
            {"name": "CMB", "radius": round(cmb, 3)},
        ],
        "hud": {
            "chi_Mpc":      [round(x, 4) for x in chi_s.tolist()],
            "z":            [round(x, 6) for x in z_s.tolist()],
            "lookback_Myr": [round(x, 3) for x in lb_s.tolist()],
        },
    }

    out_path = ROOT / 'aux' / 'radar.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f"\nWritten: {out_path}")

if __name__ == '__main__':
    main()
