import json
import numpy as np
from pathlib import Path
from functools import partial
from jax import numpy as jnp
from jax_cosmo import Cosmology, background
from jax_cosmo.scipy.ode import odeint
from scipy.stats import gaussian_kde
from zorder import zorder

# Cosmology (Planck 2018 VI, Table 2 best fit — https://arxiv.org/abs/1807.06209)
Planck18 = partial(Cosmology,
    Omega_c=0.2607,
    Omega_b=0.0490,
    sigma8=0.8102,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    w0=-1.0,
    wa=0.0,)

dist_log10_amin: float = -3.
dist_steps: int = 2048

def a2chi(cosmo, a, log10_amin=dist_log10_amin, steps=dist_steps):
    if "background.radial_comoving_distance" not in cosmo._workspace:
        atab = jnp.logspace(log10_amin, 0.0, steps)
        def dchioverdlna(y, x):
            xa = jnp.exp(x)
            return background.dchioverda(cosmo, xa) * xa
        chitab = odeint(dchioverdlna, 0.0, jnp.log(atab))
        chitab = chitab[-1] - chitab
        cosmo._workspace["background.radial_comoving_distance"] = {"a": atab, "chi": chitab}
    cache = cosmo._workspace["background.radial_comoving_distance"]
    return jnp.clip(jnp.interp(a, cache["a"], cache["chi"]), 0.0)

def chi2a(cosmo, chi, log10_amin=dist_log10_amin, steps=dist_steps):
    # Check if distances have already been computed, force computation otherwise
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        a2chi(cosmo, 1.0, log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.radial_comoving_distance"]
    return jnp.interp(chi, cache["chi"][::-1], cache["a"][::-1]) # NOTE: chi is decreasing with a

def a2DL(cosmo, a):
    return a2chi(cosmo, a) / a

def DL2a(cosmo, DL):
    if "background.radial_comoving_distance" not in cosmo._workspace:
        a2chi(cosmo, 1.0)
    cache = cosmo._workspace["background.radial_comoving_distance"]
    return jnp.interp(DL, (cache["chi"] / cache["a"])[::-1], cache["a"][::-1]) # NOTE: chi/a is decreasing with a

def radecrad2cart(ra, dec, radius):
    ra = jnp.deg2rad(ra)
    dec = jnp.deg2rad(dec)
    x = jnp.cos(dec) * jnp.cos(ra)
    y = jnp.cos(dec) * jnp.sin(ra)
    z = jnp.sin(dec)
    return jnp.moveaxis(radius * jnp.stack((x, y, z)), 0, -1)

def radecz2cart(cosmo, ra, dec, z):
    ra = jnp.array(ra, dtype=float)
    dec = jnp.array(dec, dtype=float)
    z = jnp.array(z, dtype=float)
    radius = a2chi(cosmo, 1 / (1 + z)) / cosmo.h  # Mpc/h → Mpc
    return radecrad2cart(ra, dec, radius)

def write_output(out_dir, pos, tracer_id, tracer_name, color):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = zorder(np.array(pos)).astype('<f2')  # float16 little-endian
    with open(out_dir / 'positions.bin', 'wb') as f:
        f.write(pos.tobytes())
    meta = {'id': tracer_id, 'name': tracer_name, 'count': len(pos), 'color': color}
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, separators=(',', ':'))
    print(f'  {tracer_id}: {len(pos):,} objects → {out_dir}')

def _sig(arr, n):
    """Round array to n significant figures."""
    arr = np.asarray(arr, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = 10.0 ** (n - 1 - np.floor(np.log10(np.abs(np.where(arr == 0, 1.0, arr)))))
    return np.round(arr * scale) / scale

def write_densities(manifest_path, densities):
    """Write KDE density profiles into an existing manifest.json.

    densities: list of (chis, dndchis, zs, dndzs) tuples, one per tracer.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest['densities'] = {
        'chi_Mpc': [_sig(chis, 4).tolist() for chis, _, _, _  in densities],
        'dndchi':  [_sig(d,    4).tolist()  for _, d, _, _      in densities],
        'z':       [_sig(zs,   4).tolist()  for _, _, zs, _     in densities],
        'dndz':    [_sig(d,    4).tolist()  for _, _, _, d       in densities],
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, separators=(',', ':'))
    print(f'\n  Saved densities ({len(densities)} tracers) → {manifest_path}')





def kde_val(x, n_disc=64, scale=1., prob=0.99):
    # kde = gaussian_kde(x)
    kde = gaussian_kde(x, bw_method=scale * len(x)**(-1/5))
    
    x_lo, x_hi = np.quantile(x, [(1 - prob) / 2, (1 + prob) / 2])
    xs = np.linspace(x_lo, x_hi, n_disc)
    # xs = np.linspace(0, 7000, n_disc)
    dndxs = kde(xs) * len(x)
    return xs, dndxs

def kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.):
    chi = np.linalg.norm(pos, axis=-1)  # Mpc

    print(f"  Redshift range: {z.min():.2f} - {z.max():.2f}")
    print(f"  Comoving distance range: {chi.min():.2f} - {chi.max():.2f} Mpc")

    chis, dndchis = kde_val(chi, n_disc=n_disc, scale=scale)

    # Convert chi [Mpc] → a → z using cached chi(a) table
    a = np.array(chi2a(cosmo, chis * cosmo.h))  # chi [Mpc] → chi [Mpc/h] → a
    zs = 1 / a - 1

    # Jacobian: dn/dz = dn/dchi_Mpc * |dchi_Mpc/dz| = dn/dchi * a^2 * dchioverda / h
    jacobian = a**2 * np.array(background.dchioverda(cosmo, a)) / cosmo.h
    dndzs = dndchis * jacobian

    return chis, dndchis, zs, dndzs

# def kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.):
#     chi = np.linalg.norm(pos, axis=-1)

#     print(f"  Redshift range: {z.min():.2f} - {z.max():.2f}")
#     print(f"  Comoving distance range: {chi.min():.2f} - {chi.max():.2f} Mpc")
#     chis, dndchis = kde_val(chi, n_disc=n_disc, scale=scale)
#     zs, dndzs = kde_val(z, n_disc=n_disc, scale=scale)
#     return chis, dndchis, zs, dndzs




import matplotlib.pyplot as plt
def plot_chi_z(cosmo, pos, z, tracer, n_disc=64, scale=1/3):

    chis, dndchis, zs, dndzs = kde_chi_z(cosmo, pos, z, n_disc=n_disc, scale=scale)
    # dndchis[np.array([0,-1])] = 0
    # dndzs[np.array([0,-1])] = 0
    dens_of_chi = dndchis / (4 * np.pi * chis**2)

    plt.subplot(2, 2, 1)
    plt.plot(chis, dndchis, label=tracer['name'], color='#'+tracer['color'][2:])
    plt.xlabel(r'$\chi$ [Mpc]')
    plt.ylabel(r'$\mathrm{d}N/\mathrm{d}\chi$')
    plt.yscale('log')
    # plt.ylim(1e1, 1e4)
    

    plt.subplot(2, 2, 2)
    plt.plot(zs, dndzs, label=tracer['name'], color='#'+tracer['color'][2:])
    plt.xlabel('z')
    plt.ylabel(r'$\mathrm{d}N/\mathrm{d}z$')
    plt.yscale('log')
    # plt.xlim(0, 4)
    # plt.ylim(1e4, 1e8)


    plt.subplot(2, 2, 3)
    plt.plot(chis, dens_of_chi, label=tracer['name'], color='#'+tracer['color'][2:])
    plt.xlabel(r'$\chi$ [Mpc]')
    plt.ylabel(r'comoving density $n(\chi)$')
    plt.yscale('log')
    # plt.ylim(1e-7, 1e-2)

    plt.subplot(2, 2, 4)
    plt.plot(zs, dens_of_chi, label=tracer['name'], color='#'+tracer['color'][2:])
    plt.xlabel('z')
    plt.ylabel(r'comoving density $n(z)$')
    plt.yscale('log')
    # plt.xlim(0, 4)
    # plt.ylim(1e-7, 1e-2)



from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def alternate(a, b, axis=0):
    a, b = np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0)
    assert a.shape == b.shape
    mid = a.shape[0]
    out = np.zeros((2*mid, *a.shape[1:]))
    out[:mid:2] = a[::2]
    out[mid::2] = b[::2]
    out[1:mid:2] = b[1::2]
    out[mid+1::2] = a[1::2]
    return np.moveaxis(out, 0, axis)


c1 = plt.get_cmap('Dark2').colors
c2 = plt.get_cmap('Set2').colors

colors = alternate(c2, c1)
colors[np.array([0,2])] = colors[np.array([2,0])]
SetDark2 = ListedColormap(colors)
DarkSet2 = ListedColormap(alternate(c1, c2))


ROOT = Path(__file__).parent.parent
RAW  = ROOT / 'raw'
