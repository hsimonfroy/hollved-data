import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.stats import gaussian_kde
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, chi2a, a2chi, kde_val, plot_chi_z, kde_chi_z, write_densities

TRACERS = [
    {'id': 'cfa1', 'name': 'CfA1', 'color': '0x26ff00ff', 'file': 'cfa1.txt'},  # neon green
    {'id': 'cfa2', 'name': 'CfA2', 'color': '0xff0088ff', 'file': 'cfa2.txt'},  # neon pink
]

C = 299792.458  # speed of light, km/s
# Note on epoch: RA DEC follow old B1950 convention. 
# The ~50 arcsec offset from J2000 conveiton causes ≲1 Mpc positional error at the survey edge (z ~ 0.05)
# negligible for float16 visualization.

def parse_cfa(filepath):
    """Parse a CfA fixed-width text file (B1950 coords).

    Column layout (0-indexed):
      11-18  RA  HHMMSS.S
      19     Dec sign (' ' or '-')
      20-25  Dec DDMMSS
      30-35  v_h  heliocentric velocity (km/s)
    """
    ras, decs, zs = [], [], []
    with open(filepath) as f:
        for line in f:
            if len(line) < 36:
                continue
            try:
                rs = line[11:19]
                ra = (int(rs[0:2]) + int(rs[2:4]) / 60 + float(rs[4:8]) / 3600) * 15
                sign = -1 if line[19] == '-' else 1
                ds = line[20:26]
                dec = sign * (int(ds[0:2]) + int(ds[2:4]) / 60 + int(ds[4:6]) / 3600)
                vh = float(line[30:36])
            except (ValueError, IndexError):
                continue
            ras.append(ra)
            decs.append(dec)
            zs.append(vh / C)
    return np.array(ras), np.array(decs), np.array(zs)

cosmo = Planck18()
# from matplotlib import pyplot as plt
# plt.figure(figsize=(8, 6), layout='constrained')

densities = []
for t in TRACERS:
    print(f"\n=== CfA {t['name']} ===")
    ra, dec, z = parse_cfa(RAW / t['file'])
    print(f"  Parsed {len(z):,} objects")
    pos = radecz2cart(cosmo, ra, dec, z)

    # Remove objects whose measurements are probably affected by peculiar velocities or systematic errors.
    # Velocity dispersion is estimated to ~340 km/s <=> z ~ 0.001
    # [David&Peebles1983](https://articles.adsabs.harvard.edu/pdf/1983ApJ...267..465D)
    z_min = 0.001
    pos = pos[z > z_min]
    z = z[z > z_min]
    print(f"  Cut z <= {z_min} removed {len(ra) - len(pos)} objects over {len(ra):,}")
    densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
    # write_output(ROOT / 'cfa' / t['id'], pos, t['id'], t['name'], t['color'])

    
    
    # plot_chi_z(cosmo, pos, z, t, n_disc=64, scale=1.)
# plt.legend()
# plt.savefig(ROOT / 'src' / 'cfa_kde1_jac.png', dpi=300)

write_densities(ROOT / 'cfa' / 'manifest.json', densities)

print("\nDone.")


