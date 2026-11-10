import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import fitsio
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, plot_chi_z, kde_chi_z, write_densities

# ── Preprocess large spectro files to save only necessary columns ─────────────────
# COLUMNS = ['id', 'ra', 'dec', 'zfinal', 'zerrfinal', 'zconffinal']
# DRS = ['123a', '123b', '4', '5', '6', '7']
# OUT_DIR = ROOT / 'raw'

# TRACERS = [
#     ('main_gals', 'sdss_dr7_main_gals.fits'),
#     ('main_lrgs', 'sdss_dr7_main_lrgs.fits'),
#     ('main_qsos', 'sdss_dr7_main_qsos.fits'),
# ]

# for prefix, out_name in TRACERS:
#     parts = []
#     for dr in DRS:
#         path = RAW / f'{prefix}-DR{dr}.fit'
#         data = fitsio.read(str(path), columns=COLUMNS)
#         parts.append(data)
#         print(f'  {path.name}: {len(data)} rows')
#     combined = np.concatenate(parts)
#     out_path = OUT_DIR / out_name
#     fitsio.write(str(out_path), combined, clobber=True)
#     print(f'Wrote {len(combined)} rows -> {out_path}')


# ── Compute positions and write output ────────────────────────────────────────

TRACERS = [
    {'id': 'mgs', 'name': 'MGS', 'color': '0x24bddcff', 'file': 'sdss_dr7_main_gals.fits'},
    {'id': 'lrg', 'name': 'LRG', 'color': '0xf2df32ff', 'file': 'sdss_dr7_main_lrgs.fits'},
    {'id': 'qso', 'name': 'QSO', 'color': '0xff0011ff', 'file': 'sdss_dr7_main_qsos.fits'},
]

cosmo = Planck18()
# from matplotlib import pyplot as plt
# plt.figure(figsize=(8, 6), layout='constrained')

# 30 km/sec rms for Main Galaxy Sample => dz~0.0001, but take conservative margin
z_min = 0.001

densities = []
for t in TRACERS:
    print(f"\n=== SDSS2 {t['name']} ===")
    data = fitsio.read(str(RAW / t['file']), columns=['ra', 'dec', 'zfinal', 'zconffinal'])
    ra  = data['ra'].astype(np.float64)
    dec = data['dec'].astype(np.float64)
    z   = data['zfinal'].astype(np.float64)
    pos = radecz2cart(cosmo, ra, dec, z)
    mask = z > z_min
    pos = pos[mask]
    z = z[mask]
    print(f"  {len(data):,} total, cut z <= {z_min}: {len(data) - mask.sum()} removed, {len(pos):,} kept")
    densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
    write_output(ROOT / 'sdss2' / t['id'], pos, t['id'], t['name'], t['color'])


    # plot_chi_z(cosmo, pos, z, t, n_disc=64, scale=1.)
# plt.legend()
# plt.savefig(ROOT / 'src' / 'sdss2_kde.png', dpi=300)

write_densities(ROOT / 'sdss2' / 'manifest.json', densities)

print("\nDone.")
