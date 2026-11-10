import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fitsio
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, kde_chi_z, write_densities

TRACERS = [
    {'id': 'bright', 'name': 'Bright QSO', 'color': '0xf99459ff', 'mag_max': 20.0},
    {'id': 'faint',  'name': 'Faint QSO',  'color': '0xff0011ff', 'mag_min': 20.0},
]

cosmo = Planck18()

print("=== Quaia QSO ===")
data = fitsio.read(str(RAW / 'Quaia.fits'), columns=['ra', 'dec', 'redshift_quaia', 'phot_g_mean_mag'])
print(f"  Loaded {len(data):,} objects")

densities = []
for t in TRACERS:
    if 'mag_max' in t:
        mask = data['phot_g_mean_mag'] < t['mag_max']
    else:
        mask = data['phot_g_mean_mag'] >= t['mag_min']
    sub = data[mask]
    print(f"\n=== Quaia {t['name']} ===")
    print(f"  {len(sub):,} objects")
    z = sub['redshift_quaia']
    pos = radecz2cart(cosmo, sub['ra'], sub['dec'], z)
    densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
    write_output(ROOT / 'quaia' / t['id'], pos, t['id'], t['name'], t['color'])

write_densities(ROOT / 'quaia' / 'manifest.json', densities)

print("\nDone.")
