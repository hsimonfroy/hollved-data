import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fitsio
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, plot_chi_z, kde_chi_z, write_densities

TRACERS = [
    {'id': 'bgs', 'name': 'BGS', 'color': '0x24bddcff', 'file': 'DESI_DR1_BGS.fits'},
    {'id': 'lrg', 'name': 'LRG', 'color': '0xf2df32ff', 'file': 'DESI_DR1_LRG.fits'},
    {'id': 'elg', 'name': 'ELG', 'color': '0xf99459ff', 'file': 'DESI_DR1_ELG.fits'},
    {'id': 'qso', 'name': 'QSO', 'color': '0xff0011ff', 'file': 'DESI_DR1_QSO.fits'},
]

cosmo = Planck18()
# from matplotlib import pyplot as plt
# plt.figure(figsize=(8, 6), layout='constrained')

densities = []
for t in TRACERS:
    print(f"\n=== DESI {t['name']} ===")
    data = fitsio.read(str(RAW / t['file']), columns=['ra', 'dec', 'z'])
    print(f"  Loaded {len(data):,} objects")
    z = data['z']
    pos = radecz2cart(cosmo, data['ra'], data['dec'], z)
    densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
    # write_output(ROOT / 'desi' / t['id'], pos, t['id'], t['name'], t['color'])


#     plot_chi_z(cosmo, pos, data['z'], t, n_disc=16, scale=1.)
# plt.legend()
# plt.savefig(ROOT / 'src' / 'desi_kde_n16.png', dpi=300)

write_densities(ROOT / 'desi' / 'manifest.json', densities)

print("\nDone.")
