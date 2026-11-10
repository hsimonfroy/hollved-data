import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from astropy.io.votable import parse as vot_parse
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, kde_chi_z, write_densities

TRACER = {'id': 'gxy', 'name': 'GXY', 'color': '0x24bddcff'}

cosmo = Planck18()

print("=== Euclid Q1 GXY ===")
vot = vot_parse(str(RAW / 'Euclid_Q1.vot'))
table = vot.get_first_table().to_table()

ra  = np.array(table['right_ascension'], dtype=np.float64)
dec = np.array(table['declination'],     dtype=np.float64)
z   = np.array(table['spe_z'],           dtype=np.float64)

mask = z > 0
ra, dec, z = ra[mask], dec[mask], z[mask]
print(f"  {len(table):,} total, cut z <= 0: {(~mask).sum()} removed, {mask.sum():,} kept")
pos = radecz2cart(cosmo, ra, dec, z)

write_output(ROOT / 'euclid' / 'gxy', pos, TRACER['id'], TRACER['name'], TRACER['color'])
write_densities(ROOT / 'euclid' / 'manifest.json', [kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.)])
print("Done.")
