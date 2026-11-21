import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, kde_chi_z, write_densities

TRACER = {'id': 'ha', 'name': 'Hα', 'color': '0x24bddcff'}

cosmo = Planck18()

# Catalogue from src/euclid_dr1_query.py: spe_z_prob >= 0.999, Halpha flux >= 2e-16, S/N >= 7, extended sources
print("=== Euclid DR1 Hα ===")
df = pd.read_csv(RAW / 'Euclid_DR1.csv.gz')

ra  = df['right_ascension'].to_numpy(np.float64)
dec = df['declination'].to_numpy(np.float64)
z   = df['spe_z'].to_numpy(np.float64)

mask = (z > 0.84) & (z < 1.88)  # Halpha range in the red grism
ra, dec, z = ra[mask], dec[mask], z[mask]
print(f"  {len(df):,} total, cut outside 0.84 < z < 1.88: {(~mask).sum()} removed, {mask.sum():,} kept")
pos = radecz2cart(cosmo, ra, dec, z)

write_output(ROOT / 'euclid' / 'ha', pos, TRACER['id'], TRACER['name'], TRACER['color'])
write_densities(ROOT / 'euclid' / 'manifest.json', [kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.)])
print("Done.")
