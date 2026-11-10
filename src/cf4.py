import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fitsio
import numpy as np
from jax import numpy as jnp
from utils import Planck18, DL2a, a2chi, radecrad2cart, write_output, kde_chi_z, write_densities, ROOT, RAW

TRACERS = [
    {'id': 'individual', 'name': 'Individual', 'color': '0x26ff00ff', 'file': 'cf4_individual.fits'},
    {'id': 'group',      'name': 'Group',      'color': '0xff1493ff', 'file': 'cf4_group.fits'},
]

cosmo = Planck18()

out_root = ROOT / 'cf4'
manifest_path = out_root / 'manifest.json'
with open(manifest_path, 'w') as f:
    json.dump({'tracers': [t['id'] for t in TRACERS]}, f, separators=(',', ':'))

densities = []
for t in TRACERS:
    print(f"\n=== CF4 {t['name']} ===")
    data = fitsio.read(str(RAW / t['file']), columns=['RAJ2000', 'DEJ2000'] + (['DM'] if t['id'] == 'individual' else ['Dist']))
    print(f"  {len(data):,} objects")

    ra  = data['RAJ2000'].astype(np.float64)
    dec = data['DEJ2000'].astype(np.float64)

    if t['id'] == 'individual':
        dm   = data['DM'].astype(np.float64)
        dist = 10 ** ((dm - 25) / 5)              # distance modulus → luminosity distance [Mpc]
    else:
        dist = data['Dist'].astype(np.float64)    # luminosity distance [Mpc]

    DL_h = jnp.array(dist * cosmo.h)              # Mpc → Mpc/h (jax_cosmo internal units)
    a    = DL2a(cosmo, DL_h)
    chi  = a2chi(cosmo, a) / cosmo.h              # Mpc/h → Mpc comoving
    z    = np.array(1 / a - 1)

    pos = np.array(radecrad2cart(ra, dec, chi))
    densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
    write_output(out_root / t['id'], pos, t['id'], t['name'], t['color'])

write_densities(manifest_path, densities)
print("\nDone.")
