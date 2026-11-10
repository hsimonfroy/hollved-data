import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import fitsio
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, plot_chi_z, kde_chi_z, write_densities

TRACERS = [
    {
        'id': 'lowz', 'name': 'LOWZ', 'color': '0x24bddcff',
        'files': [
            'galaxy_DR12v5_LOWZ_North.fits',
            'galaxy_DR12v5_LOWZ_South.fits',
        ],
    },
    {
        'id': 'cmass', 'name': 'CMASS', 'color': '0x258e66ff',
        'files': [
            'galaxy_DR12v5_CMASS_North.fits',
            'galaxy_DR12v5_CMASS_South.fits',
        ],
    },
    {
        'id': 'lrg', 'name': 'LRG', 'color': '0xf2df32ff',
        'files': [
            'eBOSS_LRG_clustering_data-NGC-vDR16.fits',
            'eBOSS_LRG_clustering_data-SGC-vDR16.fits',
        ],
    },
    {
        'id': 'elg', 'name': 'ELG', 'color': '0xf99459ff',
        'files': [
            'eBOSS_ELG_clustering_data-NGC-vDR16.fits',
            'eBOSS_ELG_clustering_data_rec-SGC-vDR16.fits',
        ],
    },
    {
        'id': 'qso', 'name': 'QSO', 'color': '0xff0011ff',
        'files': [
            'eBOSS_QSO_clustering_data-NGC-vDR16.fits',
            'eBOSS_QSO_clustering_data-SGC-vDR16.fits',
        ],
    },
]

# cosmo = Planck18()
# # from matplotlib import pyplot as plt
# # plt.figure(figsize=(8, 6), layout='constrained')

# densities = []
# for t in TRACERS:
#     print(f"\n=== SDSS4 {t['name']} ===")
#     parts = [fitsio.read(str(RAW / f), columns=['RA', 'DEC', 'Z']) for f in t['files']]
#     print(f"  Files: {' + '.join(str(len(p)) for p in parts)} = {sum(len(p) for p in parts):,}")
#     data = np.concatenate(parts)
#     ra  = data['RA'].astype(np.float64)
#     dec = data['DEC'].astype(np.float64)
#     z   = data['Z'].astype(np.float64)  # BOSS Z is float32; promote for consistency
#     pos = radecz2cart(cosmo, ra, dec, z)

#     # LOWZ selection over z~0.15, but take daring margin to showcase neighborhood
#     z_min = 0.01
#     pos = pos[z > z_min]
#     z = z[z > z_min]
#     print(f"  Cut z <= {z_min} removed {len(data) - len(pos)} objects over {len(data):,}")
#     densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
#     write_output(ROOT / 'sdss4' / t['id'], pos, t['id'], t['name'], t['color'])


# #     plot_chi_z(cosmo, pos, z, t, n_disc=64, scale=1.)
# # plt.legend()
# # plt.savefig(ROOT / 'src' / 'sdss4_kde_cut.png', dpi=300)

# write_densities(ROOT / 'sdss4' / 'manifest.json', densities)

# print("\nDone.")










def zstats(z):
    nnan = int(np.isnan(z).sum())
    zmin, zmax = np.nanquantile(z, 0.005), np.nanquantile(z, 0.995)
    return zmin, zmax, nnan


def row(label, z):
    zmin, zmax, nnan = zstats(z)
    print(f"  {label:<10} {len(z):>8,}      {zmin:.3f}   {zmax:.3f}   nan: {nnan:,}")


def combined_row(label, z1, z2):
    z = np.concatenate([z1, z2])
    zmin, zmax, nnan = zstats(z)
    print(f"  {label:<10} {len(z):>8,}      {zmin:.3f}   {zmax:.3f}   nan: {nnan:,}")


def read_z(path):
    return fitsio.read(path, columns=["Z"])["Z"]


header = f"  {'':10} {'rows':>8}       zmin    zmax"

# eBOSS LRG
lrg_ngc  = read_z(f"{RAW}/eBOSS_LRG_clustering_data-NGC-vDR16.fits")
lrg_sgc  = read_z(f"{RAW}/eBOSS_LRG_clustering_data-SGC-vDR16.fits")
lrg_full = read_z(f"{RAW}/eBOSS_LRG_full_ALLdata-vDR16.fits")

print("eBOSS LRG")
print(header)
row("NGC", lrg_ngc)
row("SGC", lrg_sgc)
combined_row("NGC+SGC", lrg_ngc, lrg_sgc)
row("fulldata", lrg_full)

# eBOSS ELG
elg_ngc  = read_z(f"{RAW}/eBOSS_ELG_clustering_data-NGC-vDR16.fits")
elg_sgc  = read_z(f"{RAW}/eBOSS_ELG_clustering_data_rec-SGC-vDR16.fits")
elg_full = read_z(f"{RAW}/eBOSS_ELG_full_ALLdata-vDR16.fits")

print("\neBOSS ELG")
print(header)
row("NGC", elg_ngc)
row("SGC", elg_sgc)
combined_row("NGC+SGC", elg_ngc, elg_sgc)
row("fulldata", elg_full)

# eBOSS QSO
qso_ngc  = read_z(f"{RAW}/eBOSS_QSO_clustering_data-NGC-vDR16.fits")
qso_sgc  = read_z(f"{RAW}/eBOSS_QSO_clustering_data-SGC-vDR16.fits")
qso_full = read_z(f"{RAW}/eBOSS_QSO_full_ALLdata-vDR16.fits")

print("\neBOSS QSO")
print(header)
row("NGC", qso_ngc)
row("SGC", qso_sgc)
combined_row("NGC+SGC", qso_ngc, qso_sgc)
row("fulldata", qso_full)
