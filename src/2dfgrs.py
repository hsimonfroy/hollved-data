import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Planck18, RAW, ROOT, radecz2cart, write_output, plot_chi_z, kde_chi_z, write_densities

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW / '2dFGRS.txt', sep=r'\s+', header=0, engine='python')
print(f"Loaded {len(df):,} objects")

# ── J2000 RA/Dec → decimal degrees ────────────────────────────────────────────
df['ra'] = (df['ra_h_j2000'] + df['ra_m_j2000'] / 60 + df['ra_s_j2000'] / 3600) * 15
sign = np.sign(df['dec_d_j2000'])
sign[sign == 0] = 1
df['dec'] = sign * (df['dec_d_j2000'].abs() + df['dec_m_j2000'] / 60 + df['dec_s_j2000'] / 3600)

# ── Star cut ───────────────────────────────────────────────────────────────────
# Stars are identified by z_helio < 0.002
# http://www.2dfgrs.net/Public/Survey/statusfinal.html
z_min = 0.002 
n_all = len(df)
z     = df['z_helio'].astype(float)
df    = df[z > z_min].reset_index(drop=True)
print(f"  Cut z <= {z_min} removed {n_all - len(df)} objects over {n_all:,}")

# ── Red / Blue split on SB_BJ - SR_R ──────────────────────────────────────────
color  = df['SB_BJ'].astype(float) - df['SR_R'].astype(float)
is_blue = color < 1.07
df_red  = df[~is_blue]
df_blue = df[is_blue]

z_kc = df['z_helio'].astype(float)
y    = z_kc / (1 + z_kc)
k_bj = (-1.63 + 4.53 * color)*y + (-4.03 - 2.01 * color)*y**2 - z_kc / (1 + (10*z_kc)**4)
k_rf = (-0.08 + 1.45 * color)*y + (-2.88 - 0.48 * color)*y**2
color_rest = color - k_bj + k_rf

is_blue_rest = color_rest < 1.07
df_red_rest  = df[~is_blue_rest]
df_blue_rest = df[is_blue_rest]
print(f"  Red: {len(df_red_rest):,}  Blue: {len(df_blue_rest):,}")

# ── ETA_TYPE validation histograms ────────────────────────────────────────────
# eta_red  = df_red['ETA_TYPE'].astype(float)
# eta_blue = df_blue['ETA_TYPE'].astype(float)
# eta_red  = eta_red[eta_red > -99.]
# eta_blue = eta_blue[eta_blue > -99.]

# eta_red_rest  = df_red_rest['ETA_TYPE'].astype(float)
# eta_blue_rest = df_blue_rest['ETA_TYPE'].astype(float)
# eta_red_rest  = eta_red_rest[eta_red_rest > -99.]
# eta_blue_rest = eta_blue_rest[eta_blue_rest > -99.]

# plt.figure(figsize=(8, 6), layout='constrained')
# plt.suptitle('Comparing raw vs. restframe color split, and link to $\eta < -1.4$ split')
# # eta < -1.4 motivated by [Madgwick+2002](https://adsabs.harvard.edu/pdf/2002MNRAS.333..133M)
# p = 0.99
# q_lo = (1 - p) / 2
# q_hi = 1 - q_lo
# color_lo, color_hi = np.quantile(color, [q_lo, q_hi])
# color_rest_lo, color_rest_hi = np.quantile(color_rest, [q_lo, q_hi])
# eta_lo, eta_hi = np.quantile(np.concatenate([eta_red_rest, eta_blue_rest]), [q_lo, q_hi])

# plt.subplot(2, 2, 1)
# plt.hist(color[~is_blue],  bins=80, range=(color_rest_lo, color_rest_hi), color='#ff2020aa', label='Red')
# plt.hist(color[is_blue], bins=80, range=(color_rest_lo, color_rest_hi), color='#24bddcaa', label='Blue')
# plt.xlabel('Raw BJ - RF'), plt.ylabel('count')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.hist(eta_red,  bins=80, range=(eta_lo, eta_hi), color='#ff2020aa', label='Red')
# plt.hist(eta_blue, bins=80, range=(eta_lo, eta_hi), color='#24bddcaa', label='Blue')
# plt.axvline(-1.4, color='k', ls='--', lw=1)
# plt.xlabel('ETA_TYPE'), plt.ylabel('count')

# plt.subplot(2, 2, 3)
# plt.hist(color_rest[~is_blue_rest],  bins=80, range=(color_rest_lo, color_rest_hi), color='#ff2020aa', label='Restframe Red')
# plt.hist(color_rest[is_blue_rest], bins=80, range=(color_rest_lo, color_rest_hi), color='#24bddcaa', label='Restframe Blue')
# plt.xlabel('Restframe BJ - RF'), plt.ylabel('count')
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.hist(eta_red_rest,  bins=80, range=(eta_lo, eta_hi), color='#ff2020aa', label='Restframe Red')
# plt.hist(eta_blue_rest, bins=80, range=(eta_lo, eta_hi), color='#24bddcaa', label='Restframe Blue')
# plt.axvline(-1.4, color='k', ls='--', lw=1)
# plt.xlabel('ETA_TYPE'), plt.ylabel('count')

# plt.savefig('./2dFGRS_color_split.png', dpi=200)

# ── Compute positions and write output ────────────────────────────────────────
cosmo = Planck18()
# from matplotlib import pyplot as plt
# plt.figure(figsize=(8, 6), layout='constrained')

TRACERS = [
    {'df': df_red_rest,  'id': 'red',  'name': 'Red',  'color': '0xff0011ff'},
    {'df': df_blue_rest, 'id': 'blue', 'name': 'Blue', 'color': '0x24bddcff'},
]
densities = []
for t in TRACERS:
    print(f"\n=== 2dFGRS {t['name']} ===")
    d   = t['df']
    ra  = d['ra'].to_numpy(dtype=float)
    dec = d['dec'].to_numpy(dtype=float)
    z   = d['z_helio'].to_numpy(dtype=float)
    pos = radecz2cart(cosmo, ra, dec, z)
    densities.append(kde_chi_z(cosmo, pos, z, n_disc=64, scale=1.))
    # write_output(ROOT / '2dfgrs' / t['id'], pos, t['id'], t['name'], t['color'])



    # plot_chi_z(cosmo, pos, z, t, n_disc=128, scale=1.)
# plt.legend()
# plt.savefig(ROOT / 'src' / '2dfgrs_n128.png', dpi=300)

write_densities(ROOT / '2dfgrs' / 'manifest.json', densities)

print("\nDone.")
