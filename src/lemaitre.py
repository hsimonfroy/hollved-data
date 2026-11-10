import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from utils import ROOT, RAW, radecrad2cart, write_output

# Sources
#   Velocities (heliocentric, corrected for solar motion):
#     Strömberg 1925, ApJ 61, 353 — "Analysis of Radial Velocities of Globular
#     Clusters and Non-Galactic Nebulae"
#   Distances (Cepheid and other distance indicators):
#     Hubble 1929, PNAS 15, 168 — "A Relation between Distance and Radial
#     Velocity among Extra-Galactic Nebulae"
#   Sample and H₀ estimate:
#     Lemaître 1927, Ann. Soc. Sci. Bruxelles A47, 49 — "Un Univers homogène
#     de masse constante et de rayon croissant rendant compte de la vitesse
#     radiale des nébuleuses extra-galactiques"

# Comparison of H₀ estimates from Lemaître 1927 data
# H₀ = Σ(w·v) / Σ(w·d),  r = distance (Mpc)
#
# Method                                mean v    mean d      H₀
#                                       (km/s)    (Mpc)   (km/s/Mpc)
# Lemaître unweighted                     670      1.16       575
# my unweighted                           646      1.22       530
# Lemaître weighted  (w ∝ 1/√(1+r²)?)      600      0.95       625
# my weighted,       w = 1/√(1+r²)        600      1.03       582
# my weighted,       w = 1/(1+r²)         558      0.90       623

data = np.genfromtxt(RAW / 'lemaitre1927.csv', delimiter=',', names=True)
ra  = data['ra_deg']
dec = data['dec_deg']
d   = data['dist_Mpc']
v   = data['vel_hel_kms']  # heliocentric velocities from Strömberg 1925

H0_lemaitre = (1.0 / (1.0 + d**2) * v).sum() / (1.0 / (1.0 + d**2) * d).sum()  # ≈ 623
H0_planck   = 67.66

TRACERS = [
    {'id': 'original',  'name': 'Original',  'color': '0x26ff00ff', 'dist_scale': 1.0},
    {'id': 'corrected', 'name': 'Corrected', 'color': '0xff0088ff', 'dist_scale': H0_lemaitre / H0_planck},
]

out_dir = ROOT / 'lemaitre'
with open(out_dir / 'manifest.json', 'w') as f:
    json.dump({'tracers': [t['id'] for t in TRACERS]}, f, separators=(',', ':'))
for t in TRACERS:
    pos = np.array(radecrad2cart(ra, dec, d * t['dist_scale']))
    write_output(out_dir / t['id'], pos, t['id'], t['name'], t['color'])






plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
})

# Hubble diagram — H₀ = Σ(w·v) / Σ(w·d)
def hubble_constant(v, d, w):
    return (w * v).sum() / (w * d).sum()

H0_unw = hubble_constant(v, d, np.ones_like(d))     # ≈ 530 km/s/Mpc
H0_w   = hubble_constant(v, d, 1.0 / (1.0 + d**2)) # ≈ 623 km/s/Mpc
H0_planck = 67.66


print(f"H₀ weighted 1/(1+d²) : {H0_w:.1f} km/s/Mpc  (expect ~623)")
print(f"H₀ unweighted        : {H0_unw:.1f} km/s/Mpc  (expect ~530)")

fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
ax.scatter(d, v, c='k')
d_fit = np.linspace(0, d.max() * 1.05, 50)
# ax.plot(d_fit, H0_w * d_fit, 'k', label=f'Weighted $H_0 = {H0_w:.0f}$ km/s/Mpc')
# ax.plot(d_fit, H0_unw * d_fit, 'k--', label=f'Unweighted  $H_0 = {H0_unw:.0f}$ km/s/Mpc')

ax.plot(d_fit, H0_w * d_fit, 'k', label=f'weighted')
ax.plot(d_fit, H0_unw * d_fit, 'k--', label=f'unweighted')
ax.set_xlabel('Distance [Mpc]')
ax.set_ylabel('Velocity [km/s]')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.savefig(ROOT / 'src' / 'lemaitre1927.pdf', dpi=300)

print("\nDone.")
