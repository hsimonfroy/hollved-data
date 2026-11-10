import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from utils import Planck18, ROOT, write_output, kde_chi_z, write_densities, chi2a

META_TRACERS = [
    {'id': 'cfa1',   'name': 'CfA1',            'color': '0x26ff00ff', 'sources': [('cfa', 'cfa1')],                                                            'program_ref': ('cfa', 0)},
    {'id': 'cfa2',   'name': 'CfA2',            'color': '0xff1493ff', 'sources': [('cfa', 'cfa2')],                                                            'program_ref': ('cfa', 1)},
    {'id': '2dfgrs', 'name': '2dFGRS',          'color': '0xfeed92ff', 'sources': [('2dfgrs', 'blue'), ('2dfgrs', 'red')],                                      'program_ref': ('2dfgrs', 0)},
    {'id': 'sdss2',  'name': 'SDSS Legacy',     'color': '0x24bddcff', 'sources': [('sdss2', 'mgs'), ('sdss2', 'lrg'), ('sdss2', 'qso')],                      'program_ref': ('sdss2', 0)},
    {'id': 'sdss4',  'name': 'SDSS BOSS-eBOSS', 'color': '0xafa6e1ff', 'sources': [('sdss4', 'lowz'), ('sdss4', 'cmass'), ('sdss4', 'lrg'), ('sdss4', 'elg'), ('sdss4', 'qso')], 'program_ref': ('sdss4', 0)},
    {'id': 'quaia',  'name': 'Quaia',           'color': '0xff5c00ff', 'sources': [('quaia', 'bright'), ('quaia', 'faint')],                                    'program_ref': ('quaia', 0)},
    {'id': 'euclid', 'name': 'Euclid',          'color': '0xf07d81ff', 'sources': [('euclid', 'gxy')],                                                          'program_ref': ('euclid', 0)},
    {'id': 'desi',   'name': 'DESI',            'color': '0x44ddaaff', 'sources': [('desi', 'bgs'), ('desi', 'lrg'), ('desi', 'elg'), ('desi', 'qso')],        'program_ref': ('desi', 0)},
]

cosmo = Planck18()
densities = []
programs  = []
tracer_ids = []
footprints = []

for mt in META_TRACERS:
    print(f"\n=== {mt['name']} ===")

    # Load source manifest for program metadata and footprint values
    src_survey, prog_idx = mt['program_ref']
    src_manifest = json.loads((ROOT / src_survey / 'manifest.json').read_text())
    prog = src_manifest['programs'][prog_idx]

    # Load and concatenate positions from all source tracers
    parts = []
    for survey, tracer in mt['sources']:
        sm = json.loads((ROOT / survey / 'manifest.json').read_text())
        path = ROOT / survey / tracer / 'positions.bin'
        part = np.fromfile(path, dtype='<f2').reshape(-1, 3).astype(np.float32)
        parts.append(part)
        print(f"  {survey}/{tracer}: {len(part):,}")
    pos = np.concatenate(parts, axis=0)
    print(f"  Total: {len(pos):,}")

    # Footprint = max of source tracers' footprint values
    fp_values = []
    for survey, tracer in mt['sources']:
        sm = json.loads((ROOT / survey / 'manifest.json').read_text())
        fp_values.append(sm['footprint'][sm['tracers'].index(tracer)])
    fp = max(fp_values)

    # Reconstruct z from positions (positions stored in Mpc)
    chi = np.linalg.norm(pos, axis=-1)
    a   = np.array(chi2a(cosmo, chi * cosmo.h))
    z   = 1.0 / a - 1.0

    # Write positions and meta
    write_output(ROOT / 'all' / mt['id'], pos, mt['id'], mt['name'], mt['color'])

    # Compute density profiles
    densities.append(kde_chi_z(cosmo, pos.astype(np.float64), z))

    programs.append({
        'name':         prog['name'],
        'start':        prog['start'],
        'end':          prog['end'],
        'count':        len(pos),
        'spectroscopy': prog['spectroscopy'],
    })
    tracer_ids.append(mt['id'])
    footprints.append(fp)

# Write manifest (without densities first, then write_densities adds them)
manifest_path = ROOT / 'all' / 'manifest.json'
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_path, 'w') as f:
    json.dump({'programs': programs, 'tracers': tracer_ids, 'footprint': footprints}, f, separators=(',', ':'))

write_densities(manifest_path, densities)
