"""
Preview script for the hollved welcome page timeline.

Reads manifest.json from each survey folder and plots a 2D timeline where:
  - x axis: years
  - y axis: galaxy count (log scale)
  - each program is a horizontal bar from start to end year at height = count
  - survey logos sit above the bars, unifying multi-program surveys (e.g. CfA1/CfA2)

Usage:
    python src/timeline.py
"""

from email import errors
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np

from utils import SetDark2, Planck18, a2chi, chi2a


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
})

FOLDER = Path(__file__).parent
ROOT = FOLDER.parent
SURVEYS = ['cfa', '2dfgrs', 'sdss2', 'sdss4', 'quaia', 'desi', 'euclid']
POSITIONS = ['above', 'below', 'above', 'above', 'below', 'above', 'below']
MAX_LOGO_WIDTH = 80
MAX_LOGO_HEIGHT = 60
LOGO_OFFSET = 17
TEXT_OFFSET = 1.1
SPECTROSCOPY_COLORS = {
    'single-slit':       "#ada497",
    'plate-based fibers': "#a29ad3",
    'slitless':          "#dd7276",
    'robotic fibers':    "#3fd1a0",
}
SPECTROSCOPY_ORDER = ['single-slit', 'plate-based fibers', 'slitless', 'robotic fibers']

# h from Planck18 cosmology (utils.py)
COSMO = Planck18()
H_PLANCK18 = float(COSMO.h)


def add_redshift_axis(ax, cosmo=COSMO, zticks=(0, 0.2, 0.5, 1, 1.5, 2, 3, 4, 6)):
    """Top x-axis showing the redshift corresponding to comoving distance [Mpc/h].

    NOTE: jax_cosmo distances are natively in Mpc/h, matching the plotted unit.
    """
    def chih2z(chih):
        a = chi2a(cosmo, np.maximum(np.asarray(chih, dtype=float), 0.0))
        return np.asarray(1.0 / a - 1.0)
    def z2chih(z):
        a = 1.0 / (1.0 + np.maximum(np.asarray(z, dtype=float), 0.0))
        return np.asarray(a2chi(cosmo, a))
    secax = ax.secondary_xaxis('top', functions=(chih2z, z2chih))
    secax.set_xlabel(r'Redshift $z$')
    secax.set_xticks(list(zticks))
    secax.set_xticklabels([f'{z:g}' for z in zticks])
    return secax

# tracer colors, from all/*/meta.json (0xRRGGBBFF → #RRGGBB)
TRACER_COLORS = {
    'cfa1':   "#d4cabb",
    'cfa2':   "#9f978b",
    '2dfgrs': "#24bddc",
    'sdss2':  '#afa6e1',
    'sdss4':  "#6f5dd7",
    'quaia':  '#ff5c00',
    # 'euclid': '#f07d81',
    'desi':   '#44ddaa',
}
TRACER_NAMES = {
    'cfa1':   'CfA1',
    'cfa2':   'CfA2',
    '2dfgrs': '2dFGRS',
    'sdss2':  'SDSS Legacy',
    'sdss4':  'SDSS BOSS+',
    'quaia':  'Quaia',
    'euclid': 'Euclid',
    'desi':   'DESI DR1',
}

# DESI tracer colors and names, from desi/*/meta.json (0xRRGGBBFF → #RRGGBB)
DESI_TRACER_COLORS = {
    'bgs': '#24bddc',
    'lrg': '#f2df32',
    'elg': '#f99459',
    'qso': '#ff0011',
}
DESI_TRACER_NAMES = {
    'bgs': 'BGS',
    'lrg': 'LRG',
    'elg': 'ELG',
    'qso': 'QSO',
}


def parse_color(hex_str):
    """Convert '0xRRGGBBFF' to normalized (r, g, b, a) floats."""
    c = int(hex_str, 16)
    return (
        (c >> 24 & 0xFF) / 255,
        (c >> 16 & 0xFF) / 255,
        (c >>  8 & 0xFF) / 255,
        (c       & 0xFF) / 255,
    )


def get_logo_zoom_and_height(logo_path):
    """Return (zoom, rendered_height_in_points) to fit logo within MAX_LOGO_WIDTH/HEIGHT."""
    logo = mpimg.imread(logo_path)
    height, width = logo.shape[:2]
    zoom_w = MAX_LOGO_WIDTH / width
    zoom_h = MAX_LOGO_HEIGHT / height
    zoom = min(zoom_w, zoom_h)
    rendered_height = height * zoom  # in display points (72 dpi basis for OffsetImage)
    return zoom, rendered_height


def load_surveys():
    surveys = []
    for survey_id in SURVEYS:
        with open(ROOT / survey_id / 'manifest.json') as f:
            manifest = json.load(f)

        logo_path = ROOT / survey_id / 'logo_light.png'
        if not logo_path.exists():
            logo_path = ROOT / survey_id / 'logo.png'
        surveys.append({
            'id': survey_id,
            'logo': logo_path if logo_path.exists() else None,
            'programs': manifest['programs'],
        })
    return surveys


def load_all():
    with open(ROOT / 'all' / 'manifest.json') as f:
        return json.load(f)


def load_manifest(survey_id):
    with open(ROOT / survey_id / 'manifest.json') as f:
        return json.load(f)


def linear_regression(surveys):
    """Extract (log_count, mean_year, std_year) from all programs."""
    log_counts = []
    mean_years = []
    std_years = []

    for survey in surveys:
        for prog in survey['programs']:
            mean_year = (prog['start'] + prog['end']) / 2
            std_year = (prog['end'] - prog['start']) / 2

            log_counts.append(np.log10(prog['count']))
            mean_years.append(mean_year)
            std_years.append(std_year)

    log_counts, mean_years, std_years = np.array(log_counts), np.array(mean_years), np.array(std_years)
    # coeffs = np.polyfit(log_counts, mean_years, deg=1, w=1/std_years)
    coeffs = np.polyfit(log_counts, mean_years, deg=1, w=np.ones_like(std_years))
    return coeffs


def draw_timeline(surveys):
    fig, ax = plt.subplots(figsize=(7, 5.5), layout='constrained')
    ax.set_yscale('log')

    for survey_idx, survey in enumerate(surveys):
        # r, g, b, _ = survey['color']
        # face_color = (r, g, b, 0.70)
        # edge_color = (r, g, b, 1.00)

        position = POSITIONS[survey_idx]  # 'above' or 'below'

        for prog_idx, prog in enumerate(survey['programs']):
            start = prog['start']
            end = prog['end']
            count = prog['count']
            name = prog['name']
            if name == 'SDSS BOSS-eBOSS': ### XXX
                name = 'SDSS BOSS+'

            # Draw horizontal line/bar from start to end at height = count
            spec_type = prog['spectroscopy']
            color = SPECTROSCOPY_COLORS[spec_type]
            ax.plot([start, end], [count, count],
                    color=color, linewidth=2, solid_capstyle='butt')

            # Add program name annotation
            mid_year = (start + end) / 2
            if position == 'above':
                # ax.text(mid_year, count*TEXT_OFFSET, r'\textbf{' + name + r'}', ha='center', va='bottom',
                ax.text(mid_year, count*TEXT_OFFSET, rf'{name}', ha='center', va='bottom',
                        fontsize=12, weight='bold')
            else:
                # ax.text(mid_year, count/TEXT_OFFSET/1.05, r'\textbf{' + name + r'}', ha='center', va='top',
                ax.text(mid_year, count/TEXT_OFFSET/1.05, rf'{name}', ha='center', va='top',
                        fontsize=12, weight='bold')

            # Add logo only for first program of the survey
            if prog_idx == 0 and survey['logo']:
                logo = mpimg.imread(survey['logo'])
                zoom, logo_height = get_logo_zoom_and_height(survey['logo'])
                imagebox = OffsetImage(logo, zoom=zoom)
                logo_offset = logo_height / 2 + LOGO_OFFSET  # half logo height + small gap

                if position == 'above':
                    # Logo above the line
                    ab = AnnotationBbox(imagebox, (mid_year, count),
                                        xybox=(0, logo_offset), boxcoords='offset points',
                                        frameon=False)
                else:
                    # Logo below the line
                    ab = AnnotationBbox(imagebox, (mid_year, count),
                                        xybox=(0, -logo_offset), boxcoords='offset points',
                                        frameon=False)
                ax.add_artist(ab)

    # Add linear regression line
    coeffs = linear_regression(surveys)
    slope, intercept = coeffs[0], coeffs[1]
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    count_range = np.linspace(*ylim, 2)
    year_pred = slope * np.log10(count_range) + intercept
    # ax.legend(loc='upper left', fontsize=9)

    ax.plot(year_pred, count_range, 
            'k--', alpha=0.2, linewidth=1, 
            label=f'log10(N) = {slope:.3f}·year + {intercept:.1f}')
    print(f"Linear regression: log10(N) = {slope:.3f}·year + {intercept:.1f}")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim[0], ylim[1] * 1.9)
    ax.set_xlabel('Year')
    ax.set_ylabel('Galaxies')
    ax.grid(True, alpha=0.3)
    ax.spines[['right', 'top']].set_visible(False)

    seen = {prog['spectroscopy'] for s in surveys for prog in s['programs'] if 'spectroscopy' in prog}
    handles = [
        Line2D([0], [0], color=SPECTROSCOPY_COLORS[t], linewidth=2, label=t)
        for t in SPECTROSCOPY_ORDER if t in seen
    ]
    ax.legend(handles=handles, title='Spectroscopy', loc='upper left', framealpha=0.7)

    # plt.savefig(FOLDER / 'timeline.png', dpi=300)
    plt.savefig(FOLDER / 'timeline.pdf', dpi=300)





def draw_vdensities(manifest, colors, names, suffix='', legend=True, fontsize=None):
    h   = H_PLANCK18
    tracers   = manifest['tracers']
    footprint = manifest['footprint']
    dens      = manifest['densities']
    with plt.rc_context({} if fontsize is None else {'font.size': fontsize}):
        fig, ax = plt.subplots(figsize=(6, 4.5), layout='constrained')
        ax.set_yscale('log')
        for i, tracer_id in enumerate(tracers):
            if tracer_id not in colors:
                continue  # skip tracers without defined colors
            chi    = np.array(dens['chi_Mpc'][i])
            dndchi = np.array(dens['dndchi'][i])
            fsky   = footprint[i] / 41253.0
            chi_h  = chi * h
            vdens  = dndchi / h / (4 * np.pi * chi_h**2 * fsky)  # [gal/(Mpc/h)^3]
            color  = colors[tracer_id]
            name   = names[tracer_id]
            ax.plot(chi_h, vdens, color=color, label=name, linewidth=1.5)
        ax.set_xlabel(r'Comoving distance $\chi$ [Mpc/$h$]')
        ax.set_ylabel(r'Volume density $\frac{\mathrm{d}N}{\mathrm{d}V}$ [Mpc/$h$]$^{-3}$')
        add_redshift_axis(ax)
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.spines[['right', 'top']].set_visible(False)
        # plt.xscale('log')
        # plt.savefig(FOLDER / f'densities_vol{suffix}.png', dpi=300)
        plt.savefig(FOLDER / f'densities_vol{suffix}.pdf', dpi=300)


def draw_rdensities(manifest, colors, names, suffix='', legend=False, fontsize=None):
    h   = H_PLANCK18
    tracers   = manifest['tracers']
    dens      = manifest['densities']
    with plt.rc_context({} if fontsize is None else {'font.size': fontsize}):
        fig, ax = plt.subplots(figsize=(6, 4.5), layout='constrained')
        ax.set_yscale('log')
        for i, tracer_id in enumerate(tracers):
            if tracer_id not in colors:
                continue  # skip tracers without defined colors
            chi      = np.array(dens['chi_Mpc'][i])
            dndchi   = np.array(dens['dndchi'][i])
            chi_h    = chi * h
            dndchi_h = dndchi / h                   # [gal/(Mpc/h)]
            color    = colors[tracer_id]
            name     = names[tracer_id]
            ax.plot(chi_h, dndchi_h, color=color, label=name, linewidth=1.5)
        ax.set_xlabel(r'Comoving distance $\chi$ [Mpc/$h$]')
        ax.set_ylabel(r'Radial density $\frac{\mathrm{d}N}{\mathrm{d}\chi}$ [Mpc/$h$]$^{-1}$')
        add_redshift_axis(ax)
        if legend:
            ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.spines[['right', 'top']].set_visible(False)
        # plt.xscale('log')
        # plt.savefig(FOLDER / f'densities_rad{suffix}.png', dpi=300)
        plt.savefig(FOLDER / f'densities_rad{suffix}.pdf', dpi=300)


if __name__ == '__main__':
    surveys = load_surveys()
    # with plt.xkcd():
    draw_timeline(surveys)

    # per-survey densities (all/manifest.json)
    all_manifest = load_all()
    draw_vdensities(all_manifest, TRACER_COLORS, TRACER_NAMES)
    draw_rdensities(all_manifest, TRACER_COLORS, TRACER_NAMES)

    # per-DESI-tracer densities (desi/manifest.json)
    desi_manifest = load_manifest('desi')
    draw_vdensities(desi_manifest, DESI_TRACER_COLORS, DESI_TRACER_NAMES, suffix='_desi', fontsize=16)
    draw_rdensities(desi_manifest, DESI_TRACER_COLORS, DESI_TRACER_NAMES, suffix='_desi', legend=True, fontsize=16)
