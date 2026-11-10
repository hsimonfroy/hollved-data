# hollved-data

Provide data for hollved visualization.

## Datasets
Redshift catalogs:
* [CfA](https://lweb.cfa.harvard.edu/~dfabricant/huchra/zcat/) with [CfA1](https://lweb.cfa.harvard.edu/~dfabricant/huchra/zcat/nz40.dat) and [CfA2](https://lweb.cfa.harvard.edu/~dfabricant/huchra/zcat/cfa2.dat)
* [2dFGRS](http://www.2dfgrs.net/Public/Release/SpecCat/index.html) with [color split procedure](https://arxiv.org/pdf/astro-ph/0501174)
* [SDSS2](https://www.sdss2.org/home.php) with [DR7 Legacy Survey](https://classic.sdss.org/dr7/products/spectra/getspectra.php)
* [SDSS4](https://www.sdss4.org/) with [DR12 BOSS](https://data.sdss.org/sas/dr12/boss/lss/) and [DR16 eBOSS](https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/)
* [Quaia](https://zenodo.org/records/10403370)
* [Euclid](https://eas.esac.esa.int/sas/) with [Q1 quality cut](https://euclid.caltech.edu/page/euclid-q1-data-faq)
* [DESI](https://www.desi.lbl.gov/) with [DR1](https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/)

Auxiliary data:
* [Planck](https://www.esa.int/Science_Exploration/Space_Science/Planck) with [100GHz frequency map](https://pla.esac.esa.int/#maps)
* Main local group galaxies: [M31](https://www.reddit.com/r/spaceporn/comments/1f0xbeb/i_simulated_a_faceon_view_of_the_andromeda_galaxy/), [MW](https://www.esa.int/ESA_Multimedia/Images/2025/01/The_best_Milky_Way_map_by_Gaia), [M33](https://www.eso.org/public/images/eso1424a/), [LMC](https://www.esa.int/ESA_Multimedia/Images/2018/04/Large_Magellanic_Cloud), [SMC](https://www.esa.int/ESA_Multimedia/Images/2018/04/Small_Magellanic_Cloud), [M32](https://pages.astronomy.ua.edu/gifimages/m32.html), [M110](https://noirlab.edu/public/images/noao-m110/), [NGC 147](https://simbad.u-strasbg.fr/simbad/sim-id?Ident=NGC%20147), [NGC 185](https://simbad.u-strasbg.fr/simbad/sim-id?Ident=NGC+185), [NGC 6822](https://www.esa.int/Science_Exploration/Space_Science/Euclid/Euclid_s_view_of_irregular_galaxy_NGC_6822), [IC 1613](https://www.eso.org/public/images/eso1603a/), [IC 10](https://noirlab.edu/public/images/noirlab2013a/), [WLM](https://noirlab.edu/public/images/noao-wlm/)
* The [EDD](https://edd.ifa.hawaii.edu/dfirst.php) with [CF4](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/ApJ/944/94)
* Planet, moon and probe orbits: [JPL approximate elements](https://ssd.jpl.nasa.gov/planets/approx_pos.html) with [Horizons](https://ssd.jpl.nasa.gov/horizons/) vectors, IAU WGCCRE 2015 poles and the Astronomical Almanac lunar series
* Stars and constellations: [AT-HYG](https://codeberg.org/astronexus/athyg) with Stellarium's [Sky & Telescope figures](https://github.com/Stellarium/stellarium-skycultures/tree/master/western_SnT)

Recent datasets are also downloadable from [CosmoHub](https://cosmohub.pic.es/).

## Usage

Process survey files:
```
python src/<survey>.py
```

Local development:
```
npx http-server --cors -p 9090
```
