"""
Milky Way stars for aux/stars/: the HYG catalogue plus Sky & Telescope's
constellation figures, packed for direct upload as typed arrays.

Two sources, both CC-BY-SA:

  hyglike_from_athyg_v40   Hipparcos + Yale BSC + Gliese, with Gaia DR3/DR2
                           distances. Same schema as plain HYG v4.4 and the same
                           size, but 119,524 stars have a usable parallax rather
                           than 109,389, and 744 of the 746 constellation-figure
                           stars resolve rather than 736.

  western_SnT/index.json   Stellarium's Sky & Telescope sky culture. There is no
                           official standard for constellation stick figures;
                           this is the simplest widely-used set. Polylines of
                           Hipparcos ids, which is why they join onto HYG's `hip`
                           with nothing in between.

The catalogue's x/y/z are ALREADY this scene's frame -- +X to the vernal equinox,
+Z to the north celestial pole, +Y to RA 6h -- so positions are a straight read.
They stay in PARSECS; the renderer carries pc -> Mpc on the object's scale, which
keeps the vertex buffer at O(100) instead of O(1e-6).

Run:  python stars.py
"""
import csv
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

HYG_URL = ("https://codeberg.org/astronexus/athyg/media/branch/main/"
           "data/subsets/hyglike_from_athyg_v40.csv.gz")
SNT_URL = ("https://raw.githubusercontent.com/Stellarium/stellarium-skycultures/"
           "master/western_SnT/index.json")

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "raw"                       # the repo's convention for downloads
OUT  = ROOT / "aux" / "stars"

# Beyond this the parallax is noise, and a wrong distance becomes a wrong
# LUMINOSITY: Polis comes out at absmag -16, brighter than a dwarf galaxy.
# Genuine values in this catalogue top out near 1 kpc.
MAX_DIST_PC = 10000.0

DEFAULT_CI = 0.65   # solar B-V, for the 1% with no colour index


def fetch(url, name):
    """Downloaded once into raw/, alongside every other source catalogue."""
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def bv_to_rgb(bv):
    """
    B-V colour index -> sRGB, vectorised.

    Ballesteros (2012) gives the effective temperature from the index alone --
    it treats the star as two black bodies and is good to a few percent over
    -0.4 < B-V < 2.0, which is the whole main sequence plus the giants:

        T = 4600 * (1/(0.92*BV + 1.7) + 1/(0.92*BV + 0.62))

    Temperature then goes to sRGB through the standard piecewise fit to the
    black-body locus, normalised so the brightest channel is 1: the dot's
    brightness is the flux, so the colour must carry hue only.
    """
    bv = np.clip(np.asarray(bv, dtype=np.float64), -0.4, 2.0)
    t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))

    x = np.clip(t, 1000.0, 40000.0) / 100.0
    r = np.where(x <= 66, 255.0,
                 329.698727446 * np.power(np.maximum(x - 60, 1e-9), -0.1332047592))
    g = np.where(x <= 66,
                 99.4708025861 * np.log(np.maximum(x, 1e-9)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(x - 60, 1e-9), -0.0755148492))
    b = np.where(x >= 66, 255.0,
                 np.where(x <= 19, 0.0,
                          138.5177312231 * np.log(np.maximum(x - 10, 1e-9))
                          - 305.0447927307))

    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0)
    # Hue only: divide out the intensity the fit carries, so a red dwarf and a
    # blue giant of equal flux render equally bright in different colours.
    rgb = rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1e-9) * 255.0
    return np.round(rgb).astype(np.uint8)


def load_stars():
    path = fetch(HYG_URL, "hyglike_from_athyg_v40.csv.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  {len(rows)} rows in the catalogue")

    def col(name, dtype=np.float64):
        out = np.empty(len(rows), dtype=dtype)
        for i, r in enumerate(rows):
            try:
                out[i] = float(r[name])
            except (ValueError, KeyError, TypeError):
                out[i] = np.nan
        return out

    dist = col("dist")
    keep = np.isfinite(dist) & (dist > 0) & (dist <= MAX_DIST_PC)
    keep &= np.isfinite(col("absmag"))
    print(f"  {keep.sum()} with a usable distance (0 < d <= {MAX_DIST_PC:.0f} pc)")

    idx = np.flatnonzero(keep)
    xyz = np.stack([col("x")[idx], col("y")[idx], col("z")[idx]], axis=-1)

    # HYG parks "Sol" at a placeholder 1 AU along +X. It is the catalogue's
    # ORIGIN, and the renderer translates the whole field onto the Sun, so the
    # placeholder would put the Sun 1 AU away from itself -- 3 degrees of error
    # seen from Neptune, which is exactly where the corona hands over to it.
    sol = [k for k, i in enumerate(idx) if rows[i]["id"] == "0"]
    if sol:
        xyz[sol[0]] = 0.0
        print(f"  Sol found at row {sol[0]}, moved to the catalogue origin")
    absmag = col("absmag")[idx]
    ci = col("ci")[idx]
    ci = np.where(np.isfinite(ci), ci, DEFAULT_CI)

    # hip -> row in the packed arrays, for the constellation figures
    hip_to_row = {}
    for row, i in enumerate(idx):
        h = rows[i].get("hip")
        if h:
            hip_to_row.setdefault(int(h), row)

    # Proper names, with the apparent magnitude that lets the renderer choose how
    # many to show. Both are catalogue facts; WHICH of them get drawn is a look
    # decision and stays on the client. Sol is excluded -- the solar overlay owns
    # that name, and it is the one object that is both a body and a star.
    mag = col("mag")
    names = [{"i": row, "name": rows[i]["proper"].strip(),
              "mag": round(float(mag[i]), 3) if np.isfinite(mag[i]) else None}
             for row, i in enumerate(idx)
             if rows[i]["proper"].strip() and rows[i]["proper"].strip() != "Sol"]
    print(f"  {len(names)} stars carry a proper name")

    return xyz, absmag, ci, hip_to_row, names


def load_lines(hip_to_row):
    """
    Flat segment pairs, plus the per-figure grouping the source already has.

    The pairs alone would be enough to DRAW the figures, but not to label them:
    a name needs to know which segments are its own so the client can derive an
    anchor from the shared position buffer. Two integers per figure buys that,
    and keeps the anchor from becoming a second definition stored in this file.
    """
    path = fetch(SNT_URL, "western_SnT.json")
    culture = json.loads(path.read_text(encoding="utf-8"))

    seg, figures, dropped, missing = [], [], 0, set()
    for con in culture["constellations"]:
        start = len(seg)
        for path_pts in con.get("lines", []):
            # 'bold' / 'thin' tokens are Sky & Telescope's print line weights.
            hips = [p for p in path_pts if isinstance(p, int)]
            for a, b in zip(hips, hips[1:]):
                ra, rb = hip_to_row.get(a), hip_to_row.get(b)
                if ra is None or rb is None:
                    dropped += 1
                    missing.update(h for h, r in ((a, ra), (b, rb)) if r is None)
                    continue
                seg.append((ra, rb))
        if len(seg) > start:
            figures.append({"iau": con["iau"],
                            "name": con["common_name"]["english"],
                            "seg": [start, len(seg) - start]})

    print(f"  {len(figures)} figures, {len(seg)} segments"
          f" ({dropped} dropped, stars without distance: {sorted(missing)})")
    return np.asarray(seg, dtype=np.uint32).reshape(-1), figures


def main():
    print("stars:")
    xyz, absmag, ci, hip_to_row, names = load_stars()
    lines, figures = load_lines(hip_to_row)
    rgb = bv_to_rgb(ci)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "positions.bin").write_bytes(xyz.astype("<f4").tobytes())
    (OUT / "absmag.bin").write_bytes(absmag.astype("<f4").tobytes())
    (OUT / "colors.bin").write_bytes(rgb.tobytes())
    (OUT / "lines.bin").write_bytes(lines.astype("<u4").tobytes())

    # Names are the only text here, so they stay JSON rather than becoming a
    # fifth binary. `i` indexes the packed arrays; `seg` is [start, count] into
    # lines.bin. Neither carries a position -- both derive from positions.bin, so
    # there is exactly one copy of every coordinate.
    (OUT / "names.json").write_text(json.dumps({
        "stars": names,
        "constellations": figures
    }, indent=1) + "\n", encoding="utf-8")

    (OUT / "meta.json").write_text(json.dumps({
        "count": int(len(absmag)),
        "segments": int(len(lines) // 2),
        "maxDistPc": MAX_DIST_PC,
        "frame": "ICRS equatorial, +X vernal equinox, +Z north celestial pole, "
                 "origin at the solar-system barycentre",
        "units": "positions in parsecs; absmag is absolute visual magnitude",
        "_source": {
            "stars": "HYG / AT-HYG (hyglike_from_athyg_v40), David Nash, "
                     "astronexus.com, CC BY-SA 4.0. Hipparcos + Yale Bright Star "
                     "+ Gliese, distances updated from Gaia DR3/DR2.",
            "lines": "Sky & Telescope constellation figures via Stellarium's "
                     "western_SnT sky culture (Paul Krizak, Jonathan E. Piskor; "
                     "rework by the Stellarium team), CC BY-SA 2.0. There is no "
                     "official standard for constellation stick figures.",
            "color": "B-V converted to temperature by Ballesteros (2012), then "
                     "to sRGB along the black-body locus, normalised to hue only."
        }
    }, indent=2) + "\n", encoding="utf-8")

    total = sum((OUT / f).stat().st_size
                for f in ("positions.bin", "absmag.bin", "colors.bin",
                          "lines.bin", "names.json"))
    print(f"  wrote {len(absmag)} stars, {len(lines)//2} segments,"
          f" {len(names)} star names, {len(figures)} figure names"
          f" -> {total/1e6:.2f} MB in {OUT}")


if __name__ == "__main__":
    main()
