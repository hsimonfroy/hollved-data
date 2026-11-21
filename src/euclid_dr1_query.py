"""Download the Euclid DR1 Wide spectroscopic redshift sample from the ESA archive TAP.

Selection: rank-0 galaxy solution with spe_z_prob >= 0.999, Halpha flux >= 2e-16 erg/s/cm2
and S/N >= 7, extended (point_like_prob < 0.05) and non-spurious sources.

Anonymous TAP jobs are capped at 60 s and 50,000 rows, and the selected columns are not indexed,
so the sky is split into RA strips through object_id, which encodes position:
|object_id| = RA * 1e16 + |Dec| * 1e7, with the sign of Dec. A strip that times out or hits
the row cap is split in two. Each strip is saved, so an interrupted run resumes where it stopped.
"""
import sys
import io
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from utils import RAW

TAP = 'https://eas.esac.esa.int/tap-server/tap/sync'
MAXROWS = 50_000
UNIT = 10**7          # object_id RA unit per degree
STEP = 1.0            # initial strip width in degrees
MIN_STEP = 0.005
N_WORKERS = 4
PARTS = RAW / 'euclid_dr1_parts'
OUT = RAW / 'Euclid_DR1.csv.gz'

QUERY = """SELECT z.object_id, m.right_ascension, m.declination, z.spe_z, z.spe_z_err, z.spe_z_prob,
 l.spe_line_flux_gf, l.spe_line_snr_gf
FROM dr1.spectro_z_spe_galaxy_candidates_wide_survey AS z
JOIN dr1.spectro_line_features_spe_line_features_wide_survey AS l
 ON z.object_id = l.object_id AND z.spe_rank = l.spe_rank
JOIN dr1.mer_catalogue_wide_survey AS m ON z.object_id = m.object_id
WHERE z.spe_rank = 0 AND z.spe_z_prob >= 0.999
 AND l.spe_line_name = 'Halpha' AND l.spe_line_flux_gf >= 2e-16 AND l.spe_line_snr_gf >= 7
 AND m.point_like_prob < 0.05 AND m.spurious_prob < 0.5
 AND z.object_id BETWEEN {lo} AND {hi}"""


def id_range(sign, a, b):
    """object_id bounds for RA in [a, b) (units of 1e-7 deg) and the given Dec sign."""
    lo, hi = a * 10**9, b * 10**9 - 1
    return (lo, hi) if sign > 0 else (-hi, -lo)


def query(sign, a, b):
    lo, hi = id_range(sign, a, b)
    data = urllib.parse.urlencode({'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                                   'QUERY': QUERY.format(lo=lo, hi=hi)}).encode()
    try:
        with urllib.request.urlopen(TAP, data=data, timeout=120) as r:
            return pd.read_csv(io.BytesIO(r.read()))
    except urllib.error.HTTPError as e:
        if e.code == 408:  # server-side timeout
            return None
        raise


def fetch(sign, a, b):
    """Fetch one strip, splitting it until every piece fits the limits. Returns rows saved."""
    part = PARTS / f"{'n' if sign > 0 else 's'}_{a:010d}_{b:010d}.csv"
    if part.exists():
        return len(pd.read_csv(part))
    df = None
    for attempt in range(3):
        try:
            df = query(sign, a, b)
            break
        except Exception as e:
            print(f"  retry {part.name}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"failed {part.name}")
    if df is None or len(df) >= MAXROWS:
        if b - a <= MIN_STEP * UNIT:
            raise RuntimeError(f"cannot split {part.name} further")
        mid = (a + b) // 2
        return fetch(sign, a, mid) + fetch(sign, mid, b)
    df.to_csv(part, index=False)
    return len(df)


def main():
    PARTS.mkdir(exist_ok=True)
    edges = [round(i * STEP * UNIT) for i in range(int(360 / STEP) + 1)]
    tasks = [(s, a, b) for s in (-1, 1) for a, b in zip(edges[:-1], edges[1:])]
    t0, total = time.time(), 0
    with ThreadPoolExecutor(N_WORKERS) as ex:
        for i, n in enumerate(ex.map(lambda t: fetch(*t), tasks)):
            total += n
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(tasks)} strips, {total:,} rows, {time.time() - t0:.0f} s", flush=True)
    df = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob('*.csv'))], ignore_index=True)
    n = len(df)
    df = df.drop_duplicates('object_id')
    df.to_csv(OUT, index=False)
    print(f"Done: {len(df):,} rows ({n - len(df)} duplicates dropped) in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == '__main__':
    main()
