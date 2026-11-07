"""CLI: PM2.5 estimate/forecast for a coordinate and date.

  python -m src.aq.cli --lat -23.50 --lon -47.45 --date 2021-08-24            # same-day estimate
  python -m src.aq.cli --municipio Sorocaba --date 2021-08-23 --horizon 1     # next-day (D+1) forecast
"""
from __future__ import annotations

import argparse
import json
import sys

from src.aq.data.geo import geocode
from src.aq.predict import DataUnavailable, predict_one


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--municipio")
    ap.add_argument("--date", required=True, help="reference date (yyyy-mm-dd)")
    ap.add_argument("--horizon", type=int, default=0, choices=[0, 1], help="0 = same-day estimate, 1 = next-day (D+1) forecast")
    a = ap.parse_args(argv)
    if a.municipio:
        g = geocode(a.municipio)
        if g is None:
            sys.exit(f"municipality not found in São Paulo state: {a.municipio}")
        a.lat, a.lon = g["lat"], g["lon"]
    if a.lat is None or a.lon is None:
        sys.exit("provide --lat/--lon or --municipio")
    try:
        print(json.dumps(predict_one(a.lat, a.lon, a.date, a.horizon), ensure_ascii=False, indent=2))
    except DataUnavailable as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
