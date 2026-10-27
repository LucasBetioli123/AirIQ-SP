"""CLI: estimativa/previsão de PM2.5 para uma coordenada e data.

  python -m src.aq.cli --lat -22.90 --lon -47.06 --date 2023-03-10            # estimativa do dia
  python -m src.aq.cli --municipio Campinas --date 2023-03-10 --horizon 1     # previsão D+1
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
    ap.add_argument("--date", required=True, help="data de referência (aaaa-mm-dd)")
    ap.add_argument("--horizon", type=int, default=0, choices=[0, 1], help="0 = estimativa do dia, 1 = previsão D+1")
    a = ap.parse_args(argv)
    if a.municipio:
        g = geocode(a.municipio)
        if g is None:
            sys.exit(f"município não encontrado em SP: {a.municipio}")
        a.lat, a.lon = g["lat"], g["lon"]
    if a.lat is None or a.lon is None:
        sys.exit("informe --lat/--lon ou --municipio")
    try:
        print(json.dumps(predict_one(a.lat, a.lon, a.date, a.horizon), ensure_ascii=False, indent=2))
    except DataUnavailable as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
