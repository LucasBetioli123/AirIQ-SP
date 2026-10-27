"""Focos de queimada do GOES-16 ABI L2 FDCF (Fire Detection and Characterization).

Troca em relação ao sugerido (INPE BDQueimadas): o servidor do INPE não está
acessível neste ambiente. O próprio BDQueimadas usa o GOES-16 como um dos
satélites de detecção. O FDCF traz máscara de fogo e potência radiativa (FRP) em
2 km a cada 10 min. Amostramos 8 varreduras por dia (de 3 em 3 h) numa bbox
ampliada (SP e estados vizinhos), porque a fumaça viaja centenas de km.

Categorias aceitas (máscara): 10-14 e 30-34, ou seja, fogo processado, saturado,
contaminado por nuvem e de alta/média probabilidade, com e sem filtro temporal. A
baixa probabilidade (15/35) fica de fora para reduzir falsos positivos.

Saída: data/interim/fires_daily.parquet (date, lat, lon, frp_max, n_det), um
registro por pixel de 2 km com fogo no dia local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.aq.data.goes_common import list_hour, open_remote, scan_start, window_for_bbox
from src.common.config import data_path, load_config

CFG = load_config()
PRODUCT = CFG["goes"]["fire_product"]
BBOX = tuple(CFG["region"]["fire_bbox"])
FIRE_CODES = [10, 11, 12, 13, 14, 30, 31, 32, 33, 34]


def read_scan(key: str) -> pd.DataFrame:
    (r0, r1, c0, c1), lat, lon = window_for_bbox(PRODUCT, BBOX)
    ds = open_remote(key)
    mask = ds["Mask"].isel(y=slice(r0, r1), x=slice(c0, c1)).values
    power = ds["Power"].isel(y=slice(r0, r1), x=slice(c0, c1)).values
    sel = np.isin(mask, FIRE_CODES) & np.isfinite(lat)
    t = scan_start(key).tz_convert(CFG["region"]["tz"]).tz_localize(None)
    return pd.DataFrame({"time": t, "lat": lat[sel].round(3), "lon": lon[sel].round(3),
                         "frp": np.nan_to_num(power[sel], nan=0.0)})


def process_day(date: pd.Timestamp) -> str:
    out = data_path("raw", "goes_fire", f"fire_{date.date()}.parquet")
    if out.exists():
        return "cached"
    frames = []
    for h in CFG["goes"]["fire_hours_utc"]:
        # varreduras cujo dia LOCAL é `date`: UTC de date 03h até date+1 02h
        t = date + pd.Timedelta(hours=h + 3)
        try:
            ks = list_hour(PRODUCT, t.year, t.day_of_year, t.hour)
            if ks:
                frames.append(read_scan(ks[0]))
        except Exception:
            continue
    df = pd.concat(frames) if frames else pd.DataFrame(columns=["time", "lat", "lon", "frp"])
    df["date"] = date
    df = (df.groupby(["date", "lat", "lon"]).agg(frp_max=("frp", "max"), n_det=("frp", "size")).reset_index()
          if len(df) else pd.DataFrame(columns=["date", "lat", "lon", "frp_max", "n_det"]))
    df.to_parquet(out, index=False)
    return str(len(df))


def main(start=None, end=None, workers: int = 12):
    dates = pd.date_range(start or CFG["period"]["start"], end or CFG["period"]["end"], freq="D")
    window_for_bbox(PRODUCT, BBOX)
    with ProcessPoolExecutor(workers) as ex:  # processos: o h5py serializa threads (lock global)
        list(tqdm(ex.map(process_day, dates, chunksize=1), total=len(dates), desc="GOES FDC"))
    files = [data_path("raw", "goes_fire", f"fire_{d.date()}.parquet") for d in dates]
    df = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    df.to_parquet(data_path("interim", "fires_daily.parquet"), index=False)
    print(f"{len(df)} pixel-dias com fogo; mediana por dia = {df.groupby('date').size().median():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    main(a.start, a.end)
