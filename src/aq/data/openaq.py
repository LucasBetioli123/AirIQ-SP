"""PM2.5 medido nas estações da CETESB, via espelho público da OpenAQ no AWS S3.

Por que não o QUALAR direto: o QUALAR exige cadastro/login e limita o download a
1 parâmetro x 1 estação x 1 ano por consulta. A OpenAQ ingeriu a rede da CETESB e
publica o histórico horário em `s3://openaq-data-archive` (acesso anônimo), em
arquivos `records/csv.gz/locationid=<id>/year=<aaaa>/month=<mm>/location-<id>-<aaaammdd>.csv.gz`.

Saídas:
  data/raw/openaq/hourly_<id>.parquet   PM2.5 horário por estação
  data/interim/stations.parquet         metadados (id, nome, lat, lon)
  data/interim/pm25_daily.parquet       média diária (dia local) com >= min_valid_hours

Uso:
  python -m src.aq.data.openaq                 # baixa o período do config
  python -m src.aq.data.openaq --discover      # (re)varre o bucket atrás de estações em SP
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.common.config import data_path, load_config
from src.common.http import s3_list, session

CFG = load_config()
OAQ = CFG["openaq"]
BUCKET = OAQ["bucket_url"]


def _keys_for_location(loc_id: int, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    keys = []
    for year in range(start.year, end.year + 1):
        ks, _ = s3_list(BUCKET, f"records/csv.gz/locationid={loc_id}/year={year}/")
        for k, _size in ks:
            m = re.search(r"-(\d{8})\.csv\.gz$", k)
            if m and start <= pd.Timestamp(m.group(1)) <= end + pd.Timedelta(days=1):
                keys.append(k)
    return keys


def _read_key(key: str) -> pd.DataFrame:
    r = session().get(BUCKET + key, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(gzip.decompress(r.content)))


def download_location(loc_id: int, start: pd.Timestamp, end: pd.Timestamp, workers: int = 24) -> pd.DataFrame | None:
    out = data_path("raw", "openaq", f"hourly_{loc_id}.parquet")
    if out.exists():
        return pd.read_parquet(out)
    keys = _keys_for_location(loc_id, start, end)
    if not keys:
        return None
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(_read_key, keys))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["parameter"] == "pm25"]
    if df.empty:
        # salva vazio para não varrer de novo uma estação que não mede PM2.5
        pd.DataFrame(columns=["location_id"]).to_parquet(out)
        return None
    df = df[["location_id", "sensors_id", "location", "datetime", "lat", "lon", "value"]].copy()
    df.to_parquet(out, index=False)
    return df


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Média diária no dia local. O registro horário da OpenAQ marca o FIM da hora,
    então subtraímos 1 h antes de atribuir o dia (a hora 00:00 pertence ao dia anterior)."""
    tz = CFG["region"]["tz"]
    h = hourly.copy()
    h["ts"] = pd.to_datetime(h["datetime"], utc=True).dt.tz_convert(tz) - pd.Timedelta(hours=1)
    h["date"] = h["ts"].dt.tz_localize(None).dt.normalize()
    v = h["value"]
    h = h[(v >= 0) & (v <= OAQ["max_valid_value"])]
    # duplicatas (mesmo instante reportado 2x) -> mantém a média
    h = h.groupby(["location_id", "date", h["ts"].dt.hour]).agg(value=("value", "mean")).reset_index()
    d = h.groupby(["location_id", "date"]).agg(pm25=("value", "mean"), n_hours=("value", "size"),
                                                pm25_max_h=("value", "max")).reset_index()
    return d[d["n_hours"] >= OAQ["min_valid_hours"]].reset_index(drop=True)


def discover(lat_rng=(-25.5, -19.5), lon_rng=(-53.5, -44.0), max_id: int = 12000) -> pd.DataFrame:
    """Varre os location IDs baixos do bucket (onde está a rede CETESB) e devolve os que caem em SP."""

    def probe(i):
        try:
            ks, _ = s3_list(BUCKET, f"records/csv.gz/locationid={i}/")
            if not ks:
                return None
            row = _read_key(ks[-1][0]).iloc[0]
            if lat_rng[0] < row["lat"] < lat_rng[1] and lon_rng[0] < row["lon"] < lon_rng[1]:
                return dict(location_id=i, location=row["location"], lat=row["lat"], lon=row["lon"])
        except Exception:
            return None

    with ThreadPoolExecutor(64) as ex:
        rows = [r for r in tqdm(ex.map(probe, range(1, max_id)), total=max_id - 1) if r]
    return pd.DataFrame(rows)


def main(start: str | None = None, end: str | None = None, ids: list[int] | None = None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start or CFG["period"]["start"])
    end_ts = pd.Timestamp(end or CFG["period"]["end"])
    ids = ids or OAQ["candidate_location_ids"]
    hourly = []
    for loc in tqdm(ids, desc="OpenAQ/CETESB"):
        df = download_location(loc, start_ts, end_ts)
        if df is not None and len(df):
            hourly.append(df)
    hourly = pd.concat(hourly, ignore_index=True)
    stations = (hourly.groupby("location_id")
                .agg(name=("location", "last"), lat=("lat", "median"), lon=("lon", "median"))
                .reset_index())
    stations["name"] = stations["name"].str.replace(r"-\d+$", "", regex=True)
    daily = to_daily(hourly)
    daily = daily[(daily["date"] >= start_ts) & (daily["date"] <= end_ts)]
    # estações com pouquíssimos dias não ajudam nem a treinar nem a validar
    counts = daily.groupby("location_id").size()
    keep = counts[counts >= 30].index
    daily = daily[daily["location_id"].isin(keep)].reset_index(drop=True)
    stations = stations[stations["location_id"].isin(keep)].reset_index(drop=True)
    stations.to_parquet(data_path("interim", "stations.parquet"), index=False)
    daily.to_parquet(data_path("interim", "pm25_daily.parquet"), index=False)
    print(f"{len(stations)} estações com PM2.5, {len(daily)} estação-dias "
          f"({daily['date'].min().date()} a {daily['date'].max().date()})")
    return daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    if a.discover:
        print(discover().to_string())
    else:
        main(a.start, a.end)
