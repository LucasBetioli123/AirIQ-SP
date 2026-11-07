"""PM2.5 measured at CETESB stations, via OpenAQ's public mirror on AWS S3.

Why not QUALAR (CETESB's portal) directly: it requires registration and limits each
download to 1 parameter x 1 station x 1 year. OpenAQ ingests the CETESB network and
publishes the hourly history in `s3://openaq-data-archive` (anonymous access), as
`records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/location-<id>-<yyyymmdd>.csv.gz`.

Outputs:
  data/raw/openaq/hourly_<id>.parquet   hourly PM2.5 per station
  data/interim/stations.parquet         metadata (id, name, lat, lon)
  data/interim/pm25_daily.parquet       daily mean (local day) with >= min_valid_hours

Usage:
  python -m src.aq.data.openaq                 # download the configured period
  python -m src.aq.data.openaq --discover      # (re)scan the bucket for stations in SP
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
        # save an empty file so a station without PM2.5 is not scanned again
        pd.DataFrame(columns=["location_id"]).to_parquet(out)
        return None
    df = df[["location_id", "sensors_id", "location", "datetime", "lat", "lon", "value"]].copy()
    df.to_parquet(out, index=False)
    return df


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily mean over the local day. OpenAQ's hourly timestamp marks the END of the hour,
    so 1 h is subtracted before assigning the day (00:00 belongs to the previous day)."""
    tz = CFG["region"]["tz"]
    h = hourly.copy()
    h["ts"] = pd.to_datetime(h["datetime"], utc=True).dt.tz_convert(tz) - pd.Timedelta(hours=1)
    h["date"] = h["ts"].dt.tz_localize(None).dt.normalize()
    v = h["value"]
    h = h[(v >= 0) & (v <= OAQ["max_valid_value"])]
    # duplicates (same instant reported twice) -> keep the mean
    h = h.groupby(["location_id", "date", h["ts"].dt.hour]).agg(value=("value", "mean")).reset_index()
    d = h.groupby(["location_id", "date"]).agg(pm25=("value", "mean"), n_hours=("value", "size"),
                                                pm25_max_h=("value", "max")).reset_index()
    return d[d["n_hours"] >= OAQ["min_valid_hours"]].reset_index(drop=True)


def discover(lat_rng=(-25.5, -19.5), lon_rng=(-53.5, -44.0), max_id: int = 12000) -> pd.DataFrame:
    """Scan the bucket's low location IDs (where the CETESB network lives) and return those inside SP."""

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
    # stations with very few days help neither training nor validation
    counts = daily.groupby("location_id").size()
    keep = counts[counts >= 30].index
    daily = daily[daily["location_id"].isin(keep)].reset_index(drop=True)
    stations = stations[stations["location_id"].isin(keep)].reset_index(drop=True)
    stations.to_parquet(data_path("interim", "stations.parquet"), index=False)
    daily.to_parquet(data_path("interim", "pm25_daily.parquet"), index=False)
    print(f"{len(stations)} stations with PM2.5, {len(daily)} station-days "
          f"({daily['date'].min().date()} to {daily['date'].max().date()})")
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
