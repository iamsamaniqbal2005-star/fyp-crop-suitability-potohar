"""Download daily NASA POWER climate for each field centroid (cached as CSV)."""
from pathlib import Path

import pandas as pd
import requests

from parcels import field_centroids

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "power"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M"


def fetch_point(lon, lat, start=2001, end=2024):
    r = requests.get(URL, params={
        "parameters": PARAMS, "community": "AG", "longitude": round(lon, 4), "latitude": round(lat, 4),
        "start": f"{start}0101", "end": f"{end}1231", "format": "JSON"}, timeout=180)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, float("nan"))
    df.index.name = "date"
    return df


def fetch_all(start=2001, end=2024, force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for fid, (lon, lat) in field_centroids().items():
        path = RAW / f"{fid}.csv"
        if path.exists() and not force:
            out[fid] = pd.read_csv(path, index_col="date", parse_dates=True)
            continue
        print(f"NASA POWER {fid} ({lat:.3f},{lon:.3f})")
        out[fid] = fetch_point(lon, lat, start, end)
        out[fid].to_csv(path)
    return out


if __name__ == "__main__":
    fetch_all()
