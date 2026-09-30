"""Soil properties per field from ISRIC SoilGrids (REST). Cached to data/raw/soil.csv.

If the API is unavailable, regional defaults for Potohar loess/loam soils are used and flagged
(`source = default`) so the gap is visible downstream.
"""
import time
from pathlib import Path

import pandas as pd
import requests

from parcels import field_centroids

PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "soil.csv"
URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ["phh2o", "clay", "sand", "silt", "soc"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]
DEFAULT = {"ph": 7.3, "clay": 25.0, "sand": 40.0, "silt": 35.0, "soc": 0.8, "source": "default"}


def _query(lon, lat):
    r = requests.get(URL, params=[("lon", lon), ("lat", lat)] + [("property", p) for p in PROPS]
                     + [("depth", d) for d in DEPTHS] + [("value", "mean")], timeout=90)
    r.raise_for_status()
    out = {}
    for layer in r.json()["properties"]["layers"]:
        vals = [d["values"]["mean"] for d in layer["depths"] if d["values"]["mean"] is not None]
        if vals:
            out[layer["name"]] = sum(vals) / len(vals)
    return {"ph": out["phh2o"] / 10, "clay": out["clay"] / 10, "sand": out["sand"] / 10,
            "silt": out["silt"] / 10, "soc": out["soc"] / 100, "source": "soilgrids"}  # %, %, %, dg/kg->%


def texture_class(clay, sand):
    if clay >= 35:
        return "heavy"
    if sand >= 55 and clay < 20:
        return "light"
    return "medium"


def fetch_all(force=False):
    if PATH.exists() and not force:
        return pd.read_csv(PATH)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid, (lon, lat) in field_centroids().items():
        rec = None
        for attempt in range(3):
            try:
                rec = _query(lon, lat)
                break
            except Exception as ex:
                print(f"SoilGrids {fid} attempt {attempt + 1} failed: {ex}")
                time.sleep(15)
        rows.append({"field_id": fid, **(rec or DEFAULT)})
        time.sleep(13)  # public API is rate limited
    df = pd.DataFrame(rows)
    df["texture"] = [texture_class(c, s) for c, s in zip(df.clay, df.sand)]
    df.to_csv(PATH, index=False)
    return df


if __name__ == "__main__":
    print(fetch_all())
