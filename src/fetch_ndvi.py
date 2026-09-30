"""Seasonal NDVI per parcel from Sentinel-2 L2A (Microsoft Planetary Computer STAC, no account needed).

Used as the public-satellite fallback described in the project document (section 8.3); archived
government UAV orthomosaics can be dropped in as data/external/ndvi_uav.csv with the same columns.
Output: data/raw/ndvi.csv  [parcel_id, year, season, ndvi_mean, ndvi_std, ndvi_min, ndvi_max, n_scenes]
"""
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from config_loader import load_study_area
from parcels import build_parcels, N_SPLIT

PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ndvi.csv"
# peak-greenness windows: rabi (wheat/mustard/chickpea) and kharif (maize/groundnut)
SEASONS = {"rabi": ((1, 15), (3, 15), 0), "kharif": ((8, 1), (9, 15), 0)}
MAX_SCENES = 3


def _read_band(href, bbox4326):
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox4326)
        arr = src.read(1, window=from_bounds(*b, transform=src.transform), boundless=False).astype("float32")
    return arr


def _ndvi_scene(item, bbox):
    red = _read_band(item.assets["B04"].href, bbox)
    nir = _read_band(item.assets["B08"].href, bbox)
    scl = _read_band(item.assets["SCL"].href, bbox)
    h, w = min(red.shape[0], nir.shape[0]), min(red.shape[1], nir.shape[1])
    red, nir = red[:h, :w], nir[:h, :w]
    if scl.shape != (h, w):  # SCL is 20 m: nearest-neighbour resample to the 10 m grid
        scl = scl[(np.arange(h) * scl.shape[0] // h)[:, None], (np.arange(w) * scl.shape[1] // w)[None, :]]
    ndvi = (nir - red) / (nir + red + 1e-6)
    bad = np.isin(scl, [0, 1, 3, 8, 9, 10, 11]) | (red == 0) | (nir == 0)  # nodata, shadow, cloud, snow
    ndvi[bad] = np.nan
    return ndvi


def fetch_all(years=range(2017, 2025), force=False):
    if PATH.exists() and not force:
        return pd.read_csv(PATH)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    parcels = build_parcels()
    rows = []
    for f in load_study_area()["fields"]:
        bbox = f["bbox"]
        fp = parcels[parcels.field_id == f["id"]]
        for y in years:
            for season, ((m1, d1), (m2, d2), _) in SEASONS.items():
                rng = f"{y}-{m1:02d}-{d1:02d}/{y}-{m2:02d}-{d2:02d}"
                try:
                    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=rng,
                                            query={"eo:cloud_cover": {"lt": 25}}).items())
                    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:MAX_SCENES]
                    stack = np.stack([_ndvi_scene(i, bbox) for i in items]) if items else None
                except Exception as ex:
                    print(f"  {f['id']} {y} {season}: failed ({type(ex).__name__}: {ex})")
                    continue
                if stack is None:
                    continue
                h = min(s.shape[0] for s in stack); w = min(s.shape[1] for s in stack)
                comp = np.nanmedian(stack[:, :h, :w], axis=0)
                for _, p in fp.iterrows():
                    # parcel (col, row): row counts south->north, image rows run north->south
                    r0, r1 = int((N_SPLIT - 1 - p.row) * h / N_SPLIT), int((N_SPLIT - p.row) * h / N_SPLIT)
                    c0, c1 = int(p.col * w / N_SPLIT), int((p.col + 1) * w / N_SPLIT)
                    sub = comp[r0:r1, c0:c1]
                    if np.isfinite(sub).sum() < 50:
                        continue
                    rows.append({"parcel_id": p.parcel_id, "year": y, "season": season,
                                 "ndvi_mean": np.nanmean(sub), "ndvi_std": np.nanstd(sub),
                                 "ndvi_min": np.nanmin(sub), "ndvi_max": np.nanmax(sub), "n_scenes": len(items)})
                print(f"{f['id']} {y} {season}: {len(items)} scenes")
    df = pd.DataFrame(rows)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH, index=False)
    return df


if __name__ == "__main__":
    fetch_all()
