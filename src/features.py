"""Fuse climate + soil + water + NDVI into the land-profile table (one row per parcel x sowing-year x crop)."""
from pathlib import Path

import numpy as np
import pandas as pd

import ecocrop_model as eco
from fetch_ndvi import PATH as NDVI_PATH
from fetch_power import fetch_all as fetch_power
from fetch_soil import fetch_all as fetch_soil
from parcels import build_parcels
from water import load_water

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
CLIMATE_COLS = ["tmean", "tmin_mean", "tmax_mean", "gdd", "frost_days", "heat_days", "precip_total",
                "rain_days", "rh_mean", "solar_mean", "wind_mean"]
LAND_COLS = ["ph", "clay", "sand", "silt", "soc"]
WATER_COLS = ["irrigation_mm", "supply_index", "shortage_pct", "salinity_risk", "waterlogging_risk"]
NDVI_COLS = ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max"]


def growing_window(crop, year):
    s = eco.CROPS[crop]["sowing"]
    start = pd.Timestamp(year=year, month=s["month"], day=s["day"])
    return start, start + pd.Timedelta(days=s["cycle_days"] - 1)


def climate_features(clim, crop, year):
    a, b = growing_window(crop, year)
    w = clim.loc[a:b]
    if len(w) < 0.95 * ((b - a).days + 1):
        return None
    t = eco.CROPS[crop]["temperature"]
    monthly = w["T2M"].resample("MS").mean().values
    frost_thr = t["ktmp"] if t.get("ktmp") is not None else 0
    return {
        "tmean": w.T2M.mean(), "tmin_mean": w.T2M_MIN.mean(), "tmax_mean": w.T2M_MAX.mean(),
        "gdd": np.clip(w.T2M - t["tmin"], 0, t["tmax"] - t["tmin"]).sum(),
        "frost_days": int((w.T2M_MIN < frost_thr).sum()), "heat_days": int((w.T2M_MAX > t["tmax"]).sum()),
        "precip_total": w.PRECTOTCORR.sum(), "rain_days": int((w.PRECTOTCORR >= 1).sum()),
        "rh_mean": w.RH2M.mean(), "solar_mean": w.ALLSKY_SFC_SW_DWN.mean(), "wind_mean": w.WS2M.mean(),
        "_monthly_t": monthly,
    }


def build_dataset(ndvi=None):
    parcels = build_parcels()
    power, soil = fetch_power(), fetch_soil().set_index("field_id")
    water = load_water()
    if ndvi is None and NDVI_PATH.exists():
        ndvi = pd.read_csv(NDVI_PATH)
    ndvi_idx = ndvi.set_index(["parcel_id", "year", "season"]) if ndvi is not None and len(ndvi) else None
    rows = []
    for _, p in parcels.iterrows():
        clim, sl = power[p.field_id], soil.loc[p.field_id]
        for crop, spec in eco.CROPS.items():
            season = spec["sowing"]["season"]
            wrow = water[(water.field_id == p.field_id) & (water.season == season)].iloc[0]
            for year in range(2001, 2024):
                cf = climate_features(clim, crop, year)
                if cf is None:
                    continue
                monthly = cf.pop("_monthly_t")
                if season == "perennial":
                    nd_keys = [(p.parcel_id, year, "rabi"), (p.parcel_id, year, "kharif")]
                else:  # rabi crop sown in autumn is observed (peak NDVI) the following calendar year
                    nd_keys = [(p.parcel_id, year + (1 if season == "rabi" else 0), season)]
                nd = [ndvi_idx.loc[k] for k in nd_keys if ndvi_idx is not None and k in ndvi_idx.index]
                ndvi_f = {c: float(np.mean([r[c] for r in nd])) for c in NDVI_COLS} if nd else {c: np.nan for c in NDVI_COLS}
                water_mm = cf["precip_total"] + wrow.irrigation_mm * (1 - wrow.shortage_pct / 100)
                score = eco.ecocrop_score(crop, monthly, cf["frost_days"], water_mm, sl.ph, sl.texture,
                                          wrow.salinity_risk, wrow.waterlogging_risk)
                rows.append({"parcel_id": p.parcel_id, "field_id": p.field_id, "year": year, "crop": crop,
                             **cf, **{c: sl[c] for c in LAND_COLS}, **{c: wrow[c] for c in WATER_COLS},
                             **ndvi_f, "score": score, "label": eco.score_to_class(score)})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "land_profiles.csv", index=False)
    return df


if __name__ == "__main__":
    d = build_dataset()
    print(d.shape)
    print(d.groupby("crop")[["score", "precip_total", "tmean"]].mean().round(1))
    print(d.label.value_counts())
