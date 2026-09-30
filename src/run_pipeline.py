"""End-to-end pipeline: python src/run_pipeline.py [--refresh]   (--refresh re-downloads NDVI/soil/climate)"""
import sys
from pathlib import Path

import pandas as pd

import features
import fetch_ndvi
import fetch_power
import fetch_soil
import recommend
import train

if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    fetch_power.fetch_all(force=refresh)
    fetch_soil.fetch_all(force=refresh)
    ndvi = fetch_ndvi.fetch_all(force=refresh)                 # Sentinel-2 fallback (or UAV csv, see module doc)
    df = features.build_dataset(ndvi)                          # land-profile vectors + FAO-rule labels
    report = train.run(df)                                     # RF / SVM / DNN / LASSO / XGB
    ranked, cur = recommend.recommend(df)                      # rule filter -> ML rank
    recommend.make_maps(ranked, cur)
    print("best regression R2:", max(v["r2"] for v in report["regression"].values()))
    print(ranked[ranked["rank"] == 1].crop.value_counts().to_string())
