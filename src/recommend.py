"""Rank crops per parcel (rule filter -> ML score) and export CSV / GeoJSON / maps."""
import json
from pathlib import Path

import folium
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ecocrop_model as eco
from features import CLIMATE_COLS, LAND_COLS, NDVI_COLS, WATER_COLS
from parcels import build_parcels
from train import design_matrix

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
NORMAL_FROM_YEAR = 2014   # recent-climate "normal" used for the recommendation
COLORS = {"wheat": "#d9a520", "maize": "#6aa84f", "groundnut": "#a0522d", "mustard": "#e3d800",
          "chickpea": "#8e7cc3", "olive": "#3d5a3d"}


def recommend(df, reg_model="xgboost", top_n=3):
    OUTPUTS.mkdir(exist_ok=True)
    num = CLIMATE_COLS + LAND_COLS + WATER_COLS + NDVI_COLS + ["score"]
    cur = df[df.year >= NORMAL_FROM_YEAR].groupby(["parcel_id", "field_id", "crop"], as_index=False)[num].mean()
    bundle = joblib.load(ROOT / "models" / f"reg_{reg_model}.joblib")
    X = design_matrix(cur, CLIMATE_COLS + LAND_COLS + WATER_COLS + NDVI_COLS).reindex(columns=bundle["columns"], fill_value=0.0)
    cur["ml_score"] = np.clip(bundle["model"].predict(X), 0, 100)
    cur["rule_score"] = cur["score"]

    rows = []
    for pid, g in cur.groupby("parcel_id"):
        kept = eco.rule_filter(dict(zip(g.crop, g.rule_score)))            # Layer 4a: rule filter
        g = g[g.crop.isin(kept)].sort_values("ml_score", ascending=False)  # Layer 4b: ML ranking
        for rank, (_, r) in enumerate(g.head(top_n).iterrows(), 1):
            rows.append({"parcel_id": pid, "field_id": r.field_id, "rank": rank, "crop": r.crop,
                         "ml_score": round(r.ml_score, 1), "rule_score": round(r.rule_score, 1),
                         "class": eco.score_to_class(r.ml_score)})
    ranked = pd.DataFrame(rows)
    ranked.to_csv(OUTPUTS / "ranked_crops.csv", index=False)
    cur[["parcel_id", "crop", "ml_score", "rule_score"]].to_csv(OUTPUTS / "all_crop_scores.csv", index=False)
    return ranked, cur


def make_maps(ranked, cur):
    parcels = build_parcels()
    top = ranked[ranked["rank"] == 1].set_index("parcel_id")
    feats = []
    for _, p in parcels.iterrows():
        scores = cur[cur.parcel_id == p.parcel_id].set_index("crop").ml_score.round(1).to_dict()
        props = {"parcel_id": p.parcel_id, "field_id": p.field_id, "scores": scores}
        if p.parcel_id in top.index:
            props.update(top_crop=top.loc[p.parcel_id, "crop"], top_score=float(top.loc[p.parcel_id, "ml_score"]))
        ring = [[p.west, p.south], [p.east, p.south], [p.east, p.north], [p.west, p.north], [p.west, p.south]]
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    gj = {"type": "FeatureCollection", "features": feats}
    (OUTPUTS / "suitability.geojson").write_text(json.dumps(gj), encoding="utf-8")

    m = folium.Map(location=[parcels.lat.mean(), parcels.lon.mean()], zoom_start=9, tiles="CartoDB positron")
    folium.GeoJson(gj, style_function=lambda f: {
        "color": "#333", "weight": 1, "fillOpacity": 0.6,
        "fillColor": COLORS.get(f["properties"].get("top_crop"), "#bbbbbb")},
        tooltip=folium.GeoJsonTooltip(["parcel_id", "top_crop", "top_score", "scores"])).add_to(m)
    m.save(str(OUTPUTS / "suitability_map.html"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, crop in zip(axes.ravel(), eco.CROPS):
        d = cur[cur.crop == crop].merge(parcels, on="parcel_id")
        sc = ax.scatter(d.lon, d.lat, c=d.ml_score, cmap="RdYlGn", vmin=0, vmax=100, s=260, marker="s")
        ax.set_title(f"{crop.title()} suitability (0-100)"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(sc, ax=axes, shrink=0.8, label="ML suitability score")
    fig.savefig(OUTPUTS / "suitability_by_crop.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "land_profiles.csv")
    r, c = recommend(df)
    make_maps(r, c)
    print(r[r["rank"] == 1].groupby(["field_id", "crop"]).size())
