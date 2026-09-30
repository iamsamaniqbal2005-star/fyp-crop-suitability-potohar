# Intelligent Crop Suitability Assessment – Potohar (Rawalpindi/Islamabad)

Fuses climate (NASA POWER), soil (SoilGrids), water (IRSA/WAPDA/PCRWR table) and Sentinel-2 NDVI into a
land profile per parcel, filters crops with FAO EcoCrop rules, then ranks them with ML.
Crops: Wheat, Maize, Groundnut, Mustard, Chickpea, Olive.

## Pipeline (`python src/run_pipeline.py`)
| Layer | Module | Output |
|---|---|---|
| 1 Acquisition | `fetch_power.py`, `fetch_soil.py`, `fetch_ndvi.py`, `water.py` | `data/raw/*`, `data/external/water_indices.csv` |
| 2-3 Features / land profile | `parcels.py`, `features.py` | `data/processed/land_profiles.csv` |
| 4a Rule filter | `ecocrop_model.py` (EcoCrop trapezoids, Liebig minimum) | 0-100 score + class |
| 4b ML | `train.py` – RF/SVM/DNN (class), LASSO/XGBoost/RF (score) | `models/`, `outputs/metrics.json` |
| 5 Decision support | `recommend.py` | `outputs/ranked_crops.csv`, `suitability.geojson`, `suitability_map.html`, `suitability_by_crop.png` |

Config: `config/study_area.json` (field boxes, split into 3x3 parcels), `config/ecocrop.json` (crop requirements + sowing windows).

## Setup
`pip install -r requirements.txt`, then `python src/run_pipeline.py` (first run downloads data; later runs use the cache, `--refresh` re-downloads).

## Known limitations (read before quoting results)
- **Labels are rule-derived** (FAO EcoCrop scores), as the project plan prescribes. ML metrics therefore measure how well models reproduce the rules from the fused features, **not** agronomic truth. Real validation needs field/agronomist labels or historical crop maps.
- Climate is NASA POWER (~50 km grid) so parcels within one field share climate; parcels differ only by NDVI.
- **Water table is an assumed barani (rainfed) default** – fill `data/external/water_indices.csv` from IRSA/WAPDA/PCRWR. (Those are Indus-basin/canal sources; Potohar is mostly rainfed, so their relevance here is limited.)
- EcoCrop values in `config/ecocrop.json` are unverified (`verified: false`); field boxes are placeholders.
- The document's example crops (rice, cotton, sugarcane) are Sindh/canal crops; this repo uses the six Potohar crops chosen in Phase 1.
- Government UAV imagery is not included; to use it, supply `data/external/ndvi_uav.csv` (same columns as `data/raw/ndvi.csv`) – hook not yet wired.
