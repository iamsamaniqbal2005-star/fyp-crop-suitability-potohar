# FYP: Crop Suitability Mapping for the Potohar Region (Rawalpindi/Islamabad)

Crops: Wheat, Maize, Groundnut, Mustard, Chickpea, Olive (FAO EcoCrop requirements).

## Layout
- `config/study_area.json` – region + target field bounding boxes (`[west, south, east, north]`, EPSG:4326)
- `config/ecocrop.json` – EcoCrop temperature / rainfall / cycle / pH ranges per crop
- `src/config_loader.py` – loads and validates both configs (`python src/config_loader.py`)
- `notebooks/00_setup.ipynb` – Colab/Kaggle bootstrap (clone repo, install deps, validate config)
- `data/` – local data (git-ignored)

## Setup
Local: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
Colab/Kaggle: open `notebooks/00_setup.ipynb` and set `REPO_URL` in the first cell.

## Status
- [x] Phase 1.2 configs drafted
- [ ] EcoCrop values verified against official FAO records (`verified: false` in JSON)
- [ ] Real field bounding boxes replace placeholders
- [ ] Repo pushed to GitHub/GitLab; teammates invited
