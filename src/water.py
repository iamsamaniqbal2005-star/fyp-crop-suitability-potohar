"""Water pillar features (IRSA / WAPDA / PCRWR / Punjab Irrigation).

These sources publish PDFs / HTML tables rather than an API, so the pipeline reads a manually
maintained CSV: data/external/water_indices.csv  (one row per field x season; year optional).

    field_id, season, irrigation_mm, supply_index, shortage_pct, salinity_risk, waterlogging_risk, source

- irrigation_mm      : seasonal irrigation depth reliably available (0 = purely rainfed / barani)
- supply_index       : 0-1 seasonal canal/river supply index (IRSA rim-station inflows, WAPDA flows)
- shortage_pct       : IRSA reported shortage %
- salinity_risk      : 0-1 from PCRWR reports/expert rules
- waterlogging_risk  : 0-1 from PCRWR reports/expert rules

A template with clearly flagged assumed Potohar defaults (barani, no canal command) is written on the
first run. Replace values with extracted data as it is collected.
"""
from pathlib import Path

import pandas as pd

from config_loader import load_study_area

PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "water_indices.csv"
COLS = ["irrigation_mm", "supply_index", "shortage_pct", "salinity_risk", "waterlogging_risk"]


def write_template():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"field_id": f["id"], "season": s, "irrigation_mm": 0, "supply_index": 0.0, "shortage_pct": 0.0,
             "salinity_risk": 0.1, "waterlogging_risk": 0.05, "source": "ASSUMED barani default - replace"}
            for f in load_study_area()["fields"] for s in ("rabi", "kharif", "perennial")]
    pd.DataFrame(rows).to_csv(PATH, index=False)


def load_water():
    if not PATH.exists():
        write_template()
    return pd.read_csv(PATH)


if __name__ == "__main__":
    print(load_water())
