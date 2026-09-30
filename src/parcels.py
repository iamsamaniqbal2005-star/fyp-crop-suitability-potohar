"""Split each target field into a regular grid of parcels (the unit of assessment)."""
import pandas as pd

from config_loader import load_study_area

N_SPLIT = 3  # N_SPLIT x N_SPLIT parcels per field


def build_parcels(n=N_SPLIT):
    rows = []
    for f in load_study_area()["fields"]:
        w, s, e, nn = f["bbox"]
        dx, dy = (e - w) / n, (nn - s) / n
        for i in range(n):          # i: column (west->east), j: row (south->north)
            for j in range(n):
                pw, ps = w + i * dx, s + j * dy
                rows.append({
                    "parcel_id": f"{f['id']}_{i}{j}", "field_id": f["id"], "district": f["district"],
                    "west": pw, "south": ps, "east": pw + dx, "north": ps + dy,
                    "lon": pw + dx / 2, "lat": ps + dy / 2, "col": i, "row": j,
                })
    return pd.DataFrame(rows)


def field_centroids():
    return {f["id"]: ((f["bbox"][0] + f["bbox"][2]) / 2, (f["bbox"][1] + f["bbox"][3]) / 2)
            for f in load_study_area()["fields"]}
