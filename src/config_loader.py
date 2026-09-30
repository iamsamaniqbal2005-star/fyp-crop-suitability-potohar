"""Load and validate the study-area and EcoCrop configuration files."""
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

REQUIRED_GROUPS = {
    "temperature": ["tmin", "topmin", "topmax", "tmax"],
    "rainfall": ["rmin", "ropmin", "ropmax", "rmax"],
    "cycle": ["gmin", "gmax"],
    "ph": ["ph_absmin", "ph_optmin", "ph_optmax", "ph_absmax"],
}
EXPECTED_CROPS = {"wheat", "maize", "groundnut", "mustard", "chickpea", "olive"}


def _load(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_study_area():
    cfg = _load("study_area.json")
    w, s, e, n = cfg["region"]["bbox"]
    assert w < e and s < n, "region bbox must be [west, south, east, north]"
    for field in cfg["fields"]:
        fw, fs, fe, fn = field["bbox"]
        assert fw < fe and fs < fn, f"bad bbox in {field['id']}"
        assert w <= fw and fe <= e and s <= fs and fn <= n, f"{field['id']} outside region bbox"
    return cfg


def load_ecocrop():
    cfg = _load("ecocrop.json")
    crops = cfg["crops"]
    missing = EXPECTED_CROPS - set(crops)
    assert not missing, f"missing crops: {missing}"
    for name, c in crops.items():
        for group, keys in REQUIRED_GROUPS.items():
            for k in keys:
                assert k in c[group], f"{name}.{group}.{k} missing"
        t, r, p = c["temperature"], c["rainfall"], c["ph"]
        assert t["tmin"] <= t["topmin"] <= t["topmax"] <= t["tmax"], f"{name}: temperature order"
        assert r["rmin"] <= r["ropmin"] <= r["ropmax"] <= r["rmax"], f"{name}: rainfall order"
        assert p["ph_absmin"] <= p["ph_optmin"] <= p["ph_optmax"] <= p["ph_absmax"], f"{name}: pH order"
    return cfg


if __name__ == "__main__":
    sa, eco = load_study_area(), load_ecocrop()
    print(f"OK: {len(sa['fields'])} fields, {len(eco['crops'])} crops validated")
