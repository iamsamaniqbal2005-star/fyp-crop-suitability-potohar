"""Rule layer: FAO EcoCrop-style suitability scoring (0-100) and the pre-ML crop filter."""
import numpy as np

from config_loader import load_ecocrop

CROPS = load_ecocrop()["crops"]
SUITABLE, MODERATE = 60.0, 30.0


def trapezoid(x, amin, opmin, opmax, amax):
    """1 inside the optimal range, linear fall to 0 at the absolute limits."""
    x = np.asarray(x, dtype=float)
    up = np.clip((x - amin) / max(opmin - amin, 1e-9), 0, 1)
    down = np.clip((amax - x) / max(amax - opmax, 1e-9), 0, 1)
    return np.minimum(up, down)


def temperature_score(crop, tmean_monthly, frost_days):
    t = CROPS[crop]["temperature"]
    s = trapezoid(tmean_monthly, t["tmin"], t["topmin"], t["topmax"], t["tmax"])
    score = float(np.min(s)) * 0.5 + float(np.mean(s)) * 0.5   # limiting month + average month
    if t.get("ktmp") is not None and CROPS[crop]["life_cycle"] == "annual":
        score *= 1.0 if frost_days == 0 else max(0.0, 1 - frost_days / 15)
    return score


def rainfall_score(crop, water_mm):
    r = CROPS[crop]["rainfall"]
    return float(trapezoid(water_mm, r["rmin"], r["ropmin"], r["ropmax"], r["rmax"]))


def soil_score(crop, ph, texture):
    p = CROPS[crop]["ph"]
    s = float(trapezoid(ph, p["ph_absmin"], p["ph_optmin"], p["ph_optmax"], p["ph_absmax"]))
    return s * (1.0 if texture in CROPS[crop]["soil"]["texture"] else 0.7)


def ecocrop_score(crop, tmean_monthly, frost_days, water_mm, ph, texture, salinity_risk=0.0, waterlog_risk=0.0):
    """Limiting-factor (Liebig) combination of temperature, water and soil, with water-quality penalties."""
    parts = (temperature_score(crop, tmean_monthly, frost_days), rainfall_score(crop, water_mm),
             soil_score(crop, ph, texture))
    base = 100 * min(parts) * 0.7 + 100 * float(np.mean(parts)) * 0.3
    return base * (1 - 0.5 * salinity_risk) * (1 - 0.3 * waterlog_risk)


def score_to_class(score):
    return "Suitable" if score >= SUITABLE else "Moderately suitable" if score >= MODERATE else "Unsuitable"


def rule_filter(scores_by_crop, floor=MODERATE / 2):
    """Crops scoring below `floor` are clearly impossible and dropped before ML ranking."""
    return {c: s for c, s in scores_by_crop.items() if s >= floor}
