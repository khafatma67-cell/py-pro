"""App logic kept separate from the UI so it can be tested without Streamlit."""
import pandas as pd

# Reasonable defaults for any field the user does not fill in.
DEFAULTS = {
    "age": 17, "Medu": 2, "Fedu": 2, "traveltime": 1, "studytime": 2, "failures": 0,
    "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1, "health": 3,
    "absences": 4, "sex": "F", "address": "U", "famsup": "yes", "schoolsup": "no",
    "paid": "no", "activities": "no", "higher": "yes", "internet": "yes",
    "romantic": "no",
}


def build_row(values: dict, model) -> pd.DataFrame:
    """Build the one-row DataFrame the model expects.

    Reads the column list from the trained model itself, so the app works with
    both the full model (22 features) and a smaller one (e.g. 6 features).
    """
    data = {**DEFAULTS, **values}
    cols = getattr(model, "feature_names_in_", None)
    cols = list(cols) if cols is not None else list(data)
    return pd.DataFrame([{c: data[c] for c in cols}])


def verdict(p_pass: float) -> str:
    if p_pass >= 0.75:
        return "pass"
    if p_pass >= 0.5:
        return "border"
    return "fail"


def get_tips(values: dict) -> list:
    """Rule-based advice keys (translated in i18n.py)."""
    tips = []
    if values.get("absences", 0) >= 10:
        tips.append("tip_absences")
    if values.get("failures", 0) >= 1:
        tips.append("tip_failures")
    if values.get("studytime", 2) <= 1:
        tips.append("tip_study")
    if values.get("higher") == "no":
        tips.append("tip_higher")
    if values.get("internet") == "no":
        tips.append("tip_internet")
    return tips or ["tip_ok"]