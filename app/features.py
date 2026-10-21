"""Feature engineering shared by the notebooks and the Streamlit app.

Keeping these functions in one importable module means the app computes
features exactly the way the models were trained, and lets pickled
scikit-learn pipelines refer to them (a function defined inside a notebook
cannot be loaded back by the app).
"""
import io

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hydraulic system (clustering track)
# ---------------------------------------------------------------------------

# sensor -> (description, unit, sampling rate in Hz); one cycle lasts 60 s
HYDRAULIC_SENSORS = {
    "PS1": ("Pressure", "bar", 100), "PS2": ("Pressure", "bar", 100),
    "PS3": ("Pressure", "bar", 100), "PS4": ("Pressure", "bar", 100),
    "PS5": ("Pressure", "bar", 100), "PS6": ("Pressure", "bar", 100),
    "EPS1": ("Motor power", "W", 100),
    "FS1": ("Volume flow", "l/min", 10), "FS2": ("Volume flow", "l/min", 10),
    "TS1": ("Temperature", "°C", 1), "TS2": ("Temperature", "°C", 1),
    "TS3": ("Temperature", "°C", 1), "TS4": ("Temperature", "°C", 1),
    "VS1": ("Vibration", "mm/s", 1),
    "CE": ("Cooling efficiency (virtual)", "%", 1),
    "CP": ("Cooling power (virtual)", "kW", 1),
    "SE": ("Efficiency factor (virtual)", "%", 1),
}
CYCLE_SECONDS = 60

STATS = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}


def extract_features(raw_signals):
    """Summarise every cycle of every sensor with mean, std, min and max.

    raw_signals: dict {sensor name: array of shape (cycles, time steps)}
    returns:     DataFrame of shape (cycles, 4 * number of sensors), columns like 'PS1_mean'
    """
    columns = {}
    for sensor, values in raw_signals.items():
        values = np.atleast_2d(np.asarray(values, dtype=float))
        for stat_name, fn in STATS.items():
            columns[f"{sensor}_{stat_name}"] = fn(values, axis=1)
    return pd.DataFrame(columns)


def read_cycle_csv(file):
    """Read one hydraulic cycle from a CSV.

    Expected layout: one column per sensor (PS1 ... SE), one row per time step.
    Slower sensors have fewer readings, so their columns are shorter and the
    remaining cells are left empty.

    returns: (signals, problems)
        signals  dict {sensor: 1-D array} for every sensor that was found
        problems list of human-readable issues (empty when the file is usable)
    """
    df = pd.read_csv(file)
    df.columns = [str(c).strip().upper() for c in df.columns]
    signals, problems = {}, []

    missing = [s for s in HYDRAULIC_SENSORS if s not in df.columns]
    if missing:
        problems.append("Missing sensor columns: " + ", ".join(missing))

    for sensor, (_, _, hz) in HYDRAULIC_SENSORS.items():
        if sensor not in df.columns:
            continue
        values = pd.to_numeric(df[sensor], errors="coerce").dropna().to_numpy()
        expected = hz * CYCLE_SECONDS
        if len(values) != expected:
            problems.append(f"{sensor}: found {len(values)} readings, expected {expected} ({hz} Hz × 60 s)")
        if len(values) == 0:
            continue
        signals[sensor] = values
    return signals, problems


def cycle_to_csv(signals):
    """Inverse of read_cycle_csv: one column per sensor, shorter columns padded with blanks."""
    longest = max(len(v) for v in signals.values())
    return pd.DataFrame({
        s: pd.Series(np.asarray(v, dtype=float)).reindex(range(longest))
        for s, v in signals.items()
    })


# ---------------------------------------------------------------------------
# AI4I 2020 predictive maintenance (classification track)
# ---------------------------------------------------------------------------

AI4I_INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AI4I_TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}


def ai4i_engineer_features(df):
    """Same engineered features as notebook 02 (section B.3).

    Takes the six raw AI4I inputs (Type as 'L'/'M'/'H') and returns a numeric
    DataFrame. Use it inside the model pipeline, e.g.
        FunctionTransformer(ai4i_engineer_features)
    so the saved pipeline accepts raw inputs directly.
    """
    out = pd.DataFrame(df, copy=True)[AI4I_INPUT_COLUMNS]
    out["Temp_Difference_K"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Temp_Ratio"] = out["Process temperature [K]"] / out["Air temperature [K]"]
    out["Power_kW"] = (out["Torque [Nm]"] * out["Rotational speed [rpm]"] * (2 * np.pi / 60.0)) / 1000.0
    out["Cumulative_Strain"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    if not pd.api.types.is_numeric_dtype(out["Type"]):
        out["Type"] = out["Type"].astype(str).str.strip().str.upper().map(AI4I_TYPE_MAPPING)
    if out["Type"].isna().any():
        raise ValueError("Type must be one of L, M, H")
    return out


# ---------------------------------------------------------------------------
# NASA C-MAPSS FD001 turbofan engines (regression track)
# ---------------------------------------------------------------------------

TURBOFAN_COLUMNS = (["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
                    + [f"sensor_{i}" for i in range(1, 22)])
# the 14 sensors that vary in FD001 (notebook 01, section A.1b)
TURBOFAN_SENSOR_COLS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
TURBOFAN_STEP = 3           # training keeps every 3rd cycle (section B.1)
TURBOFAN_SMOOTH_ROWS = 5    # backward rolling mean over 5 kept rows (= 15 cycles)
TURBOFAN_BASELINE_ROWS = 5  # sensor_11 baseline = mean of the first 5 kept rows
DEGRADATION_RISE = ["sensor_11_smooth", "sensor_4_smooth", "sensor_15_smooth"]
DEGRADATION_FALL = ["sensor_12_smooth"]


def turbofan_downsample(history):
    """Keep every 3rd cycle, counted back from each engine's last recorded cycle.

    Same spacing as training, while still keeping the latest cycle, which is the
    one the prediction is made for.
    """
    last = history.groupby("unit_number")["time_cycles"].transform("max")
    return history[(last - history["time_cycles"]) % TURBOFAN_STEP == 0]


def turbofan_features(history, minmax_stats, downsample=True):
    """Engineered features for every kept cycle of one or more engines (notebook 01, B.3).

    history:      DataFrame with time_cycles and the sensor columns (unit_number optional)
    minmax_stats: {column: (min, max)} learned on the training engines, saved with the model
    returns:      DataFrame with unit_number, time_cycles, the smoothed sensors,
                  Sensor11_Drift and Composite_Degradation_Index
    """
    h = pd.DataFrame(history, copy=True)
    if "unit_number" not in h.columns:
        h["unit_number"] = 1
    h = h.sort_values(["unit_number", "time_cycles"])
    if downsample:
        h = turbofan_downsample(h)
    h = h.reset_index(drop=True)
    groups = h.groupby("unit_number")

    for c in TURBOFAN_SENSOR_COLS:
        h[c + "_smooth"] = groups[c].transform(
            lambda s: s.rolling(TURBOFAN_SMOOTH_ROWS, min_periods=1).mean())
    baseline = groups["sensor_11"].transform(lambda s: s.iloc[:TURBOFAN_BASELINE_ROWS].mean())
    h["Sensor11_Drift"] = h["sensor_11_smooth"] - baseline

    parts = []
    for c in DEGRADATION_RISE:
        mn, mx = minmax_stats[c]
        parts.append((h[c] - mn) / (mx - mn + 1e-9))
    for c in DEGRADATION_FALL:
        mn, mx = minmax_stats[c]
        parts.append(1 - (h[c] - mn) / (mx - mn + 1e-9))
    h["Composite_Degradation_Index"] = np.mean(parts, axis=0)
    return h


def read_engine_history(file):
    """Read one engine's cycle history.

    Accepts either a CSV with a header row (time_cycles, sensor_2, ...) or NASA's
    original space-separated format with 26 unlabeled columns.
    returns: (history DataFrame or None, list of problems)
    """
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    first = text.strip().splitlines()[0] if text.strip() else ""
    if any(ch.isalpha() for ch in first):
        df = pd.read_csv(io.StringIO(text))
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
        if df.shape[1] != len(TURBOFAN_COLUMNS):
            return None, [f"Expected {len(TURBOFAN_COLUMNS)} columns in NASA format, found {df.shape[1]}"]
        df.columns = TURBOFAN_COLUMNS

    needed = ["time_cycles"] + TURBOFAN_SENSOR_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None, ["Missing columns: " + ", ".join(missing)]
    df[needed] = df[needed].apply(pd.to_numeric, errors="coerce")
    if df[needed].isna().any().any():
        return None, ["Some values are empty or not numbers"]

    problems = []
    if "unit_number" in df.columns and df["unit_number"].nunique() > 1:
        first_unit = df["unit_number"].iloc[0]
        problems.append(f"File has {df['unit_number'].nunique()} engines; using engine {first_unit} only")
        df = df[df["unit_number"] == first_unit]
    if df["time_cycles"].duplicated().any():
        return None, ["time_cycles has repeated values"]
    if len(df) < TURBOFAN_STEP * TURBOFAN_BASELINE_ROWS:
        problems.append(f"Only {len(df)} cycles recorded; the baseline needs at least "
                        f"{TURBOFAN_STEP * TURBOFAN_BASELINE_ROWS}, so the estimate is rough")
    return df.drop(columns=["unit_number"], errors="ignore").sort_values("time_cycles"), problems
