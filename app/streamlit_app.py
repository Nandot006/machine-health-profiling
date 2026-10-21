"""Machine Health Profiling — Streamlit app (23CSE301 capstone, Review 2).

Tab 1: Turbofan engine remaining useful life (regression track).
Tab 2: AI4I 2020 machine-failure prediction (classification track).
Tab 3: Hydraulic system health profiling with K-Means (clustering track).

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import features  # noqa: F401  (also needed so pickled pipelines can find their functions)
from features import (AI4I_INPUT_COLUMNS, HYDRAULIC_SENSORS, TURBOFAN_COLUMNS,
                      extract_features, read_cycle_csv, read_engine_history,
                      turbofan_features)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "models"
SAMPLE_DIR = APP_DIR / "samples"
AI4I_MODEL = MODEL_DIR / "ai4i_pipeline.pkl"
HYDRAULIC_MODEL = MODEL_DIR / "hydraulic_kmeans.pkl"
TURBOFAN_MODEL = MODEL_DIR / "turbofan_rul.pkl"

STATUS_COLOURS = {"Healthy": "#2e7d32", "Degraded": "#f9a825",
                  "Warning": "#ef6c00", "Critical": "#c62828"}

st.set_page_config(page_title="Machine Health Profiling", page_icon="⚙️", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached so models are read once per server, not on every click)
# ---------------------------------------------------------------------------

def load_ai4i():
    """Return (model, info dict) or (None, None) when the model file is not there yet."""
    if not AI4I_MODEL.exists():
        return None, None
    return _load_ai4i(AI4I_MODEL.stat().st_mtime)


@st.cache_resource
def _load_ai4i(_mtime):
    # cached per file timestamp, so a newly added or replaced model is picked up
    obj = joblib.load(AI4I_MODEL)
    if isinstance(obj, dict):
        info = {k: v for k, v in obj.items() if k != "pipeline"}
        return obj["pipeline"], info
    return obj, {}


@st.cache_resource
def load_hydraulic():
    return joblib.load(HYDRAULIC_MODEL)


@st.cache_data
def load_background():
    """PCA coordinates and cluster of every training cycle, for the 2-D map."""
    return pd.read_csv(SAMPLE_DIR / "pca_background.csv")


@st.cache_data
def load_cluster_profiles():
    """Average feature values of each cluster (original units), for comparison."""
    return pd.read_csv(SAMPLE_DIR / "cluster_profiles.csv", index_col="cluster")


def status_of(name):
    return name.split(":")[0].strip()


def colour_of(name):
    return STATUS_COLOURS.get(status_of(name), "#555555")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Machine Health Profiling")
    st.caption("23CSE301 Machine Learning — Capstone, Review 2")
    st.markdown(
        "**Remaining life** — how many cycles does a jet engine have left? "
        "(NASA C-MAPSS FD001, regression)\n\n"
        "**Failure prediction** — will a milling machine fail under these "
        "operating conditions? (AI4I 2020, classification)\n\n"
        "**Hydraulic health profile** — which health group does one 60-second "
        "test-rig cycle belong to? (UCI hydraulic systems, K-Means clustering)"
    )
    st.divider()
    st.caption("Datasets: NASA C-MAPSS Turbofan Degradation (FD001) · "
               "AI4I 2020 Predictive Maintenance (UCI) · "
               "Condition Monitoring of Hydraulic Systems (UCI)")

tab_reg, tab_cls, tab_clu = st.tabs(["✈️ Remaining life (Turbofan)", "🔧 Failure prediction (AI4I)",
                                     "🛢️ Hydraulic health profile"])


# ---------------------------------------------------------------------------
# AI4I failure prediction (classification)
# ---------------------------------------------------------------------------

def physics_checks(type_, air, proc, rpm, torque, wear):
    """Failure rules published with the AI4I dataset, shown next to the model's answer."""
    dt = proc - air
    power_w = torque * rpm * 2 * np.pi / 60
    strain_limit = {"L": 11000, "M": 12000, "H": 13000}[type_]
    return pd.DataFrame([
        ["Heat dissipation (HDF)", f"ΔT = {dt:.1f} K, speed = {rpm} rpm",
         "ΔT < 8.6 K and speed < 1380 rpm", dt < 8.6 and rpm < 1380],
        ["Power (PWF)", f"{power_w:,.0f} W", "below 3,500 W or above 9,000 W",
         power_w < 3500 or power_w > 9000],
        ["Overstrain (OSF)", f"{wear * torque:,.0f} min·Nm",
         f"above {strain_limit:,} min·Nm for type {type_}", wear * torque > strain_limit],
        ["Tool wear (TWF)", f"{wear} min", "tool replaced/failed between 200 and 240 min",
         200 <= wear <= 240],
    ], columns=["Failure mode", "Current value", "Rule from the dataset documentation", "Triggered"])


with tab_cls:
    st.header("Machine failure prediction")
    st.write("Set the operating conditions of the milling machine. The model estimates the "
             "probability that this run ends in a failure.")

    model, info = load_ai4i()

    c1, c2, c3 = st.columns(3)
    with c1:
        type_ = st.selectbox("Product quality type", ["L", "M", "H"],
                             format_func=lambda t: {"L": "L — low", "M": "M — medium", "H": "H — high"}[t])
        air = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
    with c2:
        proc = st.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
        rpm = st.slider("Rotational speed [rpm]", 1160, 2900, 1500, 10)
    with c3:
        torque = st.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.5)
        wear = st.slider("Tool wear [min]", 0, 255, 100, 1)

    inputs = pd.DataFrame([[type_, air, proc, rpm, torque, wear]], columns=AI4I_INPUT_COLUMNS)

    if model is None:
        st.info("The classification model has not been added yet. When "
                "`app/models/ai4i_pipeline.pkl` is in place, the prediction appears here. "
                "The physics checks below already work.", icon="⏳")
    else:
        try:
            proba = float(model.predict_proba(inputs)[0, 1])
        except Exception as exc:  # show the reason instead of crashing the page
            st.error(f"The model could not score these inputs: {exc}")
        else:
            threshold = float(info.get("threshold", 0.5))
            failing = proba >= threshold
            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Failure probability", f"{proba:.1%}")
                if failing:
                    st.error(f"**Failure predicted** (threshold {threshold:.0%})", icon="⚠️")
                else:
                    st.success(f"**No failure predicted** (threshold {threshold:.0%})", icon="✅")
            with m2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "#c62828" if failing else "#2e7d32"},
                           "threshold": {"line": {"color": "black", "width": 3},
                                         "value": threshold * 100}}))
                gauge.update_layout(height=220, margin=dict(t=10, b=10, l=30, r=30))
                st.plotly_chart(gauge, width="stretch")
            details = []
            if info.get("model_name"):
                details.append(f"Model: **{info['model_name']}**")
            for k, v in (info.get("metrics") or {}).items():
                details.append(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
            if details:
                st.caption(" · ".join(details) + " (held-out test set)")

    st.subheader("Physics checks")
    st.caption("The AI4I dataset was generated from these documented rules. They are shown "
               "for comparison only; the prediction above comes from the trained model.")
    checks = physics_checks(type_, air, proc, rpm, torque, wear)
    checks["Triggered"] = checks["Triggered"].map({True: "⚠️ yes", False: "no"})
    st.dataframe(checks, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Hydraulic health profiling (clustering)
# ---------------------------------------------------------------------------

with tab_clu:
    st.header("Hydraulic system health profile")
    st.write("Each test-rig cycle lasts 60 seconds and is recorded by 17 sensors. The app "
             "summarises the cycle into 65 features (mean, std, min, max per sensor) and "
             "assigns it to one of four health groups found by K-Means.")

    bundle = load_hydraulic()
    pipeline, pca = bundle["pipeline"], bundle["pca"]
    feature_names = bundle["feature_names"]
    cluster_names = {int(k): v for k, v in bundle["cluster_names"].items()}

    order = ["healthy", "degraded", "warning", "critical"]
    samples = sorted(SAMPLE_DIR.glob("sample_*.csv"),
                     key=lambda p: next((i for i, o in enumerate(order) if o in p.stem), 99))
    source = st.radio("Cycle to analyse", ["Use a sample cycle", "Upload my own CSV"], horizontal=True)

    signals, problems, label = None, [], None
    if source == "Use a sample cycle":
        pick = st.selectbox("Sample cycle", samples,
                            format_func=lambda p: "{} (cycle {})".format(
                                *p.stem.replace("sample_", "").replace("cycle", "").split("_")[:2]).capitalize())
        signals, problems = read_cycle_csv(pick)
        label = pick.name
    else:
        with st.expander("Expected file format"):
            st.markdown(
                "- One CSV per cycle, **one column per sensor**: "
                + ", ".join(f"`{s}`" for s in HYDRAULIC_SENSORS) + ".\n"
                "- One row per reading. 100 Hz sensors (PS1–PS6, EPS1) have 6,000 readings, "
                "10 Hz sensors (FS1, FS2) 600, and 1 Hz sensors 60; leave the rest of the "
                "shorter columns empty.\n"
                "- Download a sample below to see the layout.")
            if samples:
                st.download_button("Download a sample cycle", samples[0].read_bytes(),
                                   file_name=samples[0].name, mime="text/csv")
        upload = st.file_uploader("Cycle CSV", type="csv")
        if upload is not None:
            try:
                signals, problems = read_cycle_csv(upload)
                label = upload.name
            except Exception as exc:
                st.error(f"Could not read this file as a CSV: {exc}")

    if signals is not None:
        missing = [s for s in HYDRAULIC_SENSORS if s not in signals]
        if missing:
            st.error("This cycle can't be profiled. " + " ".join(problems))
        else:
            if problems:
                st.warning("The file was read, but some sensors have an unexpected number of "
                           "readings, so the result may be less reliable: " + "; ".join(problems))

            feats = extract_features({s: signals[s] for s in HYDRAULIC_SENSORS})[feature_names]
            cluster = int(pipeline.predict(feats)[0])
            name = cluster_names[cluster]
            scaled = pipeline.named_steps["scaler"].transform(feats)
            distances = pipeline.named_steps["kmeans"].transform(scaled)[0]
            point = pca.transform(scaled)[0]

            r1, r2 = st.columns([1, 1])
            with r1:
                st.markdown(f"<div style='padding:1rem 1.2rem;border-radius:0.6rem;"
                            f"border-left:8px solid {colour_of(name)};background:rgba(127,127,127,0.08)'>"
                            f"<div style='font-size:0.85rem;opacity:0.7'>Health group for {label}</div>"
                            f"<div style='font-size:1.6rem;font-weight:700;color:{colour_of(name)}'>"
                            f"{status_of(name)}</div>"
                            f"<div>{name.split(':', 1)[-1].strip()}</div></div>",
                            unsafe_allow_html=True)
                st.write("")
                dist = pd.DataFrame({
                    "Health group": [cluster_names[c] for c in range(len(distances))],
                    "Distance to group centre": distances,
                }).sort_values("Distance to group centre")
                st.caption("Distance from this cycle to each group centre (standardised "
                           "feature space). Smaller means more similar.")
                st.dataframe(dist.style.format({"Distance to group centre": "{:.2f}"}),
                             hide_index=True, width="stretch")
            with r2:
                bg = load_background()
                bg["Health group"] = bg["cluster"].map(cluster_names)
                severity = ["Healthy", "Degraded", "Warning", "Critical"]
                ordered = sorted(cluster_names.values(), key=lambda n: severity.index(status_of(n))
                                 if status_of(n) in severity else 99)
                # render_mode="svg": the WebGL default would draw over the "this cycle" marker
                fig = px.scatter(bg, x="PC1", y="PC2", color="Health group", opacity=0.35,
                                 render_mode="svg", category_orders={"Health group": ordered},
                                 color_discrete_map={n: colour_of(n) for n in cluster_names.values()},
                                 title="Where this cycle sits among the 2,205 training cycles (PCA)")
                fig.update_traces(marker=dict(size=5))
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers",
                                         name="This cycle",
                                         # bright blue: no group uses it, and it reads on light and dark themes
                                         marker=dict(symbol="star", size=22, color="#1e88e5",
                                                     line=dict(color="white", width=1.5))))
                fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2),
                                  margin=dict(t=50, b=10))
                st.plotly_chart(fig, width="stretch")

            st.subheader("Key readings compared with each group")
            profiles = load_cluster_profiles()
            key = ["CE_mean", "CP_mean", "TS1_mean", "PS1_mean", "FS1_mean", "SE_mean"]
            key = [k for k in key if k in profiles.columns]
            table = profiles[key].copy()
            table.index = [cluster_names[int(c)] for c in table.index]
            table.loc["▶ This cycle"] = feats[key].iloc[0].values
            units = {k: HYDRAULIC_SENSORS[k.split("_")[0]] for k in key}
            table.columns = [f"{units[k][0]} {k.split('_')[0]} ({units[k][1]})" for k in key]
            st.dataframe(table.style.format("{:.2f}"), width="stretch")

            st.subheader("Raw signals of this cycle")
            chosen = st.multiselect("Sensors to plot", list(HYDRAULIC_SENSORS),
                                    default=["PS1", "FS1", "TS1", "CE"])
            if chosen:
                cols = st.columns(min(len(chosen), 2))
                for i, s in enumerate(chosen):
                    desc, unit, hz = HYDRAULIC_SENSORS[s]
                    y = signals[s]
                    f = px.line(x=np.arange(len(y)) / hz, y=y,
                                labels={"x": "Time within cycle (s)", "y": unit},
                                title=f"{s} — {desc}")
                    f.update_layout(height=260, margin=dict(t=40, b=10))
                    cols[i % len(cols)].plotly_chart(f, width="stretch")

    st.caption("Groups were named after training by comparing them with the rig's recorded "
               "component conditions (cooler, valve, pump, accumulator), which K-Means never saw.")


# ---------------------------------------------------------------------------
# Turbofan remaining useful life (regression)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_turbofan():
    return joblib.load(TURBOFAN_MODEL)


@st.cache_data
def load_nasa_test():
    """NASA's official FD001 test engines and their true remaining life."""
    engines = pd.read_csv(SAMPLE_DIR / "nasa_test_FD001.txt", sep=r"\s+", header=None,
                          names=TURBOFAN_COLUMNS)
    true_rul = pd.read_csv(SAMPLE_DIR / "nasa_RUL_FD001.txt", header=None, names=["rul"])["rul"]
    true_rul.index = true_rul.index + 1
    return engines, true_rul


def rul_band(rul):
    """Illustrative maintenance bands for the display (not part of the model)."""
    if rul <= 25:
        return "Critical", "Schedule maintenance now", STATUS_COLOURS["Critical"]
    if rul <= 60:
        return "Warning", "Plan maintenance soon", STATUS_COLOURS["Warning"]
    return "Healthy", "No action needed yet", STATUS_COLOURS["Healthy"]


with tab_reg:
    st.header("Turbofan engine remaining useful life")
    st.write("Each row of an engine's history is one flight cycle with 21 sensor readings. "
             "The model smooths the 14 informative sensors over the recent past, measures drift "
             "from the engine's own early-life baseline, and estimates how many cycles remain "
             "before failure.")

    tb = load_turbofan()
    cap = tb["rul_cap"]
    m = tb["metrics"]
    st.caption(f"Model: **{tb['model_name']}** · test RMSE ±{m['Test RMSE (cycles)']:.1f} cycles "
               f"(R² {m['Test R2']:.3f}) · on NASA's 100 official test engines RMSE "
               f"±{m['Official test RMSE (cycles)']:.1f} cycles (R² {m['Official test R2']:.3f})")

    engines, true_rul = load_nasa_test()
    src = st.radio("Engine history", ["NASA official test engine", "Upload my own"], horizontal=True)

    history, problems, rul_at_end, label = None, [], None, None
    if src == "NASA official test engine":
        units = list(true_rul.index)
        # opens on engine 24: a long history that ends close to failure, so the curve is visible
        unit = st.selectbox("Engine", units, index=units.index(24) if 24 in units else 0,
                            format_func=lambda u: f"Engine {u} — {int((engines.unit_number == u).sum())} "
                                                  f"cycles recorded")
        history = engines[engines.unit_number == unit].drop(columns="unit_number")
        rul_at_end = int(true_rul[unit])
        label = f"NASA test engine {unit}"
    else:
        with st.expander("Expected file format"):
            st.markdown(
                "- One engine per file, one row per cycle, oldest first.\n"
                "- Either a CSV with a header (`time_cycles`, `sensor_1` … `sensor_21`; only "
                "`time_cycles` and sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21 are "
                "used), or NASA's original space-separated format (26 columns, no header).\n"
                "- At least 15 cycles, so the early-life baseline can be measured.")
            example = engines[engines.unit_number == 1].drop(columns="unit_number")
            st.download_button("Download an example engine (CSV)", example.to_csv(index=False),
                               file_name="engine_history_example.csv", mime="text/csv")
        upload = st.file_uploader("Engine history", type=["csv", "txt"])
        if upload is not None:
            try:
                history, problems = read_engine_history(upload)
                label = upload.name
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")
        if history is None and problems:
            st.error("This history can't be used. " + " ".join(problems))
            history = None

    if history is not None:
        for p in problems:
            st.warning(p)
        first, last = int(history.time_cycles.min()), int(history.time_cycles.max())
        if last - first >= 15:
            as_of = st.slider("Look at the engine as of cycle", min(first + 14, last), last, last,
                              help="Rewind to see what the model would have said earlier in "
                                   "this engine's life.")
        else:
            as_of = last
        upto = history[history.time_cycles <= as_of]

        feats = turbofan_features(upto, tb["minmax_stats"])
        preds = tb["model"].predict(tb["scaler"].transform(feats[tb["feature_names"]]))
        pred = float(np.clip(preds[-1], 0, None))
        true_now = None if rul_at_end is None else rul_at_end + (last - as_of)

        band, action, colour = rul_band(pred)
        k1, k2 = st.columns([1, 1])
        with k1:
            shown = f"{pred:.0f}" if pred < cap - 5 else f"{cap}+"
            st.markdown(f"<div style='padding:1rem 1.2rem;border-radius:0.6rem;"
                        f"border-left:8px solid {colour};background:rgba(127,127,127,0.08)'>"
                        f"<div style='font-size:0.85rem;opacity:0.7'>{label}, as of cycle {as_of}</div>"
                        f"<div style='font-size:1.6rem;font-weight:700;color:{colour}'>"
                        f"{shown} cycles left</div><div>{band} — {action}</div></div>",
                        unsafe_allow_html=True)
            if pred >= cap - 5:
                st.caption(f"The model was trained with remaining life capped at {cap} cycles "
                           f"(early-life engines all look healthy), so {cap}+ means \"early in life\".")
            st.caption("Bands (≤25 critical, ≤60 warning) are for display only; they are not "
                       "part of the model.")
        with k2:
            if true_now is not None:
                err = pred - min(true_now, cap)
                st.metric("True remaining life (from NASA)", f"{true_now} cycles",
                          delta=f"model {'over' if err > 0 else 'under'}estimates by {abs(err):.0f}",
                          delta_color="off")
                if true_now > cap:
                    st.caption(f"Compared against {cap}, the cap used in training.")

        traj = pd.DataFrame({"Cycle": feats.time_cycles, "Predicted": np.clip(preds, 0, None)})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj.Cycle, y=traj.Predicted, mode="lines+markers",
                                 name="Predicted remaining life", marker=dict(size=4)))
        if rul_at_end is not None:
            actual = (rul_at_end + last - traj.Cycle).clip(upper=cap)
            fig.add_trace(go.Scatter(x=traj.Cycle, y=actual, mode="lines", name=f"True (capped at {cap})",
                                     line=dict(dash="dash", color="gray")))
        fig.add_hrect(y0=0, y1=25, fillcolor=STATUS_COLOURS["Critical"], opacity=0.08, line_width=0)
        fig.add_hrect(y0=25, y1=60, fillcolor=STATUS_COLOURS["Warning"], opacity=0.08, line_width=0)
        fig.update_layout(title="Predicted remaining life over this engine's recorded history",
                          xaxis_title="Cycle", yaxis_title="Remaining cycles", height=380,
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Degradation signals")
        c1, c2 = st.columns(2)
        f1 = px.line(feats, x="time_cycles", y="Composite_Degradation_Index",
                     labels={"time_cycles": "Cycle", "Composite_Degradation_Index": "Index (0 healthy → 1 worn)"},
                     title="Composite degradation index")
        f2 = px.line(feats, x="time_cycles", y="Sensor11_Drift",
                     labels={"time_cycles": "Cycle", "Sensor11_Drift": "Drift from baseline"},
                     title="sensor_11 drift from this engine's early-life baseline")
        for f, col in ((f1, c1), (f2, c2)):
            f.update_layout(height=280, margin=dict(t=40, b=10))
            col.plotly_chart(f, width="stretch")
