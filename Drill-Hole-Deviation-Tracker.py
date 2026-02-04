import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Drill Hole vs Planned Hole Deviation Tracking", layout="wide")

# ----------------------------------------------------------------------
# Disclaimer
# ----------------------------------------------------------------------
st.markdown(
    """
> **Disclaimer / Test Phase**
>
> This free app helps you quickly track the deviation of a drill hole compared to a planned hole.
> You can export the data as a JSON file and reload it later or share it with other team members.
>
> This app is in a test phase and is only updated periodically. The author makes no guarantee of accuracy.
> **You should confirm hole deviation using other commercial software or independent verification methods.**
"""
)

# ----------------------------------------------------------------------
# Apply saved session BEFORE widgets
# ----------------------------------------------------------------------
def apply_cfg_to_session(cfg):
    st.session_state.plan_len   = cfg.get("planned", {}).get("plan_len", 500.0)
    st.session_state.step_m     = cfg.get("planned", {}).get("step_m", 10.0)
    st.session_state.plan_az0   = cfg.get("planned", {}).get("plan_az0", 90.0)
    st.session_state.plan_dip0  = cfg.get("planned", {}).get("plan_dip0", -60.0)
    st.session_state.plan_lift  = cfg.get("planned", {}).get("plan_lift", 2.0)
    st.session_state.plan_drift = cfg.get("planned", {}).get("plan_drift", 1.0)

    if "surveys" in cfg:
        st.session_state["loaded_surveys_df"] = pd.DataFrame(cfg["surveys"])

    st.session_state.rem_lift  = cfg.get("rem_lift", 2.0)
    st.session_state.rem_drift = cfg.get("rem_drift", 1.0)
    st.session_state.actual_len = cfg.get("actual_len", 500.0)
    st.session_state.use_planned_zero = cfg.get("use_planned_zero", False)

    st.session_state.act_az0  = cfg.get("actual", {}).get("act_az0", st.session_state.plan_az0)
    st.session_state.act_dip0 = cfg.get("actual", {}).get("act_dip0", st.session_state.plan_dip0)

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
for k, v in {
    "plan_len": 500.0,
    "step_m": 10.0,
    "plan_az0": 90.0,
    "plan_dip0": -60.0,
    "plan_lift": 2.0,
    "plan_drift": 1.0,
    "rem_lift": 2.0,
    "rem_drift": 1.0,
    "actual_len": 500.0,
    "use_planned_zero": False,
    "act_az0": 90.0,
    "act_dip0": -60.0,
}.items():
    st.session_state.setdefault(k, v)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def wrap_az(az):
    return az % 360.0

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

# ----------------------------------------------------------------------
# FLEXIBLE SURVEY PARSER  ✅ UPDATED
# ----------------------------------------------------------------------
def _parse_table_flexible_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Accepted headers:
      MD / Measured Depth
      Azimuth / Azi / AZ / True North Azimuth
      Angle / Dip / Inclination / Incl
    """
    df = df_in.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_md = pick("md", "measured depth", "measured_depth")
    c_az = pick(
        "azimuth", "azi", "az",
        "true north azimuth", "true_north_azimuth"
    )
    c_an = pick("angle", "dip", "inclination", "incl")

    if not all([c_md, c_az, c_an]):
        return pd.DataFrame(columns=["MD", "Azimuth", "Angle"])

    out = df[[c_md, c_az, c_an]].copy()
    out.columns = ["MD", "Azimuth", "Angle"]

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna()
    out["Azimuth"] = out["Azimuth"] % 360.0
    return out

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("Drill Hole vs Planned Hole Deviation Tracking")

st.subheader("Downhole surveys")

method = st.radio("Provide surveys via", ["CSV or Excel upload", "Manual entry"], horizontal=True)

df_in = pd.DataFrame(columns=["MD", "Azimuth", "Angle", "Active"])

if method == "CSV or Excel upload":
    file = st.file_uploader(
        "Upload CSV or Excel with MD, Azimuth, Dip",
        type=["csv", "xlsx", "xls"]
    )

    if "loaded_surveys_df" in st.session_state:
        df_in = st.session_state["loaded_surveys_df"].copy()

    if file is not None:
        try:
            if file.name.lower().endswith(("xls", "xlsx")):
                raw_df = pd.read_excel(file)
            else:
                raw_df = pd.read_csv(file)

            df_parsed = _parse_table_flexible_df(raw_df)

            if len(df_parsed) > 0 and df_parsed["Angle"].median() > 0:
                df_parsed["Angle"] *= -1
                st.info("Detected positive-down dips. Converted to negative-down.")

            df_in = df_parsed.copy()
        except Exception as e:
            st.error(f"File read error: {e}")

    if "Active" not in df_in.columns:
        df_in["Active"] = True

    df_view = df_in.rename(columns={"Angle": "Dip (Negative is Down)"})
    edited = st.data_editor(df_view, use_container_width=True, num_rows="dynamic")
    df_in = edited.rename(columns={"Dip (Negative is Down)": "Angle"})

else:
    manual = pd.DataFrame([
        {"MD": 0.0, "Azimuth": st.session_state.plan_az0, "Angle": st.session_state.plan_dip0, "Active": True},
        {"MD": 50.0, "Azimuth": st.session_state.plan_az0, "Angle": st.session_state.plan_dip0, "Active": True},
    ])
    edited = st.data_editor(manual, num_rows="dynamic", use_container_width=True)
    df_in = edited.copy()

# ----------------------------------------------------------------------
# Build actual stations
# ----------------------------------------------------------------------
df_use = df_in[df_in["Active"] == True]

actual_stations = [
    {
        "MD": float(r.MD),
        "Azimuth": wrap_az(float(r.Azimuth)),
        "Angle": clamp(float(r.Angle), -90.0, 90.0)
    }
    for _, r in df_use.dropna().iterrows()
]

st.success(
    "✅ Import complete. "
    "Accepted azimuth headers include **Azimuth**, **Azi**, **AZ**, and **True North Azimuth**."
)

st.dataframe(pd.DataFrame(actual_stations))
