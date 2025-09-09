import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DDH Deviation Visualizer", layout="wide")

# ---------- helpers ----------
def wrap_az(az):
    az = az % 360.0
    if az < 0:
        az += 360.0
    return az

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def min_curvature_path(stations):
    if not stations:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
    stations = sorted(stations, key=lambda d: float(d["MD"]))
    MDs = np.array([float(s["MD"]) for s in stations], float)
    AZs = np.deg2rad([wrap_az(float(s["Azimuth"])) for s in stations])
    DIP = np.array([float(s["Angle"]) for s in stations], float)
    INC = np.deg2rad(90.0 - np.abs(DIP))
    SGN = np.where(DIP <= 0.0, -1.0, 1.0)
    X, Y, Z = [0.0], [0.0], [0.0]
    for i in range(1, len(MDs)):
        dMD = MDs[i] - MDs[i-1]
        if dMD <= 0: continue
        inc1, inc2 = INC[i-1], INC[i]
        az1, az2 = AZs[i-1], AZs[i]
        s1, s2 = SGN[i-1], SGN[i]
        cos_dog = np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1) + np.cos(inc1)*np.cos(inc2)
        cos_dog = float(np.clip(cos_dog, -1.0, 1.0))
        dog = np.arccos(cos_dog)
        RF = 1.0 if dog < 1e-12 else (2.0/dog)*np.tan(dog/2.0)
        dN = 0.5*dMD*(np.sin(inc1)*np.cos(az1) + np.sin(inc2)*np.cos(az2))*RF
        dE = 0.5*dMD*(np.sin(inc1)*np.sin(az1) + np.sin(inc2)*np.sin(az2))*RF
        dZ = 0.5*dMD*(s1*np.cos(inc1) + s2*np.cos(inc2))*RF
        X.append(X[-1] + dE)
        Y.append(Y[-1] + dN)
        Z.append(Z[-1] + dZ)
    return np.array(X), np.array(Y), np.array(Z), MDs

def make_planned_stations(length_m, step_m, az0, dip0, lift_per100, drift_per100):
    md = 0.0
    az = wrap_az(az0)
    dip = clamp(dip0, -90.0, 90.0)
    stations = [{"MD": md, "Azimuth": az, "Angle": dip}]
    while md < length_m - 1e-9:
        d = min(step_m, length_m - md)
        md += d
        az = wrap_az(az + drift_per100*(d/100.0))
        dip = clamp(dip + lift_per100*(d/100.0), -90.0, 90.0)  # positive lift shallows dip
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def extend_actual(stations, to_depth, step_m, lift_per100, drift_per100):
    if not stations: return stations
    stations = sorted(stations, key=lambda d: float(d["MD"]))
    md = float(stations[-1]["MD"])
    az = float(stations[-1]["Azimuth"])
    dip = float(stations[-1]["Angle"])
    while md < to_depth - 1e-9:
        d = min(step_m, to_depth - md)
        md += d
        az = wrap_az(az + drift_per100*(d/100.0))
        dip = clamp(dip + lift_per100*(d/100.0), -90.0, 90.0)
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def derive_lift_drift_last3(stations):
    if len(stations) < 3: return None, None
    sta = sorted(stations, key=lambda d: float(d["MD"]))[-3:]
    MD = np.array([float(s["MD"]) for s in sta], float)
    AZ = np.array([float(s["Azimuth"]) for s in sta], float)
    DIP = np.array([float(s["Angle"]) for s in sta], float)
    AZu = np.unwrap(np.deg2rad(AZ))
    drift_deg_per_m = np.rad2deg(np.polyfit(MD, AZu, 1)[0])
    lift_deg_per_m = np.polyfit(MD, DIP, 1)[0]
    return float(lift_deg_per_m*100.0), float(drift_deg_per_m*100.0)

def az_diff_deg(a2, a1):
    d = np.deg2rad(a2) - np.deg2rad(a1)
    d = np.arctan2(np.sin(d), np.cos(d))
    return np.rad2deg(d)

def add_delta_columns(df):
    if df is None or df.empty: return df
    df = df.sort_values("MD").reset_index(drop=True).copy()
    df["dMD"] = df["MD"].diff()
    df.loc[1:, "dAz_deg"] = [
        az_diff_deg(df.loc[i, "Azimuth"], df.loc[i-1, "Azimuth"])
        for i in range(1, len(df))
    ]
    df["dDip_deg"] = df["Angle"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Az change deg/100m"] = np.where(df["dMD"] > 0, df["dAz_deg"]/df["dMD"]*100.0, np.nan)
        df["Dip change deg/100m"] = np.where(df["dMD"] > 0, df["dDip_deg"]/df["dMD"]*100.0, np.nan)
    return df.drop(columns=["dMD","dAz_deg","dDip_deg"], errors="ignore")

# ---------- UI ----------
st.title("Drillhole vs Planned - single 3D view")

# planned inputs
colA, colB, colC = st.columns(3)
with colA:
    plan_len = st.number_input("Planned length m", value=500.0, step=10.0, min_value=1.0)
    step_m = st.number_input("Computation step m", value=10.0, step=1.0, min_value=1.0)
with colB:
    plan_az0 = st.number_input("Planned start azimuth deg", value=90.0, step=1.0)
    plan_dip0 = st.number_input("Planned start dip-from-horizontal deg (negative down)", value=-60.0, step=1.0)
with colC:
    plan_lift = st.number_input("Planned lift deg/100m", value=2.0, step=0.1)
    plan_drift = st.number_input("Planned drift deg/100m", value=1.0, step=0.1)

planned_stations = make_planned_stations(plan_len, step_m, plan_az0, plan_dip0, plan_lift, plan_drift)
px, py, pz, pmd = min_curvature_path(planned_stations)

# actual surveys
st.subheader("Actual surveys")
method = st.radio("Provide surveys via", ["Excel upload","CSV upload","Manual entry"], horizontal=True)
df_in = pd.DataFrame(columns=["MD","Azimuth","Angle"])
if method == "Excel upload":
    file = st.file_uploader("Upload Excel workbook", type=["xlsx","xls"])
    if file:
        try:
            xls = pd.ExcelFile(file)
            sheet = st.selectbox("Pick worksheet", xls.sheet_names, index=0)
            df_in = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
elif method == "CSV upload":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            df_in = pd.read_csv(file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
else:
    df_in = st.data_editor(pd.DataFrame([{"MD":0.0,"Azimuth":plan_az0,"Angle":plan_dip0}]), num_rows="dynamic")

if not df_in.empty:
    df_show = add_delta_columns(df_in)
    st.dataframe(df_show, use_container_width=True)

# Collar inputs always available
st.markdown("#### Actual collar override")
collar_az = st.number_input("Actual collar azimuth deg", value=float(df_in.iloc[0]["Azimuth"]) if not df_in.empty else plan_az0)
collar_dip = st.number_input("Actual collar dip-from-horizontal deg (negative down)", value=float(df_in.iloc[0]["Angle"]) if not df_in.empty else plan_dip0)
