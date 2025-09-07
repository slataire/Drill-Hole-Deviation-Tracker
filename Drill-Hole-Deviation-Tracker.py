import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Drill Hole vs Planned Hole Deviation Tracking", layout="wide")

# ----------------------------------------------------------------------
# Apply saved session BEFORE any widgets are created
# ----------------------------------------------------------------------
def apply_cfg_to_session(cfg):
    # planned
    st.session_state.plan_len   = cfg.get("planned", {}).get("plan_len", st.session_state.get("plan_len", 500.0))
    st.session_state.step_m     = cfg.get("planned", {}).get("step_m", st.session_state.get("step_m", 10.0))
    st.session_state.plan_az0   = cfg.get("planned", {}).get("plan_az0", st.session_state.get("plan_az0", 90.0))
    st.session_state.plan_dip0  = cfg.get("planned", {}).get("plan_dip0", st.session_state.get("plan_dip0", -60.0))
    st.session_state.plan_lift  = cfg.get("planned", {}).get("plan_lift", st.session_state.get("plan_lift", -2.0))
    st.session_state.plan_drift = cfg.get("planned", {}).get("plan_drift", st.session_state.get("plan_drift", -1.0))
    # surveys
    if "surveys" in cfg and isinstance(cfg["surveys"], list):
        st.session_state["loaded_surveys_df"] = pd.DataFrame(cfg["surveys"])
    # actual
    st.session_state["rem_lift"]         = cfg.get("rem_lift", st.session_state.get("rem_lift", -2.0))
    st.session_state["rem_drift"]        = cfg.get("rem_drift", st.session_state.get("rem_drift", -1.0))
    st.session_state["use_planned_zero"] = cfg.get("use_planned_zero", st.session_state.get("use_planned_zero", False))
    st.session_state["actual_len"]       = cfg.get("actual_len", st.session_state.get("actual_len", st.session_state.get("plan_len", 500.0)))
    # plane
    st.session_state["plane_strike"]     = cfg.get("plane", {}).get("strike", st.session_state.get("plane_strike", 114.0))
    st.session_state["plane_dip"]        = cfg.get("plane", {}).get("dip", st.session_state.get("plane_dip", -58.0))
    st.session_state["target_md"]        = cfg.get("plane", {}).get("target_md", st.session_state.get("target_md", max(0.0, st.session_state.get("plan_len", 500.0) - 50.0)))

# defaults so first render has keys
st.session_state.setdefault("plan_len", 500.0)
st.session_state.setdefault("step_m", 10.0)
st.session_state.setdefault("plan_az0", 90.0)
st.session_state.setdefault("plan_dip0", -60.0)
st.session_state.setdefault("plan_lift", -2.0)
st.session_state.setdefault("plan_drift", -1.0)
st.session_state.setdefault("rem_lift", -2.0)
st.session_state.setdefault("rem_drift", -1.0)
st.session_state.setdefault("use_planned_zero", False)
st.session_state.setdefault("actual_len", st.session_state.get("plan_len", 500.0))
st.session_state.setdefault("plane_strike", 114.0)
st.session_state.setdefault("plane_dip", -58.0)
st.session_state.setdefault("target_md", max(0.0, st.session_state.get("plan_len", 500.0) - 50.0))

# apply pending config before widgets are created
if "pending_config" in st.session_state:
    apply_cfg_to_session(st.session_state.pop("pending_config"))

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def wrap_az(az):
    az = az % 360.0
    if az < 0:
        az += 360.0
    return az

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def normalize_dip_az(dip, az):
    """
    Keep dip in [-90, 90] and adjust az by 180 when crossing a pole.

    Convention:
    - Azimuth clockwise from north in horizontal plane.
    - Dip from horizontal, negative down.
    """
    az = float(az)
    dip = float(dip)

    while dip > 90.0:
        dip = 180.0 - dip
        dip = -dip
        az += 180.0

    while dip < -90.0:
        dip = -180.0 - dip
        dip = -dip
        az += 180.0

    return wrap_az(az), clamp(dip, -90.0, 90.0)

def _unit_vec_from_az_dip(az_deg: float, dip_deg: float) -> np.ndarray:
    """
    Convert azimuth and dip-from-horizontal (negative down) to a unit vector [E, N, Z].
    Z is positive up, azimuth is clockwise from north in the horizontal plane.
    """
    az = np.deg2rad(wrap_az(az_deg))
    elev = np.deg2rad(dip_deg)  # dip-from-horizontal, negative down
    ch = np.cos(elev)           # horizontal magnitude
    E = ch * np.sin(az)
    N = ch * np.cos(az)
    Z = np.sin(elev)
    return np.array([E, N, Z], dtype=float)

def _vector_to_az_dip(v: np.ndarray):
    """
    Inverse of _unit_vec_from_az_dip.
    """
    v = np.asarray(v, dtype=float)
    v = v / max(np.linalg.norm(v), 1e-12)
    az = wrap_az(np.rad2deg(np.arctan2(v[0], v[1])))
    dip = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return az, dip

def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotate vector v around unit axis by angle_rad using Rodrigues formula.
    """
    v = np.asarray(v, float)
    axis = np.asarray(axis, float)
    a = axis / max(np.linalg.norm(axis), 1e-12)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1.0 - c)

def step_orientation(az, dip, d, lift_per100, drift_per100):
    """
    Vector stepping on the unit sphere.

    Conventions:
    - Dip is from horizontal. Negative is down.
    - Positive lift increases dip (tilts direction upward).
    - Positive drift turns clockwise in the horizontal plane.

    Implementation:
    - Lift: rotate toward +Z about axis = normalize(cross(v, +Z)).
    - Drift: rotate about -Z (clockwise).
    - Drift is interpreted as azimuth change per 100 m, independent of elevation.
    """
    v = _unit_vec_from_az_dip(az, dip)

    dtheta = np.deg2rad(lift_per100) * (d / 100.0)          # + lift -> up
    dpsi   = np.deg2rad(drift_per100) * (d / 100.0)         # + drift -> clockwise az

    z_hat = np.array([0.0, 0.0, 1.0])

    axis_lift = np.cross(v, z_hat)
    n_axis = np.linalg.norm(axis_lift)
    if n_axis < 1e-12:
        axis_lift = np.array([1.0, 0.0, 0.0])
    else:
        axis_lift /= n_axis

    omega = dtheta * axis_lift - dpsi * z_hat
    w = np.linalg.norm(omega)

    if w > 0:
        u = omega / w
        v_new = _rodrigues_rotate(v, u, w)
        v_new /= max(np.linalg.norm(v_new), 1e-12)
    else:
        v_new = v

    az_new, dip_new = _vector_to_az_dip(v_new)
    if np.hypot(v_new[0], v_new[1]) < 1e-8:
        az_new = az
    return az_new, dip_new

def min_curvature_path(stations):
    """
    Vector-based minimum curvature robust to pole wraps.
    """
    if not stations:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

    sta = sorted(stations, key=lambda d: float(d["MD"]))
    MDs = np.array([float(s["MD"]) for s in sta], dtype=float)
    vecs = [_unit_vec_from_az_dip(float(s["Azimuth"]), float(s["Angle"])) for s in sta]

    X, Y, Z = [0.0], [0.0], [0.0]
    for i in range(1, len(MDs)):
        dMD = MDs[i] - MDs[i - 1]
        if dMD <= 0:
            continue

        v1 = vecs[i - 1]
        v2 = vecs[i]

        cos_dog = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        dog = np.arccos(cos_dog)
        RF = 1.0 if dog < 1e-12 else (2.0 / dog) * np.tan(dog / 2.0)

        dR = 0.5 * dMD * (v1 + v2) * RF
        X.append(X[-1] + dR[0])
        Y.append(Y[-1] + dR[1])
        Z.append(Z[-1] + dR[2])

    return np.array(X), np.array(Y), np.array(Z), MDs

def make_planned_stations(length_m, step_m, az0, dip0, lift_per100, drift_per100):
    md = 0.0
    az = wrap_az(az0)
    dip = clamp(dip0, -90.0, 90.0)
    stations = [{"MD": md, "Azimuth": az, "Angle": dip}]
    while md < length_m - 1e-9:
        d = min(step_m, length_m - md)
        md += d
        az, dip = step_orientation(az, dip, d, lift_per100, drift_per100)
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def extend_actual(stations, to_depth, step_m, lift_per100, drift_per100):
    if not stations:
        return stations
    stations = sorted(stations, key=lambda d: float(d["MD"]))
    md = float(stations[-1]["MD"])
    az = float(stations[-1]["Azimuth"])
    dip = float(stations[-1]["Angle"])
    while md < to_depth - 1e-9:
        d = min(step_m, to_depth - md)
        md += d
        az, dip = step_orientation(az, dip, d, lift_per100, drift_per100)
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def trim_to_md(stations, target_md):
    if not stations:
        return stations
    sta = sorted(stations, key=lambda d: float(d["MD"]))
    if target_md >= sta[-1]["MD"] - 1e-9:
        return sta
    if target_md <= sta[0]["MD"] + 1e-9:
        s0 = sta[0]
        return [{"MD": float(target_md), "Azimuth": float(s0["Azimuth"]), "Angle": float(s0["Angle"])}]
    for j in range(1, len(sta)):
        md0, md1 = float(sta[j-1]["MD"]), float(sta[j]["MD"])
        if md0 <= target_md <= md1:
            t = (target_md - md0) / max(md1 - md0, 1e-12)
            az0 = float(sta[j-1]["Azimuth"])
            az1 = float(sta[j]["Azimuth"])
            azu = np.unwrap(np.deg2rad([az0, az1]))
            az_interp = np.rad2deg(azu[0] + t*(azu[1] - azu[0]))
            dip_interp = float(sta[j-1]["Angle"]) + t*(float(sta[j]["Angle"]) - float(sta[j-1]["Angle"]))
            trimmed = sta[:j]
            trimmed.append({"MD": float(target_md), "Azimuth": wrap_az(az_interp), "Angle": float(dip_interp)})
            return trimmed
    return sta

# ---------------- Pole-safe rate derivation helpers -------------------
def _polesafe_bearing_delta_rad(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Horizontal bearing change from v1 to v2.
    Positive return value means clockwise rotation.
    Near-vertical guard avoids noise when horizontal projection is tiny.
    """
    E1, N1 = v1[0], v1[1]
    E2, N2 = v2[0], v2[1]
    h1 = E1*E1 + N1*N1
    h2 = E2*E2 + N2*N2
    if h1 < 1e-10 or h2 < 1e-10:
        return 0.0
    cross_h = E1 * N2 - N1 * E2    # positive for CCW in EN plane
    dot_h = E1 * E2 + N1 * N2
    return float(-np.arctan2(cross_h, dot_h))  # sign flip so positive = clockwise

def _polesafe_elevation_deg(v: np.ndarray) -> float:
    return float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))

def derive_lift_drift_last3(stations):
    """
    Pole-safe estimate from the last 3 surveys.
    Returns (lift_deg_per_100m, drift_deg_per_100m) with positive drift = clockwise.
    """
    if len(stations) < 3:
        return None, None
    sta = sorted(stations, key=lambda d: float(d["MD"]))[-3:]
    MD = np.array([float(s["MD"]) for s in sta], dtype=float)

    vecs = [_unit_vec_from_az_dip(float(s["Azimuth"]), float(s["Angle"])) for s in sta]
    elev_deg = np.array([_polesafe_elevation_deg(v) for v in vecs], dtype=float)

    heading = [0.0]
    for i in range(1, len(vecs)):
        dpsi = _polesafe_bearing_delta_rad(vecs[i - 1], vecs[i])  # positive = clockwise
        heading.append(heading[-1] + np.rad2deg(dpsi))
    heading = np.array(heading, dtype=float)

    if not np.isfinite(np.ptp(MD)) or np.ptp(MD) < 1e-9:
        return None, None

    lift_deg_per_m = np.polyfit(MD, elev_deg, 1)[0]
    drift_deg_per_m = np.polyfit(MD, heading, 1)[0]
    return float(lift_deg_per_m * 100.0), float(drift_deg_per_m * 100.0)

def strike_dip_to_axes(strike_deg, dip_deg_signed):
    strike = np.deg2rad(wrap_az(strike_deg))
    dip_abs = np.deg2rad(abs(dip_deg_signed))
    s_hat = np.array([np.sin(strike), np.cos(strike), 0.0])  # along strike
    dipdir = strike + np.pi/2.0
    d_hat = np.array([np.sin(dipdir)*np.cos(dip_abs), np.cos(dipdir)*np.cos(dip_abs), -np.sin(dip_abs)])  # down-dip
    n_hat = np.cross(s_hat, d_hat)
    s_hat /= np.linalg.norm(s_hat)
    d_hat /= np.linalg.norm(d_hat)
    n_hat /= np.linalg.norm(n_hat)
    return s_hat, d_hat, n_hat

def segment_plane_intersection(p0, p1, P0, n_hat):
    u = p1 - p0
    denom = np.dot(n_hat, u)
    if abs(denom) < 1e-12:
        return None
    t = np.dot(n_hat, P0 - p0)/denom
    if 0.0 <= t <= 1.0:
        return p0 + t*u
    return None

def find_plane_intersection(points_xyz, P0, n_hat):
    for i in range(1, len(points_xyz)):
        p = segment_plane_intersection(points_xyz[i-1], points_xyz[i], P0, n_hat)
        if p is not None:
            return p
    return None

def parse_csv_flexible(file):
    # accepts variants: MD, Measured Depth, Azimuth/Azi/AZ, Angle/Dip/Inclination
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    c_md = pick("md", "measured depth", "measured_depth")
    c_az = pick("azimuth", "azi", "az")
    c_an = pick("angle", "dip", "inclination", "incl")
    if not all([c_md, c_az, c_an]):
        return pd.DataFrame(columns=["MD","Azimuth","Angle"])
    out = df[[c_md, c_az, c_an]].copy()
    out.columns = ["MD","Azimuth","Angle"]
    for c in ["MD","Azimuth","Angle"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["MD","Azimuth","Angle"])
    out["Azimuth"] = out["Azimuth"] % 360.0
    return out

def point_at_md(X, Y, Z, MDs, target_md):
    tmd = float(clamp(target_md, float(MDs.min()), float(MDs.max())))
    i = int(np.searchsorted(MDs, tmd))
    if i <= 0:
        return np.array([X[0], Y[0], Z[0]])
    if i >= len(MDs):
        return np.array([X[-1], Y[-1], Z[-1]])
    t = (tmd - MDs[i-1]) / (MDs[i] - MDs[i-1] + 1e-12)
    x = X[i-1] + t*(X[i] - X[i-1])
    y = Y[i-1] + t*(Y[i] - Y[i-1])
    z = Z[i-1] + t*(Z[i] - Z[i-1])
    return np.array([x, y, z])

def ensure_zero_station(stations, use_planned=False, plan_az0=None, plan_dip0=None):
    if not stations:
        return stations
    stations = sorted(stations, key=lambda d: float(d["MD"]))
    if stations[0]["MD"] <= 1e-9:
        return stations
    if use_planned and plan_az0 is not None and plan_dip0 is not None:
        az0, dip0 = float(plan_az0), float(plan_dip0)
    else:
        az0, dip0 = float(stations[0]["Azimuth"]), float(stations[0]["Angle"])
    stations.insert(0, {"MD": 0.0, "Azimuth": wrap_az(az0), "Angle": clamp(dip0, -90.0, 90.0)})
    return stations

def local_rates_per100(stations):
    """
    Pole-safe per-100 m lift and drift.
    - Lift: change in elevation angle derived from Z component.
    - Drift: change in horizontal bearing where positive = clockwise.
    """
    sta = sorted(stations, key=lambda d: float(d["MD"]))
    if len(sta) < 2:
        return np.array([]), np.array([]), np.array([])

    MD = np.array([float(s["MD"]) for s in sta], float)
    vecs = [_unit_vec_from_az_dip(float(s["Azimuth"]), float(s["Angle"])) for s in sta]

    dMD = MD[1:] - MD[:-1]
    elev1 = np.array([_polesafe_elevation_deg(v) for v in vecs[:-1]], float)
    elev2 = np.array([_polesafe_elevation_deg(v) for v in vecs[1:]], float)
    d_elev = elev2 - elev1

    d_bear = np.array([_polesafe_bearing_delta_rad(vecs[i], vecs[i+1]) for i in range(len(vecs) - 1)], float)
    d_bear_deg = np.rad2deg(d_bear)

    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(dMD != 0, d_elev / dMD * 100.0, 0.0)
        drift = np.where(dMD != 0, d_bear_deg / dMD * 100.0, 0.0)

    MDm = 0.5*(MD[1:] + MD[:-1])
    return MDm, lift, drift

def export_config(config_dict):
    return json.dumps(config_dict, indent=2)

# Common legend style with title watermark
LEGEND_TITLE = (
    "Drill-Hole-Deviation-Tracker<br>"
    "App by Evan Slater<br>"
    "https://drill-hole-deviation-tracker-r9xxcepvfewccfw8xgbfu3.streamlit.app/"
)
legend_style = dict(
    title=dict(text=LEGEND_TITLE),
    x=0.99, y=0.99, xanchor="right", yanchor="top",
    bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1
)

# ----------------------------------------------------------------------
# Import/Export UI
# ----------------------------------------------------------------------
with st.expander("Save or load session"):
    load_col, save_col = st.columns(2)
    with load_col:
        cfg_file = st.file_uploader("Load a saved session JSON", type=["json"], key="cfg_up")
        if st.button("Apply loaded session", use_container_width=True):
            if cfg_file is None:
                st.warning("Please choose a session JSON first.")
            else:
                try:
                    cfg = json.load(cfg_file)
                    st.session_state["pending_config"] = cfg
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not parse session JSON: {e}")
    with save_col:
        st.caption("Download a JSON snapshot of everything on this page")
        if "latest_export_json" in st.session_state:
            st.download_button(
                "Download current session JSON",
                data=st.session_state["latest_export_json"].encode("utf-8"),
                file_name="ddh_session.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("The save button will appear once the page has enough info to build a JSON.")

# ----------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------
st.title("Drill Hole vs Planned Hole Deviation Tracking")

# planned inputs
colA, colB, colC = st.columns(3)
with colA:
    plan_len = st.number_input("Planned hole length", value=st.session_state.plan_len, step=10.0, min_value=1.0, key="plan_len")
    step_m = st.number_input("Computation step m", value=st.session_state.step_m, step=1.0, min_value=1.0, key="step_m")
with colB:
    plan_az0 = st.number_input("Planned hole collar azimuth", value=st.session_state.plan_az0, step=1.0, key="plan_az0")
    plan_dip0 = st.number_input("Planned hole collar dip", value=st.session_state.plan_dip0, step=1.0, key="plan_dip0")
with colC:
    plan_lift = st.number_input("Planned lift deg/100m", value=st.session_state.plan_lift, step=0.1, key="plan_lift")
    plan_drift = st.number_input("Planned drift deg/100m", value=st.session_state.plan_drift, step=0.1, key="plan_drift")

# compute planned with vector-rotation stepping
planned_stations = make_planned_stations(
    st.session_state.plan_len,
    st.session_state.step_m,
    st.session_state.plan_az0,
    st.session_state.plan_dip0,
    st.session_state.plan_lift,
    st.session_state.plan_drift,
)
px, py, pz, pmd = min_curvature_path(planned_stations)
plan_pts = np.column_stack([px, py, pz])

# down hole surveys
st.subheader("Down hole surveys")
method = st.radio("Provide surveys via", ["CSV upload", "Manual entry"], horizontal=True)

if method == "CSV upload":
    file = st.file_uploader("Upload CSV (MD, Azimuth, Angle). Example headers accepted: MD, Azimuth, Angle or Measured Depth, Azi, Dip", type=["csv"], key="csv_up")
    flip_sign = st.checkbox("CSV angle is positive-down - flip sign to negative-down", value=False, key="flip_sign")
    if "loaded_surveys_df" in st.session_state:
        df_in = st.session_state["loaded_surveys_df"].copy()
    else:
        df_in = pd.DataFrame(columns=["MD","Azimuth","Angle"])
    if file is not None:
        df_in = parse_csv_flexible(file)
        if flip_sign:
            df_in["Angle"] = -df_in["Angle"]
        if not flip_sign and len(df_in) > 0 and df_in["Angle"].median() > 0:
            df_in["Angle"] = -df_in["Angle"]
            st.info("Detected positive-down dips in CSV. Converted to negative-down.")
    st.caption(f"Loaded {len(df_in)} survey rows")
    st.dataframe(df_in, use_container_width=True, height=350)
else:
    df_in = st.data_editor(
        pd.DataFrame(
            [
                {"MD": 0.0, "Azimuth": st.session_state.plan_az0, "Angle": st.session_state.plan_dip0},
                {"MD": 50.0, "Azimuth": st.session_state.plan_az0, "Angle": st.session_state.plan_dip0},
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="manual_df"
    )

actual_stations_base = [
    {"MD": float(r["MD"]), "Azimuth": wrap_az(float(r["Azimuth"])), "Angle": clamp(float(r["Angle"]), -90.0, 90.0)}
    for _, r in pd.DataFrame(df_in).dropna(subset=["MD","Azimuth","Angle"]).iterrows()
]

# Use Collar Az and Dip for 0
use_planned_zero = st.checkbox("Use Collar Az and Dip for 0", value=st.session_state.use_planned_zero, key="use_planned_zero")
actual_stations_base = ensure_zero_station(actual_stations_base, use_planned=use_planned_zero,
                                           plan_az0=st.session_state.plan_az0, plan_dip0=st.session_state.plan_dip0)

# suggested lift/drift from last 3
sug_lift, sug_drift = derive_lift_drift_last3(actual_stations_base) if len(actual_stations_base) >= 3 else (None, None)

st.markdown("#### Remaining average lift and drift after last survey")
colS1, colS2 = st.columns(2)
with colS1:
    st.caption(f"Suggested lift from last 3: {sug_lift:.2f} deg/100m" if sug_lift is not None else "Suggested lift needs at least 3 surveys")
with colS2:
    st.caption(f"Suggested drift from last 3: {sug_drift:.2f} deg/100m" if sug_drift is not None else "Suggested drift needs at least 3 surveys")

colR1, colR2, colR3 = st.columns(3)
with colR1:
    rem_lift = st.number_input("Remaining avg lift deg/100m", value=(sug_lift if sug_lift is not None else st.session_state.rem_lift), step=0.1, key="rem_lift")
with colR2:
    rem_drift = st.number_input("Remaining avg drift deg/100m", value=(sug_drift if sug_drift is not None else st.session_state.rem_drift), step=0.1, key="rem_drift")
with colR3:
    actual_len = st.number_input("Actual hole length", value=float(st.session_state.actual_len),
                                 step=10.0, min_value=0.0, key="actual_len")

# build actual to requested length: extend or trim
actual_stations = actual_stations_base.copy()
if actual_stations:
    last_md = sorted(actual_stations, key=lambda d: d["MD"])[-1]["MD"]
    target_len = float(st.session_state.actual_len)
    eps = 1e-9
    if last_md < target_len - eps:
        actual_stations = extend_actual(
            actual_stations, target_len, st.session_state.step_m,
            st.session_state.rem_lift, st.session_state.rem_drift
        )
    elif last_md > target_len + eps:
        actual_stations = trim_to_md(actual_stations, target_len)

ax, ay, az, amd = min_curvature_path(actual_stations) if actual_stations else (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
act_pts = np.column_stack([ax, ay, az])

# Target plane and pierce points
st.subheader("Target plane and pierce points")
colP1, colP2, colP3 = st.columns(3)
with colP1:
    plane_strike = st.number_input("Plane strike deg", value=st.session_state.plane_strike, step=1.0, key="plane_strike")
with colP2:
    plane_dip = st.number_input("Plane dip-from-horizontal deg (negative down)", value=st.session_state.plane_dip, step=1.0, key="plane_dip")
with colP3:
    default_md = float(st.session_state.target_md)
    target_md = st.number_input("Down hole target depth on planned hole", value=default_md, step=5.0,
                                min_value=0.0, max_value=float(st.session_state.plan_len), key="target_md")

s_hat, d_hat, n_hat = strike_dip_to_axes(plane_strike, plane_dip)
P0 = point_at_md(px, py, pz, pmd, target_md)

# pierce points
pierce_plan = find_plane_intersection(plan_pts, P0, n_hat)
pierce_act = find_plane_intersection(act_pts, P0, n_hat)

# 3D view with equal axes
st.markdown("### 3D view")
all_chunks = [plan_pts, act_pts, P0.reshape(1,3)]
if pierce_plan is not None:
    all_chunks.append(pierce_plan.reshape(1,3))
if pierce_act is not None:
    all_chunks.append(pierce_act.reshape(1,3))
ALL = np.vstack(all_chunks)

xmin, ymin, zmin = np.min(ALL, axis=0)
xmax, ymax, zmax = np.max(ALL, axis=0)
range_x = xmax - xmin
range_y = ymax - ymin
range_z = zmax - zmin
max_span = max(range_x, range_y, range_z, 1.0)

# symmetric equal ranges around centers
cx, cy, cz = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5
half = 0.5 * max_span
pad = max(0.25 * max_span, 25.0)
L = half + pad
xr = [cx - L, cx + L]
yr = [cy - L, cy + L]
zr = [cz - L, cz + L]

# plane mesh sized to the same cube
span = 2.0 * L
uu, vv = np.meshgrid(np.linspace(-span, span, 30), np.linspace(-span, span, 30))
plane_grid = P0.reshape(1,1,3) + uu[...,None]*s_hat.reshape(1,1,3) + vv[...,None]*d_hat.reshape(1,1,3)

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", name="Planned", line=dict(width=6)))
fig3d.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode="lines", name="Actual", line=dict(width=6)))
fig3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Collar", marker=dict(size=5)))
fig3d.add_trace(go.Surface(
    x=plane_grid[...,0],
    y=plane_grid[...,1],
    z=plane_grid[...,2],
    opacity=0.35, showscale=False, name="Target plane"
))
if pierce_plan is not None:
    fig3d.add_trace(go.Scatter3d(
        x=[pierce_plan[0]], y=[pierce_plan[1]], z=[pierce_plan[2]],
        mode="markers", name="Pierce planned", marker=dict(size=6, symbol="x")
    ))
if pierce_act is not None:
    fig3d.add_trace(go.Scatter3d(
        x=[pierce_act[0]], y=[pierce_act[1]], z=[pierce_act[2]],
        mode="markers", name="Pierce actual", marker=dict(size=6)
    ))
if pierce_plan is not None and pierce_act is not None:
    fig3d.add_trace(go.Scatter3d(
        x=[pierce_plan[0], pierce_act[0]],
        y=[pierce_plan[1], pierce_act[1]],
        z=[pierce_plan[2], pierce_act[2]],
        mode="lines", name="Pierce separation", line=dict(width=3, dash="dash")
    ))
    v = pierce_act - pierce_plan
    v_plane = v - np.dot(v, n_hat)*n_hat
    dist_on_plane = float(np.linalg.norm(v_plane))
    st.info(f"Pierce separation on plane: {dist_on_plane:.2f} m")

fig3d.update_layout(
    scene=dict(
        xaxis_title="X East m", xaxis=dict(range=xr),
        yaxis_title="Y North m", yaxis=dict(range=yr),
        zaxis_title="Z m (up)", zaxis=dict(range=zr),
        aspectmode="cube",
        aspectratio=dict(x=1, y=1, z=1)
    ),
    scene_camera=dict(
        eye=dict(x=1.8, y=1.8, z=1.2),
        projection=dict(type="orthographic")
    ),
    legend=legend_style,
    margin=dict(l=0, r=0, b=0, t=0)
)
st.plotly_chart(fig3d, use_container_width=True)

# Orientation plot: Azimuth and dip vs measured depth (actual hole) - ranges expand to fit data
st.markdown("### Azimuth and dip vs measured depth (actual hole)")
fig_orient = go.Figure()
if actual_stations:
    sta = sorted(actual_stations, key=lambda d: float(d["MD"]))
    MD = [float(s["MD"]) for s in sta]
    DIP = [float(s["Angle"]) for s in sta]
    AZ = [wrap_az(float(s["Azimuth"])) for s in sta]
    AZu = np.rad2deg(np.unwrap(np.deg2rad(AZ)))

    fig_orient.add_trace(go.Scatter(x=MD, y=DIP, mode="lines+markers", name="Dip deg (negative down)", yaxis="y1"))
    fig_orient.add_trace(go.Scatter(x=MD, y=AZu, mode="lines+markers", name="Azimuth deg (unwrapped)", yaxis="y2"))

    # compute padded ranges that include all data
    dip_min, dip_max = float(np.min(DIP)), float(np.max(DIP))
    az_min, az_max = float(np.min(AZu)), float(np.max(AZu))
    pad = 2.0
    dip_range = [max(-90.0, dip_min - pad), min(90.0, dip_max + pad)]
    az_range = [az_min - pad, az_max + pad]
else:
    MD, DIP, AZu = [], [], []
    dip_range = [-90, 90]
    az_range = [0, 360]

fig_orient.update_layout(
    margin=dict(l=0, r=0, b=0, t=10),
    xaxis=dict(title="Measured depth along actual hole m"),
    yaxis=dict(
        title="Dip deg",
        range=dip_range,
        showgrid=True,
        gridcolor="#d0d0d0",
        zeroline=True,
        zerolinecolor="#a0a0a0",
        layer="below traces"
    ),
    yaxis2=dict(
        title="Azimuth deg",
        overlaying="y",
        side="right",
        range=az_range,
        showgrid=True,
        gridcolor="#e6f2ff",
        zeroline=True,
        zerolinecolor="#99ccff",
        layer="below traces"
    ),
    legend=legend_style
)
st.plotly_chart(fig_orient, use_container_width=True)

# Per 100 m deviation plot
st.markdown("### Per 100 m deviation along actual hole")
MDm, lift_series, drift_series = local_rates_per100(actual_stations) if actual_stations else (np.array([]), np.array([]), np.array([]))
fig_rate = go.Figure()
if len(MDm) > 0:
    fig_rate.add_trace(go.Scatter(x=MDm, y=lift_series, mode="lines+markers", name="Lift deg/100m"))
    fig_rate.add_trace(go.Scatter(x=MDm, y=drift_series, mode="lines+markers", name="Drift deg/100m"))
fig_rate.update_layout(
    xaxis_title="Measured depth along actual hole m",
    yaxis_title="deg per 100 m",
    legend=legend_style,
    margin=dict(l=0, r=0, b=0, t=10)
)
st.plotly_chart(fig_rate, use_container_width=True)

# Export session JSON
cfg_dict = {
    "planned": {
        "plan_len": float(st.session_state.plan_len),
        "step_m": float(st.session_state.step_m),
        "plan_az0": float(st.session_state.plan_az0),
        "plan_dip0": float(st.session_state.plan_dip0),
        "plan_lift": float(st.session_state.plan_lift),
        "plan_drift": float(st.session_state.plan_drift),
    },
    "surveys": pd.DataFrame(df_in).to_dict(orient="records"),
    "rem_lift": float(st.session_state.rem_lift),
    "rem_drift": float(st.session_state.rem_drift),
    "use_planned_zero": bool(st.session_state.use_planned_zero),
    "actual_len": float(st.session_state.actual_len),
    "plane": {
        "strike": float(st.session_state.plane_strike),
        "dip": float(st.session_state.plane_dip),
        "target_md": float(st.session_state.target_md),
    }
}
cfg_json = export_config(cfg_dict)

# keep a copy so the top expander can offer a save button even after import
st.session_state["latest_export_json"] = cfg_json

with st.expander("Export session", expanded=False):
    st.download_button(
        "Download session JSON",
        data=cfg_json.encode("utf-8"),
        file_name="ddh_session.json",
        mime="application/json",
        use_container_width=True
    )

st.caption("Angles are dip-from-horizontal (negative down). Positive lift tilts up. Positive drift turns clockwise. Vector-rotation stepping and equal 3D axes ensure constant-lift and constant-drift arcs are circles that loop through +/-90 smoothly.")
