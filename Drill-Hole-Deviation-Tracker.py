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
    # stations: list of dicts with MD, Azimuth, Angle (dip-from-horizontal, negative = down)
    if not stations:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

    stations = sorted(stations, key=lambda d: float(d["MD"]))
    MDs = np.array([float(s["MD"]) for s in stations], float)
    AZs = np.deg2rad([wrap_az(float(s["Azimuth"])) for s in stations])
    DIP = np.array([float(s["Angle"]) for s in stations], float)

    # Inclination from vertical for geometry. Sign handled separately for vertical component.
    INC = np.deg2rad(90.0 - np.abs(DIP))
    SGN = np.where(DIP <= 0.0, -1.0, 1.0)  # negative dip means down

    X, Y, Z = [0.0], [0.0], [0.0]
    for i in range(1, len(MDs)):
        dMD = MDs[i] - MDs[i-1]
        if dMD <= 0:
            continue
        inc1, inc2 = INC[i-1], INC[i]
        az1, az2 = AZs[i-1], AZs[i]
        s1, s2 = SGN[i-1], SGN[i]

        cos_dog = np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1) + np.cos(inc1)*np.cos(inc2)
        cos_dog = float(np.clip(cos_dog, -1.0, 1.0))
        dog = np.arccos(cos_dog)
        RF = 1.0 if dog < 1e-12 else (2.0/dog)*np.tan(dog/2.0)

        dN = 0.5*dMD*(np.sin(inc1)*np.cos(az1) + np.sin(inc2)*np.cos(az2))*RF
        dE = 0.5*dMD*(np.sin(inc1)*np.sin(az1) + np.sin(inc2)*np.sin(az2))*RF
        dZ = 0.5*dMD*(s1*np.cos(inc1) + s2*np.cos(inc2))*RF  # signed vertical

        X.append(X[-1] + dE)
        Y.append(Y[-1] + dN)
        Z.append(Z[-1] + dZ)
    return np.array(X), np.array(Y), np.array(Z), MDs

def make_planned_stations(length_m, step_m, az0, dip0, lift_per100, drift_per100):
    # negative lift makes the hole come up toward 0 dip
    md = 0.0
    az = wrap_az(az0)
    dip = clamp(dip0, -90.0, 90.0)
    stations = [{"MD": md, "Azimuth": az, "Angle": dip}]
    while md < length_m - 1e-9:
        d = min(step_m, length_m - md)
        md += d
        az = wrap_az(az + drift_per100*(d/100.0))
        # minus because positive lift increases downward magnitude, so to apply user lift correctly we subtract
        dip = clamp(dip - lift_per100*(d/100.0), -90.0, 90.0)
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
        az = wrap_az(az + drift_per100*(d/100.0))
        dip = clamp(dip - lift_per100*(d/100.0), -90.0, 90.0)
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def derive_lift_drift_last3(stations):
    if len(stations) < 3:
        return None, None
    sta = sorted(stations, key=lambda d: float(d["MD"]))[-3:]
    MD = np.array([float(s["MD"]) for s in sta], float)
    AZ = np.array([float(s["Azimuth"]) for s in sta], float)
    DIP = np.array([float(s["Angle"]) for s in sta], float)
    AZu = np.unwrap(np.deg2rad(AZ))
    drift_deg_per_m = np.rad2deg(np.polyfit(MD, AZu, 1)[0])
    # minus because positive lift increases downward magnitude, we want lift positive when trend is toward more negative dips
    lift_deg_per_m = -np.polyfit(MD, DIP, 1)[0]
    return float(lift_deg_per_m*100.0), float(drift_deg_per_m*100.0)

def strike_dip_to_axes(strike_deg, dip_deg_signed):
    # X East, Y North, Z Up. Strike cw from North. Dip-from-horizontal, negative down.
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

def parse_df_flexible(df):
    # Accept common variants: MD, Measured Depth, Azimuth/Azi/AZ, Angle/Dip/Inclination
    if df is None or df.empty:
        return pd.DataFrame(columns=["MD","Azimuth","Angle"])
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
    out = out.dropna(subset=["MD","Azimuth","Angle"]).sort_values("MD").reset_index(drop=True)
    return out

def point_at_md(X, Y, Z, MDs, target_md):
    # linear interpolation along the computed path
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

def az_diff_deg(a2, a1):
    # smallest signed difference a2 - a1 in degrees
    d = np.deg2rad(a2) - np.deg2rad(a1)
    d = np.arctan2(np.sin(d), np.cos(d))
    return np.rad2deg(d)

def add_delta_columns(df):
    if df is None or df.empty:
        return df
    df = df.sort_values("MD").reset_index(drop=True).copy()
    df["dMD"] = df["MD"].diff()
    df["dAz_deg"] = df["Azimuth"].diff().fillna(0.0)
    # correct az wrap per interval
    df.loc[1:, "dAz_deg"] = [
        az_diff_deg(df.loc[i, "Azimuth"], df.loc[i-1, "Azimuth"])
        for i in range(1, len(df))
    ]
    df["dDip_deg"] = df["Angle"].diff()
    # per 100 m, guard zero or negative dMD
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Az change deg/100m"] = np.where(df["dMD"] > 0, df["dAz_deg"]/df["dMD"]*100.0, np.nan)
        df["Dip change deg/100m"] = np.where(df["dMD"] > 0, df["dDip_deg"]/df["dMD"]*100.0, np.nan)
    return df.drop(columns=["dMD", "dAz_deg", "dDip_deg"])

# ---------- UI ----------
st.title("Drillhole vs Planned - single 3D view")

# planned inputs - defaults now positive per request
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
plan_pts = np.column_stack([px, py, pz])

# actual surveys
st.subheader("Actual surveys")
method = st.radio("Provide surveys via", ["Excel upload", "CSV upload", "Manual entry"], horizontal=True)

df_in = pd.DataFrame(columns=["MD","Azimuth","Angle"])
flip_sign = False
if method == "Excel upload":
    file = st.file_uploader("Upload Excel workbook", type=["xlsx", "xls"])
    if file is not None:
        try:
            xls = pd.ExcelFile(file)
            sheet = st.selectbox("Pick worksheet", xls.sheet_names, index=0)
            df_raw = pd.read_excel(xls, sheet_name=sheet)
            flip_sign = st.checkbox("Angle in file is positive-down - flip to negative-down", value=False)
            df_in = parse_df_flexible(df_raw)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
elif method == "CSV upload":
    file = st.file_uploader("Upload CSV (MD, Azimuth, Angle). Example headers accepted: MD, Azimuth, Angle or Measured Depth, Azi, Dip", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
            flip_sign = st.checkbox("CSV angle is positive-down - flip sign to negative-down", value=False)
            df_in = parse_df_flexible(df_raw)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
else:
    df_in = st.data_editor(
        pd.DataFrame(
            [
                {"MD": 0.0, "Azimuth": plan_az0, "Angle": plan_dip0},
                {"MD": 50.0, "Azimuth": plan_az0, "Angle": plan_dip0},
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
    )

# auto or manual sign flip
if not df_in.empty:
    if flip_sign:
        df_in["Angle"] = -df_in["Angle"]
    elif df_in["Angle"].median() > 0:
        df_in["Angle"] = -df_in["Angle"]
        st.info("Detected positive-down dips. Converted to negative-down.")

# manual collar always available
st.markdown("#### Actual collar")
default_collar_az = float(df_in.iloc[0]["Azimuth"]) if not df_in.empty else plan_az0
default_collar_dip = float(df_in.iloc[0]["Angle"]) if not df_in.empty else plan_dip0
collar_az = st.number_input("Actual collar azimuth deg", value=float(default_collar_az), step=1.0)
collar_dip = st.number_input("Actual collar dip-from-horizontal deg (negative down)", value=float(default_collar_dip), step=1.0)
force_manual_collar = st.checkbox("Force manual collar at MD 0", value=True)

# show table with az and dip change per 100 m
if not df_in.empty:
    df_show = add_delta_columns(df_in)
    st.caption(f"Loaded {len(df_in)} survey rows")
    st.dataframe(df_show, use_container_width=True)
else:
    st.caption("No surveys loaded")

# prepare stations list
actual_stations_base = [
    {"MD": float(r["MD"]), "Azimuth": float(r["Azimuth"]), "Angle": float(r["Angle"])}
    for _, r in pd.DataFrame(df_in).dropna(subset=["MD","Azimuth","Angle"]).iterrows()
]
actual_stations_base = sorted(actual_stations_base, key=lambda d: float(d["MD"]))

# enforce or override MD 0
if force_manual_collar:
    if actual_stations_base and actual_stations_base[0]["MD"] <= 1e-9:
        # replace first row values
        actual_stations_base[0]["Azimuth"] = float(collar_az)
        actual_stations_base[0]["Angle"] = float(collar_dip)
    else:
        # insert MD 0
        actual_stations_base.insert(0, {"MD": 0.0, "Azimuth": float(collar_az), "Angle": float(collar_dip)})
else:
    # if not forcing, still ensure a 0 station exists, using file start or planned as fallback
    if actual_stations_base:
        if actual_stations_base[0]["MD"] > 1e-9:
            # seed from first survey in file
            az0, dip0 = float(actual_stations_base[0]["Azimuth"]), float(actual_stations_base[0]["Angle"])
            actual_stations_base.insert(0, {"MD": 0.0, "Azimuth": az0, "Angle": dip0})

# suggested lift/drift from last 3
sug_lift, sug_drift = derive_lift_drift_last3(actual_stations_base) if len(actual_stations_base) >= 3 else (None, None)

st.markdown("#### Remaining average lift and drift after last survey")
colS1, colS2 = st.columns(2)
with colS1:
    st.caption(f"Suggested lift from last 3: {sug_lift:.2f} deg/100m" if sug_lift is not None else "Suggested lift needs at least 3 surveys")
with colS2:
    st.caption(f"Suggested drift from last 3: {sug_drift:.2f} deg/100m" if sug_drift is not None else "Suggested drift needs at least 3 surveys")

colR1, colR2 = st.columns(2)
with colR1:
    rem_lift = st.number_input("Remaining avg lift deg/100m", value=(sug_lift if sug_lift is not None else 2.0), step=0.1)
with colR2:
    rem_drift = st.number_input("Remaining avg drift deg/100m", value=(sug_drift if sug_drift is not None else 1.0), step=0.1)

# extension toggle
extend_to_plan = st.checkbox("Extend actual to planned length", value=True)

# extend actual to planned length if chosen
actual_stations = actual_stations_base.copy()
if actual_stations and extend_to_plan:
    last_md = sorted(actual_stations, key=lambda d: d["MD"])[-1]["MD"]
    if plan_len > last_md + 1e-6:
        actual_stations = extend_actual(actual_stations, plan_len, step_m, rem_lift, rem_drift)

ax, ay, az, amd = min_curvature_path(actual_stations) if actual_stations else (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
act_pts = np.column_stack([ax, ay, az])

# plane defined by strike and dip, positioned by downhole MD on planned hole
st.subheader("Target plane and pierce points")
colP1, colP2, colP3 = st.columns(3)
with colP1:
    plane_strike = st.number_input("Plane strike deg", value=114.0, step=1.0)
with colP2:
    plane_dip = st.number_input("Plane dip-from-horizontal deg (negative down)", value=-58.0, step=1.0)
with colP3:
    default_md = max(0.0, plan_len - 50.0)
    target_md = st.number_input("Target downhole MD on planned hole m", value=float(default_md), step=5.0, min_value=0.0, max_value=float(plan_len))

s_hat, d_hat, n_hat = strike_dip_to_axes(plane_strike, plane_dip)
P0 = point_at_md(px, py, pz, pmd, target_md)  # plane passes through planned hole at target MD

# pierce points
pierce_plan = find_plane_intersection(plan_pts, P0, n_hat)
pierce_act = find_plane_intersection(act_pts, P0, n_hat)

# ---------- 3D view with enlarged bounds and plane to edges ----------
st.markdown("### 3D view")

# bounds
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
pad = max(0.25*max_span, 25.0)

xr = [xmin - pad, xmax + pad]
yr = [ymin - pad, ymax + pad]
zr = [zmin - pad, zmax + pad]

# plane sized to cover entire axes
cube_diag = np.linalg.norm([xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]])
span = cube_diag
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
    opacity=0.35,
    showscale=False,
    name="Target plane"
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
    # distance within plane (project connector onto plane)
    v = pierce_act - pierce_plan
    v_plane = v - np.dot(v, n_hat)*n_hat
    dist_on_plane = float(np.linalg.norm(v_plane))
    st.info(f"Pierce separation on plane: {dist_on_plane:.2f} m")

fig3d.update_layout(
    scene=dict(
        xaxis_title="X East m", xaxis=dict(range=xr),
        yaxis_title="Y North m", yaxis=dict(range=yr),
        zaxis_title="Z m (up)", zaxis=dict(range=zr),
        aspectmode="cube"
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    scene_camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
)
st.plotly_chart(fig3d, use_container_width=True)

st.caption("Angles are dip-from-horizontal. Negative values point down. Target plane is positioned by downhole MD on the planned hole. If your file does not start at 0 m, a 0 m station is inserted or overridden by the manual collar, depending on the toggle.")
