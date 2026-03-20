import streamlit as st
import pandas as pd
from dataclasses import dataclass
from math import radians, sin, cos, atan2, sqrt, degrees
from typing import Dict, List, Tuple, Optional
import bisect

# -----------------------------
# CATALINA 36 POLAR TABLE
# Representative of TSC fleet (avg PHRF ~150)
# Rows: TWA (degrees), Cols: wind speed (knots) → boat speed (knots)
# -----------------------------
_POLAR_TWA   = [52, 60, 75, 90, 110, 120, 135, 150]
_POLAR_TWS   = [6, 8, 10, 12, 14, 16, 20]
_POLAR_SPEED = [
    # TWS:   6     8     10    12    14    16    20
    [4.81, 5.67, 6.27, 6.63, 6.83, 6.93, 7.00],  # 52°
    [5.10, 5.94, 6.50, 6.84, 7.04, 7.16, 7.25],  # 60°
    [5.28, 6.14, 6.69, 7.02, 7.24, 7.41, 7.61],  # 75°
    [5.20, 6.15, 6.81, 7.17, 7.37, 7.51, 7.84],  # 90°
    [5.28, 6.34, 6.98, 7.37, 7.67, 7.93, 8.30],  # 110°
    [5.13, 6.22, 6.92, 7.34, 7.67, 7.98, 8.54],  # 120°
    [4.64, 5.75, 6.60, 7.11, 7.48, 7.80, 8.41],  # 135°
    [3.96, 5.01, 5.88, 6.51, 6.90, 7.20, 7.68],  # 150°
]

def _interp1(xs: list, ys: list, x: float) -> float:
    """Linear interpolation; clamps to endpoints."""
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + t * (ys[i+1] - ys[i])

def polar_boatspeed(twa_deg: float, tws_kt: float) -> float:
    """
    Bilinear interpolation into Catalina 36 polar.
    twa_deg: True Wind Angle, 0–180° (absolute, symmetric port/stbd).
    tws_kt: True Wind Speed in knots.
    """
    twa = abs(twa_deg)
    twa = min(max(twa, _POLAR_TWA[0]), _POLAR_TWA[-1])
    if twa <= _POLAR_TWA[0]:
        return _interp1(_POLAR_TWS, _POLAR_SPEED[0], tws_kt)
    if twa >= _POLAR_TWA[-1]:
        return _interp1(_POLAR_TWS, _POLAR_SPEED[-1], tws_kt)
    i = bisect.bisect_right(_POLAR_TWA, twa) - 1
    spd_lo = _interp1(_POLAR_TWS, _POLAR_SPEED[i],   tws_kt)
    spd_hi = _interp1(_POLAR_TWS, _POLAR_SPEED[i+1], tws_kt)
    t = (twa - _POLAR_TWA[i]) / (_POLAR_TWA[i+1] - _POLAR_TWA[i])
    return spd_lo + t * (spd_hi - spd_lo)

def vmg_upwind_speed(tws_kt: float) -> Tuple[float, float]:
    """
    Scan TWA 30–60° and return (best_twa, VMG_component) for upwind.
    VMG_upwind = boatspeed * cos(twa)  (component toward wind).
    """
    best_vmg, best_twa = 0.0, 52.0
    for twa in range(30, 61):
        bs = polar_boatspeed(twa, tws_kt)
        vmg = bs * cos(radians(twa))
        if vmg > best_vmg:
            best_vmg, best_twa = vmg, float(twa)
    return best_twa, best_vmg

def vmg_downwind_speed(tws_kt: float) -> Tuple[float, float]:
    """
    Scan TWA 120–160° and return (best_twa, VMG_component) for downwind.
    VMG_downwind = boatspeed * cos(180° - twa).
    """
    best_vmg, best_twa = 0.0, 150.0
    for twa in range(120, 161):
        bs = polar_boatspeed(twa, tws_kt)
        vmg = bs * cos(radians(180 - twa))
        if vmg > best_vmg:
            best_vmg, best_twa = vmg, float(twa)
    return best_twa, best_vmg

def smallest_angle_deg(x: float) -> float:
    return abs((x + 180.0) % 360.0 - 180.0)

def effective_speed_on_bearing(bearing_deg: float, wind_from_deg: float, tws_kt: float) -> Tuple[str, float]:
    wind_to = (wind_from_deg + 180.0) % 360.0
    twa = smallest_angle_deg(bearing_deg - wind_to)   # 0 = dead downwind, 180 = dead upwind

    best_up_twa, _ = vmg_upwind_speed(tws_kt)
    best_dn_twa, _ = vmg_downwind_speed(tws_kt)

    if twa > 180.0 - best_up_twa:        # upwind cone
        _, vmg = vmg_upwind_speed(tws_kt)
        return "UPWIND (VMG)", max(0.5, vmg)
    if twa < 180.0 - best_dn_twa:        # downwind cone
        _, vmg = vmg_downwind_speed(tws_kt)
        return "DOWNWIND (VMG)", max(0.5, vmg)

    # Reach — use polar boatspeed at actual TWA
    bs = polar_boatspeed(180.0 - twa, tws_kt)
    if twa > 120:
        label = "UPWIND-ish"
    elif twa < 60:
        label = "DOWNWIND-ish"
    else:
        label = "REACH"
    return label, max(0.5, bs)

# -----------------------------
# MARKS (lat/lon in decimal degrees)
# -----------------------------
MARKS: Dict[str, Tuple[float, float]] = {
    "BB":   (33 + 51.700/60.0, -(96 + 38.630/60.0)),
    "TSC1": (33 + 52.070/60.0, -(96 + 37.900/60.0)),
    "TSC3": (33 + 53.790/60.0, -(96 + 37.780/60.0)),
    "PNJ":  (33 + 51.700/60.0, -(96 + 35.940/60.0)),
    "WE":   (33 + 49.300/60.0, -(96 + 36.000/60.0)),
    "WT":   (33 + 54.723/60.0, -(96 + 35.790/60.0)),
    "CC":   (33 + 55.410/60.0, -(96 + 42.790/60.0)),
    "TT":   (33 + 51.790/60.0, -(96 + 39.190/60.0)),
}

EARTH_RADIUS_NM = 3440.065

def haversine_nm(a: str, b: str) -> float:
    lat1, lon1 = MARKS[a]; lat2, lon2 = MARKS[b]
    phi1, lam1 = radians(lat1), radians(lon1)
    phi2, lam2 = radians(lat2), radians(lon2)
    dphi, dlam = phi2 - phi1, lam2 - lam1
    h = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return 2 * EARTH_RADIUS_NM * atan2(sqrt(h), sqrt(1-h))

def initial_bearing_deg(a: str, b: str) -> float:
    lat1, lon1 = MARKS[a]; lat2, lon2 = MARKS[b]
    phi1, lam1 = radians(lat1), radians(lon1)
    phi2, lam2 = radians(lat2), radians(lon2)
    dlam = lam2 - lam1
    x = sin(dlam)*cos(phi2)
    y = cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dlam)
    return (degrees(atan2(x, y)) + 360.0) % 360.0

# -----------------------------
# COURSE GRAPH
# -----------------------------
@dataclass(frozen=True)
class Edge:
    name: str
    start: str
    end: str
    path: Tuple[str, ...]

    def expanded_legs(self): return [(self.path[i], self.path[i+1]) for i in range(len(self.path)-1)]

def is_corridor(e: Edge): return len(e.path) > 2

CORRIDORS: List[Edge] = [
    Edge("CC->BB corridor",  "CC",  "BB",  ("CC","TSC3","TSC1","BB")),
    Edge("BB->CC corridor",  "BB",  "CC",  ("BB","TSC1","TSC3","CC")),
    Edge("PNJ->CC corridor", "PNJ", "CC",  ("PNJ","TSC3","CC")),
    Edge("CC->PNJ corridor", "CC",  "PNJ", ("CC","TSC3","PNJ")),
    Edge("PNJ->BB corridor", "PNJ", "BB",  ("PNJ","TSC1","BB")),
    Edge("BB->PNJ corridor", "BB",  "PNJ", ("BB","TSC1","PNJ")),
    Edge("WT->BB corridor",  "WT",  "BB",  ("WT","TSC1","BB")),
    Edge("BB->WT corridor",  "BB",  "WT",  ("BB","TSC1","WT")),
    Edge("WE->BB corridor",  "WE",  "BB",  ("WE","TSC1","BB")),
    Edge("BB->WE corridor",  "BB",  "WE",  ("BB","TSC1","WE")),
    Edge("CC->WE corridor",  "CC",  "WE",  ("CC","TSC3","WE")),
    Edge("WE->CC corridor",  "WE",  "CC",  ("WE","TSC3","CC")),
    Edge("TT->PNJ corridor", "TT",  "PNJ", ("TT","TSC1","PNJ")),
    Edge("PNJ->TT corridor", "PNJ", "TT",  ("PNJ","TSC1","TT")),
]

SIMPLE_EDGES: List[Edge] = [
    Edge("WE->PNJ", "WE",  "PNJ", ("WE","PNJ")),
    Edge("PNJ->WE", "PNJ", "WE",  ("PNJ","WE")),
    Edge("PNJ->WT", "PNJ", "WT",  ("PNJ","WT")),
    Edge("WT->PNJ", "WT",  "PNJ", ("WT","PNJ")),
    Edge("WE->WT",  "WE",  "WT",  ("WE","WT")),
    Edge("WT->WE",  "WT",  "WE",  ("WT","WE")),
    Edge("WT->CC",  "WT",  "CC",  ("WT","CC")),
    Edge("CC->WT",  "CC",  "WT",  ("CC","WT")),
    Edge("TT->BB",  "TT",  "BB",  ("TT","BB")),
    Edge("BB->TT",  "BB",  "TT",  ("BB","TT")),
]

ALL_EDGES: List[Edge] = SIMPLE_EDGES + CORRIDORS
ADJ: Dict[str, List[Edge]] = {}
for e in ALL_EDGES:
    ADJ.setdefault(e.start, []).append(e)

# -----------------------------
# TIME MODEL (polar-based)
# -----------------------------
def edge_time(edge: Edge, wind_from_deg: float, wind_speed_kt: float):
    total_dist, total_time = 0.0, 0.0
    details = []
    for a, b in edge.expanded_legs():
        dist = haversine_nm(a, b)
        brg  = initial_bearing_deg(a, b)
        ltype, spd = effective_speed_on_bearing(brg, wind_from_deg, wind_speed_kt)
        t = dist / spd
        total_dist += dist; total_time += t
        details.append((a, b, dist, brg, ltype, spd, t))
    return total_dist, total_time, details

# -----------------------------
# SCORING + SEARCH
# -----------------------------
@dataclass
class Candidate:
    edges: List[Edge]
    total_time: float
    total_dist: float
    score: float

def edge_score(edge: Edge, wind_from_deg: float, wind_speed_kt: float,
               prev_edge: Optional[Edge]) -> float:
    dist, t, legs = edge_time(edge, wind_from_deg, wind_speed_kt)
    score = 0.0

    for (_, _, d, _, ltype, _, _) in legs:
        if wind_speed_kt >= 8.0:
            if "UPWIND" in ltype or "DOWNWIND" in ltype:
                score += 2.0 * d
            # REACH is neutral — no bonus, no penalty
        else:
            if ltype == "REACH":
                score += 2.0 * d
            elif "DOWNWIND" in ltype:
                score += 0.6 * d
            # light-air upwind is neutral

    if dist < 0.6:
        score -= 3.0

    if prev_edge is not None:
        if prev_edge.end == edge.end:
            score -= 2.0

    if is_corridor(edge):
        score -= 0.8

    return score

def recommend_courses(
    wind_from_deg: float, wind_speed_kt: float,
    hard_cap_hr: float = 3.0,
    target_window: Tuple[float, float] = (2.00, 2.50),
    min_edges: int = 2, max_edges: int = 7, top_k: int = 10
) -> List[Candidate]:
    best: List[Candidate] = []

    def consider(c):
        best.append(c); best.sort(key=lambda x: x.score, reverse=True)
        del best[top_k:]

    def dfs(current, edges, t_hr, d_nm, score):
        if t_hr > hard_cap_hr or len(edges) > max_edges: return
        if current == "BB" and len(edges) >= min_edges:
            lo, hi = target_window; center = (lo+hi)/2
            bonus = 10.0 if lo <= t_hr <= hi else -abs(t_hr - center) * 6.0
            consider(Candidate(list(edges), t_hr, d_nm, score + bonus))
        prev = edges[-1] if edges else None
        for e in ADJ.get(current, []):
            if len(edges) < 1 and e.end == "BB": continue
            edist, etime, _ = edge_time(e, wind_from_deg, wind_speed_kt)
            escore = edge_score(e, wind_from_deg, wind_speed_kt, prev)
            dfs(e.end, edges+[e], t_hr+etime, d_nm+edist, score+escore)

    dfs("BB", [], 0.0, 0.0, 0.0)
    return best

def nodes_macro(edges, start="BB"):
    out = [start]
    for e in edges: out.append(e.end)
    return out

def nodes_expanded(edges, start="BB"):
    out = [start]
    for e in edges:
        for n in e.path[1:]: out.append(n)
    return out

# -----------------------------
# UI
# -----------------------------
COMPASS_TO_DEG = {
    "N":0,"NNE":22.5,"NE":45,"ENE":67.5,
    "E":90,"ESE":112.5,"SE":135,"SSE":157.5,
    "S":180,"SSW":202.5,"SW":225,"WSW":247.5,
    "W":270,"WNW":292.5,"NW":315,"NNW":337.5,
}

def nearest_compass(deg):
    deg = deg % 360
    return min(COMPASS_TO_DEG, key=lambda k: min(abs(COMPASS_TO_DEG[k]-deg), 360-abs(COMPASS_TO_DEG[k]-deg)))

st.set_page_config(page_title="TSC Course Generator", layout="centered")
st.title("TSC Fixed-Marks Course Generator")
st.caption("Catalina 36 polar model · representative of TSC fleet · start/finish BB · target 2–2.5 hr")

if "wind_deg" not in st.session_state: st.session_state.wind_deg = 200.0
if "wind_dir" not in st.session_state: st.session_state.wind_dir = nearest_compass(200.0)

def on_dir_change(): st.session_state.wind_deg = float(COMPASS_TO_DEG[st.session_state.wind_dir])
def on_deg_change():
    st.session_state.wind_deg = float(st.session_state.wind_deg) % 360
    st.session_state.wind_dir = nearest_compass(st.session_state.wind_deg)

col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
with col1:
    st.selectbox("Wind direction (compass)", list(COMPASS_TO_DEG), key="wind_dir", on_change=on_dir_change)
with col2:
    st.number_input("Wind direction (°)", 0.0, 359.9, step=1.0, key="wind_deg", on_change=on_deg_change)
with col3:
    wind_speed = st.number_input("Wind speed (kt)", 0.0, 60.0, value=15.0, step=0.5)
with col4:
    top_n = st.number_input("Top N courses", 1, 15, value=5, step=1)

wd = st.session_state.wind_deg
st.caption(f"Wind: **{wd:.1f}°** ({nearest_compass(wd)})")
show_legs = st.checkbox("Show expanded legs", value=False)

# Polar preview
with st.expander("Catalina 36 Polar preview"):
    tws_preview = st.slider("Preview TWS (kt)", 4, 20, 12)
    twa_range = list(range(30, 181, 5))
    polar_preview = pd.DataFrame({
        "TWA": twa_range,
        "Boat speed (kt)": [round(polar_boatspeed(t, tws_preview), 2) for t in twa_range]
    })
    st.line_chart(polar_preview.set_index("TWA"))
    up_twa, up_vmg = vmg_upwind_speed(tws_preview)
    dn_twa, dn_vmg = vmg_downwind_speed(tws_preview)
    st.caption(
        f"Best upwind VMG: {up_vmg:.2f} kt @ TWA {up_twa:.0f}°  |  "
        f"Best downwind VMG: {dn_vmg:.2f} kt @ TWA {dn_twa:.0f}°"
    )

if st.button("Generate course"):
    wind_from = float(wd) % 360.0
    candidates = recommend_courses(wind_from, float(wind_speed), top_k=int(top_n))
    if not candidates:
        st.error("No valid courses found. Try adjusting constraints.")
        st.stop()

    best = candidates[0]
    macro    = "–".join(nodes_macro(best.edges))
    expanded = "–".join(nodes_expanded(best.edges))

    st.subheader("Recommended")
    st.write(f"**Course (macro):** {macro}")
    if expanded != macro:
        st.write(f"**Course (expanded):** {expanded}")
    st.write(f"**Est time:** {best.total_time:.2f} hr  |  **Est dist:** {best.total_dist:.2f} nm")

    st.subheader("Alternates")
    rows = [{"Macro": "–".join(nodes_macro(c.edges)),
             "Expanded": "–".join(nodes_expanded(c.edges)),
             "Time (hr)": round(c.total_time, 2),
             "Dist (nm)": round(c.total_dist, 2)}
            for c in candidates[1:int(top_n)]]
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.write("No alternates.")

    if show_legs:
        st.subheader("Leg details (recommended)")
        detail_rows = []
        for e in best.edges:
            _, _, legs = edge_time(e, wind_from, float(wind_speed))
            for (a, b, dist, brg, ltype, spd, t) in legs:
                detail_rows.append({
                    "Edge": e.name, "From": a, "To": b,
                    "Dist (nm)": round(dist, 2), "Bearing": round(brg, 1),
                    "Type": ltype, "Spd (kt)": round(spd, 2), "Time (hr)": round(t, 2)
                })
        st.dataframe(detail_rows, use_container_width=True)

# Mark reference
MARK_DATA = [
    {"Short": "BB",   "Long": "Bill's Buoy",      "Aliases": "N, Start, Finish"},
    {"Short": "PNJ",  "Long": "Pete and Judy",     "Aliases": "D"},
    {"Short": "WT",   "Long": "WinningTack",        "Aliases": "WF"},
    {"Short": "WE",   "Long": "Witt's End",         "Aliases": "IF"},
    {"Short": "CC",   "Long": "Carmen's Corner",    "Aliases": "H"},
    {"Short": "TT",   "Long": "Tom's Turnaround",   "Aliases": "A"},
    {"Short": "TSC1", "Long": "TSC 1 Waypoint",     "Aliases": "TSC1"},
    {"Short": "TSC3", "Long": "TSC 3 Waypoint",     "Aliases": "TSC3"},
]
with st.expander("Mark names and aliases"):
    st.table(pd.DataFrame(MARK_DATA))
