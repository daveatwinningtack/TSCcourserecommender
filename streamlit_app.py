import streamlit as st
from dataclasses import dataclass
from math import radians, sin, cos, atan2, sqrt, degrees
from typing import Dict, List, Tuple, Optional

# -----------------------------
# MARKS (lat/lon in decimal degrees)
# -----------------------------
MARKS: Dict[str, Tuple[float, float]] = {
    "N":    (33 + 51.700/60.0, -(96 + 38.630/60.0)),  # Bill's Buoy
    "TSC1": (33 + 52.070/60.0, -(96 + 37.900/60.0)),
    "TSC3": (33 + 53.790/60.0, -(96 + 37.780/60.0)),
    "D":    (33 + 51.700/60.0, -(96 + 35.940/60.0)),  # Pete N Judi
    "IF":   (33 + 49.300/60.0, -(96 + 36.000/60.0)),  # Witt’s End
    "WF":   (33 + 54.723/60.0, -(96 + 35.790/60.0)),  # WinningTack
    "H":    (33 + 55.410/60.0, -(96 + 42.790/60.0)),  # Carmon’s Corner
}

EARTH_RADIUS_NM = 3440.065  # nautical miles

def haversine_nm(a: str, b: str) -> float:
    lat1, lon1 = MARKS[a]
    lat2, lon2 = MARKS[b]
    phi1, lam1 = radians(lat1), radians(lon1)
    phi2, lam2 = radians(lat2), radians(lon2)
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    h = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlam/2)**2
    return 2 * EARTH_RADIUS_NM * atan2(sqrt(h), sqrt(1 - h))

def initial_bearing_deg(a: str, b: str) -> float:
    lat1, lon1 = MARKS[a]
    lat2, lon2 = MARKS[b]
    phi1, lam1 = radians(lat1), radians(lon1)
    phi2, lam2 = radians(lat2), radians(lon2)
    dlam = lam2 - lam1
    x = sin(dlam) * cos(phi2)
    y = cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(dlam)
    brng = (degrees(atan2(x, y)) + 360.0) % 360.0
    return brng

def smallest_angle_deg(x: float) -> float:
    x = (x + 180.0) % 360.0 - 180.0
    return abs(x)

# -----------------------------
# COURSE GRAPH (macro-edges)
# -----------------------------
@dataclass(frozen=True)
class Edge:
    name: str
    start: str
    end: str
    path: Tuple[str, ...]  # includes start+end, corridors include interior marks

    def expanded_legs(self) -> List[Tuple[str, str]]:
        return [(self.path[i], self.path[i+1]) for i in range(len(self.path)-1)]

def is_corridor(edge: Edge) -> bool:
    return len(edge.path) > 2

# Corridors (must sail as combined corridor)
CORRIDORS: List[Edge] = [
    Edge("H->N corridor",  "H",  "N",  ("H","TSC3","TSC1","N")),
    Edge("N->H corridor",  "N",  "H",  ("N","TSC1","TSC3","H")),
    
    Edge("D->H corridor",  "D",  "H",  ("D","TSC3","H")),
    Edge("H->D corridor",  "H",  "D",  ("H","TSC3","D")),
    
    Edge("D->N corridor",  "D",  "N",  ("D","TSC1","N")),
    Edge("N->D corridor",  "N",  "D",  ("N","TSC1","D")),
    
    Edge("WF->N corridor",  "WF",  "N",  ("WF","TSC1","N")),
    Edge("N->WF corridor",  "N",  "WF",  ("N","TSC1","WF")),
    
    Edge("IF->N corridor",  "IF",  "N",  ("IF","TSC1","N")),
    Edge("N->IF corridor",  "N",  "IF",  ("N","TSC1","IF")),
    
    Edge("H->IF corridor", "H",  "IF", ("H","TSC3","IF")),
    Edge("IF->H corridor", "IF", "H",  ("IF","TSC3","H")),
]

# Simple edges (single hop only — IMPORTANT: no corridor-interior hops here)
SIMPLE_EDGES: List[Edge] = [

    Edge("IF->D",    "IF",   "D",    ("IF","D")),
    Edge("D->IF",    "D",    "IF",   ("D","IF")),

    Edge("D->WF",    "D",    "WF",   ("D","WF")),
    Edge("WF->D",    "WF",   "D",    ("WF","D")),

    Edge("IF->WF",   "IF",   "WF",   ("IF","WF")),
    Edge("WF->IF",   "WF",   "IF",   ("WF","IF")),

    Edge("WF->H",    "WF",   "H",    ("WF","H")),
    Edge("H->WF",    "H",    "WF",   ("H","WF")),

]

ALL_EDGES: List[Edge] = SIMPLE_EDGES + CORRIDORS
ADJ: Dict[str, List[Edge]] = {}
for e in ALL_EDGES:
    ADJ.setdefault(e.start, []).append(e)

# -----------------------------
# SPEED / TIME MODEL
# -----------------------------
def planning_vmg_knots(wind_speed_kt: float) -> float:
    if wind_speed_kt < 5.0:
        return 2.2
    if wind_speed_kt < 8.0:
        return 3.3
    return 4.3

def leg_type_and_speed(bearing_deg: float, wind_from_deg: float, wind_speed_kt: float) -> Tuple[str, float]:
    vmg = planning_vmg_knots(wind_speed_kt)
    wind_to = (wind_from_deg + 180.0) % 360.0
    twa = smallest_angle_deg(bearing_deg - wind_to)  # 0 downwind, 180 upwind

    if twa > 120.0:
        return "UPWIND-ish", max(1.0, vmg * 1.00)
    if twa < 60.0:
        return "DOWNWIND-ish", max(1.0, vmg * 0.95)
    return "REACH", max(1.0, vmg * 1.15)

def edge_time(edge: Edge, wind_from_deg: float, wind_speed_kt: float):
    total_dist = 0.0
    total_time = 0.0
    details = []
    for a, b in edge.expanded_legs():
        dist = haversine_nm(a, b)
        brg = initial_bearing_deg(a, b)
        ltype, spd = leg_type_and_speed(brg, wind_from_deg, wind_speed_kt)
        t = dist / spd
        total_dist += dist
        total_time += t
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

def edge_score(edge: Edge, wind_from_deg: float, wind_speed_kt: float, prev_edge: Optional[Edge]) -> float:
    dist, t, legs = edge_time(edge, wind_from_deg, wind_speed_kt)
    score = 0.0

    for (_, _, d, _, ltype, _, _) in legs:
        if wind_speed_kt >= 8.0:
            if "UPWIND" in ltype or "DOWNWIND" in ltype:
                score += 2.0 * d
            else:
                score -= 1.2 * d
        else:
            if ltype == "REACH":
                score += 2.0 * d
            elif "DOWNWIND" in ltype:
                score += 0.6 * d
            else:
                score -= 1.2 * d

    if dist < 0.6:
        score -= 3.0

    if prev_edge is not None:
        if prev_edge.start == edge.end and prev_edge.end == edge.start:
            score -= 6.0
        if prev_edge.end == edge.end:
            score -= 2.0

    if is_corridor(edge):
        score -= 0.8

    if t > 1.2:
        score -= (t - 1.2) * 2.0

    return score

def recommend_courses(
    wind_from_deg: float,
    wind_speed_kt: float,
    hard_cap_hr: float = 3.5,
    target_window: Tuple[float, float] = (2.75, 3.25),
    min_edges: int = 4,
    max_edges: int = 7,
    top_k: int = 10
) -> List[Candidate]:

    best: List[Candidate] = []

    def consider(c: Candidate):
        nonlocal best
        best.append(c)
        best.sort(key=lambda x: x.score, reverse=True)
        best = best[:top_k]

    def dfs(current: str, edges: List[Edge], t_hr: float, d_nm: float, score: float):
        if t_hr > hard_cap_hr or len(edges) > max_edges:
            return

        if current == "N" and len(edges) >= min_edges:
            lo, hi = target_window
            center = (lo + hi) / 2.0
            bonus = 10.0 if (lo <= t_hr <= hi) else -abs(t_hr - center) * 6.0
            consider(Candidate(list(edges), t_hr, d_nm, score + bonus))

        prev = edges[-1] if edges else None
        for e in ADJ.get(current, []):
            # avoid trivial early closure
            if len(edges) < 2 and e.end == "N":
                continue

            edist, etime, _ = edge_time(e, wind_from_deg, wind_speed_kt)
            escore = edge_score(e, wind_from_deg, wind_speed_kt, prev)
            dfs(e.end, edges + [e], t_hr + etime, d_nm + edist, score + escore)

    dfs("N", [], 0.0, 0.0, 0.0)
    return best

def nodes_macro(edges: List[Edge], start="N") -> List[str]:
    out = [start]
    cur = start
    for e in edges:
        out.append(e.end)
        cur = e.end
    return out

def nodes_expanded(edges: List[Edge], start="N") -> List[str]:
    out = [start]
    cur = start
    for e in edges:
        for n in e.path[1:]:
            out.append(n)
        cur = e.end
    return out

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="TSC Course Generator", layout="centered")
st.title("TSC Fixed-Marks Course Generator")
st.caption("Inputs: wind direction (from) + wind speed. Output: recommended course under 3.5 hours, 2-2.5hour target (pursuit-friendly).")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    wind_from = st.number_input("Wind FROM (deg true)", min_value=0.0, max_value=359.9, value=200.0, step=1.0)
with col2:
    wind_speed = st.number_input("Wind speed (kt)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
with col3:
    top_n = st.number_input("Show top N", min_value=1, max_value=15, value=5, step=1)

show_legs = st.checkbox("Show expanded legs (details)", value=False)

if st.button("Generate course"):
    wind_from = float(wind_from) % 360.0
    wind_speed = float(wind_speed)

    candidates = recommend_courses(wind_from, wind_speed, top_k=int(top_n))
    if not candidates:
        st.error("No valid courses found under constraints. Try increasing max edges or loosening target window.")
        st.stop()

    best = candidates[0]
    macro = "–".join(nodes_macro(best.edges))
    expanded = "–".join(nodes_expanded(best.edges))

    st.subheader("Recommended")
    st.write(f"**Course (macro):** {macro}")
    if expanded != macro:
        st.write(f"**Course (expanded):** {expanded}")
    st.write(f"**Estimated time:** {best.total_time:.2f} hr  |  **Estimated distance:** {best.total_dist:.2f} nm")

    st.subheader("Alternates")
    rows = []
    for c in candidates[1:int(top_n)]:
        rows.append({
            "Macro course": "–".join(nodes_macro(c.edges)),
            "Expanded": "–".join(nodes_expanded(c.edges)),
            "Est time (hr)": round(c.total_time, 2),
            "Est dist (nm)": round(c.total_dist, 2),
        })
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.write("No alternates (only one candidate).")

    if show_legs:
        st.subheader("Expanded legs (recommended)")
        detail_rows = []
        for e in best.edges:
            edist, etime, legs = edge_time(e, wind_from, wind_speed)
            for (a, b, dist, brg, ltype, spd, t) in legs:
                detail_rows.append({
                    "Edge": e.name,
                    "From": a,
                    "To": b,
                    "Dist (nm)": round(dist, 2),
                    "Bearing (deg)": round(brg, 1),
                    "Type": ltype,
                    "Speed (kt)": round(spd, 2),
                    "Time (hr)": round(t, 2),
                })
        st.dataframe(detail_rows, use_container_width=True)

st.divider()
with st.expander("Advanced settings"):
    st.write("If you want these exposed in the UI, we can add sliders for target window, min/max edges, etc.")
