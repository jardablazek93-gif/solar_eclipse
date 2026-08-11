import re
import requests
import numpy as np
import datetime
from datetime import timedelta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle

from skyfield.api import load, wgs84
from skyfield import almanac

# Konfigurace stránky Streamlit
st.set_page_config(
    page_title="Solar Eclipse - Web Calculator & Simulation",
    page_icon="☀️",
    layout="wide"
)

# Poloměry těles (v km)
R_SUN_KM = 696340.0
R_MOON_KM = 1737.4

# ==============================================================================
# JAZYKOVÝ SLOVNÍK (i18n)
# ==============================================================================
TEXTS = {
    "CZ": {
        "page_title": "Solar Eclipse – Kalkulátor zatmění Slunce",
        "page_desc": "Výpočet zatmění Slunce pro libovolné místo na Zemi s 2D simulací průběhu.",
        "sidebar_header": "⚙️ Vstupní parametry",
        "lang_label": "🌐 Jazyk / Language:",
        "coord_label": "Souřadnice (WGS84):",
        "start_year": "Od roku:",
        "end_year": "Do roku:",
        "btn_compute": "🚀 Vypočítat zatmění",
        "sidebar_lat": "Šířka",
        "sidebar_lon": "Délka",
        "sidebar_elev": "Výška",
        "no_eclipses": "Pro zadanou polohu a časový rozsah nebylo nalezeno žádné viditelné zatmění (nad obzorem).",
        "found_eclipses": "📊 Nalezená viditelná zatmění",
        "col_date": "Datum",
        "col_time": "Čas maxima (UTC)",
        "col_type": "Typ zatmění",
        "col_obs": "Zakrytí Slunce",
        "col_alt": "Výška Slunce",
        "sim_header": "🔍 Detailní 2D Simulace a Graf",
        "select_eclipse": "Vyberte zatmění pro simulaci:",
        "slider_label": "⏱️ Posun času při zatmění (UTC):",
        "chart_title": "Průběh zakrytí a výška Slunce",
        "chart_xlabel": "Čas (UTC)",
        "chart_ylabel_obs": "Zakrytí Slunce (%)",
        "chart_ylabel_alt": "Výška Slunce (°)",
        "sim_title": "2D Pohled",
        "sim_xlabel": "Úhlový posun (')",
        "sim_ylabel": "Úhlový posun (')",
        "sim_time": "Čas",
        "sim_obs": "Zakrytí",
        "sim_alt": "Výška",
        "type_total": "Úplné",
        "type_annular": "Prstencové",
        "type_partial": "Částečné",
        "err_coord": "Chyba ve formátu souřadnic:"
    },
    "EN": {
        "page_title": "☀️ Solar Eclipse – Calculator",
        "page_desc": "Calculation of solar eclipse for any location on Earth with 2D simulation.",
        "sidebar_header": "⚙️ Input Parameters",
        "lang_label": "🌐 Language / Jazyk:",
        "coord_label": "Coordinates (WGS84):",
        "start_year": "From year:",
        "end_year": "To year:",
        "btn_compute": "🚀 Calculate Eclipses",
        "sidebar_lat": "Latitude",
        "sidebar_lon": "Longitude",
        "sidebar_elev": "Elevation",
        "no_eclipses": "No visible solar eclipses found for the specified location and time range (above horizon).",
        "found_eclipses": "📊 Visible Eclipses Found",
        "col_date": "Date",
        "col_time": "Peak Time (UTC)",
        "col_type": "Eclipse Type",
        "col_obs": "Sun Obscuration",
        "col_alt": "Sun Altitude",
        "sim_header": "🔍 Detailed 2D Simulation & Chart",
        "select_eclipse": "Select eclipse for simulation:",
        "slider_label": "⏱️ Time offset during eclipse (UTC):",
        "chart_title": "Obscuration and Sun Altitude Progress",
        "chart_xlabel": "Time (UTC)",
        "chart_ylabel_obs": "Obscuration (%)",
        "chart_ylabel_alt": "Sun Altitude (°)",
        "sim_title": "2D View",
        "sim_xlabel": "Angular Offset (')",
        "sim_ylabel": "Angular Offset (')",
        "sim_time": "Time",
        "sim_obs": "Obscuration",
        "sim_alt": "Altitude",
        "type_total": "Total",
        "type_annular": "Annular",
        "type_partial": "Partial",
        "err_coord": "Coordinate format error:"
    }
}


def parse_coordinates(coord_str):
    """Naparsuje řetězec se souřadnicemi na dvojici float (lat, lon)."""
    parts = [p.strip() for p in coord_str.split(',')]
    if len(parts) != 2:
        raise ValueError("Zadejte souřadnice ve tvaru 'Šířka, Délka' (např. 50.0835494N, 14.4341414E)")

    def parse_part(part, pos_dirs, neg_dirs):
        match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*([NSEWnsew]?)', part)
        if not match:
            raise ValueError(f"Neplatný formát souřadnice: {part}")
        val = float(match.group(1))
        dir_char = match.group(2).upper()
        if dir_char in neg_dirs:
            val = -abs(val)
        elif dir_char in pos_dirs:
            val = abs(val)
        return val

    lat = parse_part(parts[0], ['N'], ['S'])
    lon = parse_part(parts[1], ['E'], ['W'])

    if not (-90.0 <= lat <= 90.0):
        raise ValueError("Zeměpisná šířka musí být v rozsahu -90 až 90 stupňů.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("Zeměpisná délka musí být v rozsahu -180 až 180 stupňů.")

    return lat, lon


@st.cache_data(ttl=86400)
def fetch_elevation(lat, lon):
    """Zjištění nadmořské výšky z Open-Meteo API s kešováním."""
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5).json()
        if "elevation" in res and len(res["elevation"]) > 0:
            return float(res["elevation"][0])
    except Exception:
        pass
    return 0.0


def calculate_circle_intersection_area(r_s, r_m, d):
    """Spočítá plochu překryvu dvou kružnic na obloze."""
    if d >= r_s + r_m:
        return 0.0
    if d <= abs(r_s - r_m):
        return np.pi * (min(r_s, r_m) ** 2)

    r_s2, r_m2, d2 = r_s**2, r_m**2, d**2
    alpha = np.arccos(np.clip((d2 + r_s2 - r_m2) / (2 * d * r_s), -1.0, 1.0))
    beta = np.arccos(np.clip((d2 + r_m2 - r_s2) / (2 * d * r_m), -1.0, 1.0))

    overlap = (r_s2 * alpha + r_m2 * beta - 
               0.5 * np.sqrt(max(0.0, (-d + r_s + r_m) * (d + r_s - r_m) * (d - r_s + r_m) * (d + r_s + r_m))))
    return overlap


@st.cache_resource
def load_ephemeris():
    """Načte astronomické efemeridy JPL DE440 s kešováním."""
    ts = load.timescale()
    eph = load('de440.bsp')
    return ts, eph


@st.cache_data(ttl=3600, show_spinner="Počítám astronomická data...")
def compute_eclipses(lat, lon, elevation, start_year, end_year):
    """Vypočítá viditelná zatmění pro zadané období a souřadnice."""
    ts, eph = load_ephemeris()
    sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
    observer = earth + wgs84.latlon(lat, lon, elevation_m=elevation)

    t0 = ts.utc(start_year, 1, 1)
    t1 = ts.utc(end_year, 12, 31)

    f_phases = almanac.moon_phases(eph)
    times, phases = almanac.find_discrete(t0, t1, f_phases)

    new_moons = times[phases == 0]
    results = []

    for i, nm_time in enumerate(new_moons):
        t_search = ts.utc(
            nm_time.utc_datetime().year, nm_time.utc_datetime().month, 
            nm_time.utc_datetime().day, nm_time.utc_datetime().hour - 4, 
            range(0, 480, 2)
        )

        max_obscuration = 0.0
        best_time = None
        type_key = "none"

        obs_sun = observer.at(t_search).observe(sun).apparent()
        obs_moon = observer.at(t_search).observe(moon).apparent()

        sep = obs_sun.separation_from(obs_moon).radians
        d_sun = obs_sun.distance().km
        d_moon = obs_moon.distance().km

        r_sun_rad = np.arcsin(R_SUN_KM / d_sun)
        r_moon_rad = np.arcsin(R_MOON_KM / d_moon)

        for idx in range(len(t_search)):
            s_rad = r_sun_rad[idx]
            m_rad = r_moon_rad[idx]
            d_val = sep[idx]

            if d_val < (s_rad + m_rad):
                sun_area = np.pi * (s_rad ** 2)
                intersection = calculate_circle_intersection_area(s_rad, m_rad, d_val)
                obscuration = (intersection / sun_area) * 100.0

                if obscuration > max_obscuration:
                    max_obscuration = obscuration
                    best_time = t_search[idx]
                    
                    if obscuration >= 99.9:
                        if m_rad >= s_rad:
                            type_key = "type_total"
                        else:
                            type_key = "type_annular"
                    elif obscuration > 0.0:
                        type_key = "type_partial"

        if max_obscuration > 0.1 and best_time is not None:
            best_sun_obs = observer.at(best_time).observe(sun).apparent()
            alt, az, _ = best_sun_obs.altaz()
            
            if alt.degrees > 0.0:
                dt_utc = best_time.utc_datetime()
                results.append({
                    "date": dt_utc.strftime("%Y-%m-%d"),
                    "time_utc": dt_utc.strftime("%H:%M:%S UTC"),
                    "type_key": type_key,
                    "obscuration_val": max_obscuration,
                    "obscuration": f"{max_obscuration:.2f} %",
                    "alt": f"{alt.degrees:.1f}°",
                    "raw_time": dt_utc
                })

    return results


@st.cache_data(ttl=3600, show_spinner="Příprava dat pro simulaci...")
def compute_eclipse_frames(lat, lon, elevation, peak_dt):
    """Vykešuje 240 snímků simulace pro plynulý posuvník."""
    ts, eph = load_ephemeris()
    sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
    observer = earth + wgs84.latlon(lat, lon, elevation_m=elevation)

    start_dt = peak_dt - timedelta(hours=2)
    minutes = range(241)
    t_range = ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute + np.array(minutes))

    obs_sun_range = observer.at(t_range).observe(sun).apparent()
    obs_moon_range = observer.at(t_range).observe(moon).apparent()

    sep_range = obs_sun_range.separation_from(obs_moon_range).radians
    d_sun_range = obs_sun_range.distance().km
    d_moon_range = obs_moon_range.distance().km

    r_sun_rad = np.arcsin(R_SUN_KM / d_sun_range)
    r_moon_rad = np.arcsin(R_MOON_KM / d_moon_range)
    altitudes = obs_sun_range.altaz()[0].degrees

    frames_data = []
    dt_list = []

    for i in range(len(t_range)):
        dt = t_range[i].utc_datetime()
        dt_list.append(dt)

        s_rad, m_rad, d_val = r_sun_rad[i], r_moon_rad[i], sep_range[i]

        if d_val < (s_rad + m_rad):
            sun_area = np.pi * (s_rad ** 2)
            intersection = calculate_circle_intersection_area(s_rad, m_rad, d_val)
            obs_pct = (intersection / sun_area) * 100.0
        else:
            obs_pct = 0.0

        ra_s, dec_s, _ = obs_sun_range[i].radec()
        ra_m, dec_m, _ = obs_moon_range[i].radec()

        delta_ra_arcmin = (ra_m.hours - ra_s.hours) * 15.0 * 60.0 * np.cos(dec_s.radians)
        delta_dec_arcmin = (dec_m.degrees - dec_s.degrees) * 60.0

        frames_data.append({
            "time_str": dt.strftime("%H:%M:%S UTC"),
            "dt": dt,
            "obscuration": float(obs_pct),
            "alt": float(altitudes[i]),
            "delta_ra": float(delta_ra_arcmin),
            "delta_dec": float(delta_dec_arcmin),
            "sun_r": float(np.degrees(s_rad) * 60.0),
            "moon_r": float(np.degrees(m_rad) * 60.0)
        })

    return frames_data, dt_list, [float(a) for a in altitudes]


# ==============================================================================
# BOČNÍ PANEL A VOLBA JAZYKA
# ==============================================================================
with st.sidebar:
    lang = st.radio("🌐 Language / Jazyk", ["CZ", "EN"], horizontal=True)
    t = TEXTS[lang]

    st.header(t["sidebar_header"])
    coord_input = st.text_input(t["coord_label"], value="50.0835494N, 14.4341414E")

    col_y1, col_y2 = st.columns(2)
    with col_y1:
        start_year = st.number_input(t["start_year"], min_value=1550, max_value=2650, value=2024)
    with col_y2:
        end_year = st.number_input(t["end_year"], min_value=1550, max_value=2650, value=2050)

    btn_compute = st.button(t["btn_compute"], type="primary", use_container_width=True)

# --- HLAVNÍ OBSAH ---
st.title(t["page_title"])
st.markdown(t["page_desc"])

try:
    lat, lon = parse_coordinates(coord_input)
    elevation = fetch_elevation(lat, lon)
    st.sidebar.success(f"{t['sidebar_lat']}: {lat:.4f}°\n\n{t['sidebar_lon']}: {lon:.4f}°\n\n{t['sidebar_elev']}: {elevation:.1f} m n. m.")
except Exception as e:
    st.error(f"{t['err_coord']} {e}")
    st.stop()

if btn_compute or "eclipse_results" in st.session_state:
    if btn_compute:
        st.session_state.eclipse_results = compute_eclipses(lat, lon, elevation, start_year, end_year)
        st.session_state.current_lat = lat
        st.session_state.current_lon = lon
        st.session_state.current_elev = elevation

    results = st.session_state.get("eclipse_results", [])

    if not results:
        st.warning(t["no_eclipses"])
    else:
        st.subheader(f"{t['found_eclipses']} ({len(results)})")

        table_data = [{
            t["col_date"]: r["date"],
            t["col_time"]: r["time_utc"],
            t["col_type"]: t.get(r["type_key"], "Unknown"),
            t["col_obs"]: r["obscuration"],
            t["col_alt"]: r["alt"]
        } for r in results]

        st.dataframe(table_data, use_container_width=True)

        st.divider()
        st.subheader(t["sim_header"])

        options = [f"{r['date']} | {t.get(r['type_key'], '')} ({r['obscuration']})" for r in results]
        selected_idx = st.selectbox(t["select_eclipse"], range(len(options)), format_func=lambda i: options[i])
        selected_eclipse = results[selected_idx]

        # Načtení dat pro simulaci
        frames_data, dt_list, altitudes = compute_eclipse_frames(
            st.session_state.current_lat,
            st.session_state.current_lon,
            st.session_state.current_elev,
            selected_eclipse['raw_time']
        )

        # BEZPEČNÝ SELECT_SLIDER PRO TEXTOVÉ ŘETĚZCE (Zabraňuje TypeError na Pythonu 3.14 / Streamlit Cloud)
        time_options = [f["time_str"] for f in frames_data]
        default_time = time_options[len(time_options) // 2]

        selected_time_str = st.select_slider(
            t["slider_label"],
            options=time_options,
            value=default_time
        )

        # Najde přesný rámec dat odpovídající vybranému času
        current_frame = next(f for f in frames_data if f["time_str"] == selected_time_str)

        # Vykreslení grafu a simulace
        col_g1, col_g2 = st.columns([1.2, 1])

        with col_g1:
            fig_graph, ax_graph = plt.subplots(figsize=(6, 4))
            obscuration_vals = [f["obscuration"] for f in frames_data]
            color_obs = '#ff8c00'

            ax_graph.set_xlabel(t["chart_xlabel"], fontsize=9)
            ax_graph.set_ylabel(t["chart_ylabel_obs"], color=color_obs, fontsize=9)
            ax_graph.plot(dt_list, obscuration_vals, color=color_obs, linewidth=2)
            ax_graph.fill_between(dt_list, obscuration_vals, color=color_obs, alpha=0.2)
            ax_graph.set_ylim(-2, 105)

            ax_graph_alt = ax_graph.twinx()
            color_alt = '#1f77b4'
            ax_graph_alt.set_ylabel(t["chart_ylabel_alt"], color=color_alt, fontsize=9)
            ax_graph_alt.plot(dt_list, altitudes, color=color_alt, linestyle='--', linewidth=1.2)
            ax_graph_alt.axhline(0, color='gray', linestyle=':', linewidth=1)

            # Červená ryska pozice posuvníku
            ax_graph.axvline(current_frame["dt"], color='red', linestyle='-', linewidth=2)

            ax_graph.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig_graph.autofmt_xdate()
            ax_graph.set_title(t["chart_title"], fontsize=10, fontweight='bold')
            st.pyplot(fig_graph, clear_figure=True)

        with col_g2:
            fig_sim, ax_sim = plt.subplots(figsize=(5, 4))
            ax_sim.set_facecolor('#0d1117')

            lim = current_frame["sun_r"] * 2.2
            ax_sim.set_xlim(-lim, lim)
            ax_sim.set_ylim(-lim, lim)
            ax_sim.set_aspect('equal')

            sun_patch = Circle((0, 0), current_frame["sun_r"], color='#ffcc00', ec='#ff9900', lw=1.5, zorder=1)
            moon_patch = Circle((current_frame["delta_ra"], current_frame["delta_dec"]), current_frame["moon_r"], color='#1c2128', ec='#8b949e', lw=1.2, zorder=2)

            ax_sim.add_patch(sun_patch)
            ax_sim.add_patch(moon_patch)

            ax_sim.set_xlabel(t["sim_xlabel"], fontsize=8, color='white')
            ax_sim.set_ylabel(t["sim_ylabel"], fontsize=8, color='white')
            ax_sim.tick_params(colors='white', labelsize=8)
            for spine in ax_sim.spines.values():
                spine.set_color('#30363d')

            sim_title_text = (
                f"{t['sim_title']} | {t['sim_time']}: {current_frame['time_str']}\n"
                f"{t['sim_obs']}: {current_frame['obscuration']:.2f}% | {t['sim_alt']}: {current_frame['alt']:.1f}°"
            )
            ax_sim.set_title(sim_title_text, fontsize=9, fontweight='bold', color='white')
            st.pyplot(fig_sim, clear_figure=True)
