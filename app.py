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

# Nastavení stránky Streamlit (moderní layout na celou šířku)
st.set_page_config(
    page_title="Zatmění Slunce - Webová kalkulace & Simulace",
    page_icon="☀️",
    layout="wide"
)

# Poloměry těles (v km)
R_SUN_KM = 696340.0
R_MOON_KM = 1737.4

def parse_coordinates(coord_str):
    """Naparsuje řetězec se souřadnicemi na dvojici float (lat, lon)."""
    parts = [p.strip() for p in coord_str.split(',')]
    if len(parts) != 2:
        raise ValueError("Zadejte souřadnice ve tvaru 'Šířka, Délka' (např. 49.8144375N, 14.4362431E)")

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

    progress_bar = st.progress(0, text="Probíhá astronomická analýza...")
    total_nm = len(new_moons)

    for i, nm_time in enumerate(new_moons):
        progress_bar.progress((i + 1) / total_nm, text=f"Analýza novoluní {i+1}/{total_nm} ({nm_time.utc_strftime('%Y-%m')})...")
        
        t_search = ts.utc(
            nm_time.utc_datetime().year, nm_time.utc_datetime().month, 
            nm_time.utc_datetime().day, nm_time.utc_datetime().hour - 4, 
            range(0, 480, 2)
        )

        max_obscuration = 0.0
        best_time = None
        e_type = "Žádné"

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
                            e_type = "Úplné"
                        else:
                            e_type = "Prstencové"
                    elif obscuration > 0.0:
                        e_type = "Částečné"

        if max_obscuration > 0.1 and best_time is not None:
            best_sun_obs = observer.at(best_time).observe(sun).apparent()
            alt, az, _ = best_sun_obs.altaz()
            
            if alt.degrees > 0.0:
                dt_utc = best_time.utc_datetime()
                results.append({
                    "date": dt_utc.strftime("%Y-%m-%d"),
                    "time_utc": dt_utc.strftime("%H:%M:%S UTC"),
                    "type": e_type,
                    "obscuration_val": max_obscuration,
                    "obscuration": f"{max_obscuration:.2f} %",
                    "alt": f"{alt.degrees:.1f}°",
                    "raw_time": dt_utc,
                    "best_time_obj": best_time
                })

    progress_bar.empty()
    return results

# --- ÚVOD WEBOVÉ APLIKACE ---
st.title("☀️ Zatmění Slunce – Kalkulátor & 2D Interaktivní Simulátor")
st.markdown("Aplikace počítá přesné astronomické efemeridy JPL DE440 pro libovolné místo na Zemi a generuje plynulou 2D simulaci průchodu Měsíce.")

# --- BOČNÍ PANEL (VSTUPNÍ PARAMETRY) ---
with st.sidebar:
    st.header("⚙️ Vstupní nastavení")
    
    coord_input = st.text_input("Souřadnice (WGS84):", value="49.8144375N, 14.4362431E")
    
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        start_year = st.number_input("Od roku:", min_value=1550, max_value=2650, value=2024)
    with col_y2:
        end_year = st.number_input("Do roku:", min_value=1550, max_value=2650, value=2050)
        
    btn_compute = st.button("🚀 Spočítat Zatmění", type="primary", use_container_width=True)

# --- HLAVNÍ LOGIKA ---
try:
    lat, lon = parse_coordinates(coord_input)
    elevation = fetch_elevation(lat, lon)
    st.sidebar.success(f"Zeměpisná šířka: {lat:.4f}°\n\nZeměpisná délka: {lon:.4f}°\n\nZjištěná výška: {elevation:.1f} m n. m.")
except Exception as e:
    st.error(f"Chyba ve formátu souřadnic: {e}")
    st.stop()

if btn_compute or "eclipse_results" in st.session_state:
    if btn_compute:
        with st.spinner("Počítám astronomická data..."):
            st.session_state.eclipse_results = compute_eclipses(lat, lon, elevation, start_year, end_year)
            st.session_state.current_lat = lat
            st.session_state.current_lon = lon
            st.session_state.current_elev = elevation

    results = st.session_state.get("eclipse_results", [])

    if not results:
        st.warning("V zadaném časovém rozmezí nenastane na daném místě žádné viditelné zatmění Slunce (nad obzorem).")
    else:
        st.subheader(f"📊 Nalezená viditelná zatmění ({len(results)})")
        
        # Zobrazení přehledné tabulky
        table_data = [{
            "Datum": r["date"],
            "Čas Maxima (UTC)": r["time_utc"],
            "Typ Zatmění": r["type"],
            "Zakrytí Slunce": r["obscuration"],
            "Výška Slunce": r["alt"]
        } for r in results]
        
        st.dataframe(table_data, use_container_width=True)

        # Výběr zatmění pro detailní simulaci
        st.divider()
        st.subheader("🔍 Detailní 2D Simulace a Analýza")
        
        options = [f"{r['date']} | {r['type']} ({r['obscuration']})" for r in results]
        selected_idx = st.selectbox("Vyberte zatmění pro vykreslení 2D simulace:", range(len(options)), format_func=lambda i: options[i])

        selected_eclipse = results[selected_idx]

        # Předpočet 2D časové řady pro vybrané zatmění (4 hodiny po 1 minutě)
        ts, eph = load_ephemeris()
        sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
        observer = earth + wgs84.latlon(st.session_state.current_lat, st.session_state.current_lon, elevation_m=st.session_state.current_elev)

        peak_dt = selected_eclipse['raw_time']
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
                "obscuration": obs_pct,
                "alt": altitudes[i],
                "delta_ra": delta_ra_arcmin,
                "delta_dec": delta_dec_arcmin,
                "sun_r": np.degrees(s_rad) * 60.0,
                "moon_r": np.degrees(m_rad) * 60.0
            })

        # Interaktivní posuvník času pro simulaci
        slider_frame = st.slider(
            "⏱️ Posun času v průběhu zatmění (UTC):",
            min_value=0,
            max_value=240,
            value=120,
            format_func=lambda i: frames_data[i]["time_str"]
        )

        current_frame = frames_data[slider_frame]

        # Vykreslení vedle sebe (Graf + 2D Náhled)
        col_g1, col_g2 = st.columns([1.2, 1])

        with col_g1:
            fig_graph, ax_graph = plt.subplots(figsize=(6, 4))
            obscuration_vals = [f["obscuration"] for f in frames_data]
            color_obs = '#ff8c00'
            
            ax_graph.set_xlabel('Čas (UTC)', fontsize=9)
            ax_graph.set_ylabel('Zakrytí Slunce (%)', color=color_obs, fontsize=9)
            ax_graph.plot(dt_list, obscuration_vals, color=color_obs, linewidth=2)
            ax_graph.fill_between(dt_list, obscuration_vals, color=color_obs, alpha=0.2)
            ax_graph.set_ylim(-2, 105)

            ax_graph_alt = ax_graph.twinx()
            color_alt = '#1f77b4'
            ax_graph_alt.set_ylabel('Výška Slunce (°)', color=color_alt, fontsize=9)
            ax_graph_alt.plot(dt_list, altitudes, color=color_alt, linestyle='--', linewidth=1.2)
            ax_graph_alt.axhline(0, color='gray', linestyle=':', linewidth=1)

            # Červená Ryska Aktuálního Času
            ax_graph.axvline(current_frame["dt"], color='red', linestyle='-', linewidth=2)

            ax_graph.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig_graph.autofmt_xdate()
            ax_graph.set_title("Průběh zakrytí a výšky v čase", fontsize=10, fontweight='bold')
            st.pyplot(fig_graph)

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

            ax_sim.set_xlabel('Úhlový offset (\')', fontsize=8, color='white')
            ax_sim.set_ylabel('Úhlový offset (\')', fontsize=8, color='white')
            ax_sim.tick_params(colors='white', labelsize=8)
            for spine in ax_sim.spines.values():
                spine.set_color('#30363d')

            ax_sim.set_title(f"2D Náhled | Čas: {current_frame['time_str']}\nZakrytí: {current_frame['obscuration']:.2f}% | Výška: {current_frame['alt']:.1f}°", fontsize=9, fontweight='bold')
            st.pyplot(fig_sim)
