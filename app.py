import re
import requests
import threading
import datetime
from datetime import timedelta
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skyfield.api import load, wgs84
from skyfield import almanac

# Poloměry těles (v km)
R_SUN_KM = 696340.0
R_MOON_KM = 1737.4


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


def fetch_elevation(lat, lon):
    """Zjištění nadmořské výšky z Open-Meteo API."""
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=4).json()
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


class EclipseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kalkulátor a 2D simulátor zatmění Slunce")
        self.geometry("1280x820")
        self.minsize(1024, 700)

        # Načtení astronomických dat
        self.ts = load.timescale()
        self.eph = load('de440.bsp')

        self.results = []
        self.frames_data = []
        self.dt_list = []
        self.altitudes = []

        self._build_ui()

    def _build_ui(self):
        # Levý panel pro vstupy a výsledky
        left_panel = ttk.Frame(self, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Vstupy
        ttk.Label(left_panel, text="Zeměpisné souřadnice (WGS84):", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.entry_coords = ttk.Entry(left_panel, width=32)
        self.entry_coords.insert(0, "50.0835494N, 14.4341414E")
        self.entry_coords.pack(anchor=tk.W, pady=(0, 10))

        frame_years = ttk.Frame(left_panel)
        frame_years.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame_years, text="Od roku:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.entry_start_year = ttk.Entry(frame_years, width=8)
        self.entry_start_year.insert(0, "2024")
        self.entry_start_year.grid(row=0, column=1, padx=(0, 10))

        ttk.Label(frame_years, text="Do roku:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.entry_end_year = ttk.Entry(frame_years, width=8)
        self.entry_end_year.insert(0, "2050")
        self.entry_end_year.grid(row=0, column=3)

        self.btn_compute = ttk.Button(left_panel, text="🚀 Vypočítat zatmění", command=self.start_computation)
        self.btn_compute.pack(fill=tk.X, pady=(0, 10))

        self.lbl_status = ttk.Label(left_panel, text="Připraven k výpočtu", foreground="gray")
        self.lbl_status.pack(anchor=tk.W, pady=(0, 5))

        self.progress = ttk.Progressbar(left_panel, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 10))

        # Tabulka výsledků
        ttk.Label(left_panel, text="Nalezená zatmění:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        
        columns = ("date", "time", "type", "obscuration", "alt")
        self.tree = ttk.Treeview(left_panel, columns=columns, show='headings', height=14)
        self.tree.heading("date", text="Datum")
        self.tree.heading("time", text="Čas (UTC)")
        self.tree.heading("type", text="Typ")
        self.tree.heading("obscuration", text="Zakrytí")
        self.tree.heading("alt", text="Výška")

        self.tree.column("date", width=85, anchor=tk.CENTER)
        self.tree.column("time", width=80, anchor=tk.CENTER)
        self.tree.column("type", width=70, anchor=tk.CENTER)
        self.tree.column("obscuration", width=70, anchor=tk.E)
        self.tree.column("alt", width=55, anchor=tk.E)

        self.tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", self.on_eclipse_selected)

        # Pravý panel pro simulaci a grafy
        right_panel = ttk.Frame(self, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Ovládací prvek časového posunu (Posuvník s živou aktualizací)
        slider_frame = ttk.Frame(right_panel)
        slider_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(slider_frame, text="⏱️ Časový posuvník (UTC):", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_current_time = ttk.Label(slider_frame, text="--:--:-- UTC", font=('Segoe UI', 9, 'bold'), foreground="#0066cc")
        self.lbl_current_time.pack(side=tk.LEFT)

        # Tkinter Scale zajišťující okamžitou aktualizaci při držení a tažení myší
        self.slider = tk.Scale(
            right_panel, 
            from_=0, 
            to=240, 
            orient=tk.HORIZONTAL, 
            showvalue=False,
            command=self.on_slider_move
        )
        self.slider.pack(fill=tk.X, pady=(0, 10))
        # Navázání událostí pro plynulé živé překreslování při držení tlačítka myši
        self.slider.bind("<B1-Motion>", lambda e: self.on_slider_move(self.slider.get()))

        # Matplotlib Plátno pro 2D simulaci a graf
        self.fig, (self.ax_graph, self.ax_sim) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.2, 1]})
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_empty_plots()

    def _init_empty_plots(self):
        self.ax_graph.clear()
        self.ax_sim.clear()

        self.ax_graph.set_title("Průběh zakrytí a výška Slunce", fontsize=10, fontweight='bold')
        self.ax_graph.set_xlabel("Čas (UTC)", fontsize=8)
        self.ax_graph.set_ylabel("Zakrytí Slunce (%)", fontsize=8)

        self.ax_sim.set_facecolor('#0d1117')
        self.ax_sim.set_title("2D Pohled na zatmění", fontsize=10, fontweight='bold')
        self.ax_sim.set_xticks([])
        self.ax_sim.set_yticks([])

        self.canvas.draw_idle()

    def start_computation(self):
        try:
            coord_str = self.entry_coords.get()
            self.lat, self.lon = parse_coordinates(coord_str)
            self.start_year = int(self.entry_start_year.get())
            self.end_year = int(self.entry_end_year.get())
        except Exception as e:
            messagebox.showerror("Chyba vstupních dat", str(e))
            return

        self.btn_compute.config(state=tk.DISABLED)
        self.lbl_status.config(text="Zjišťuji nadmořskou výšku...", foreground="black")
        self.progress['value'] = 0

        # Spuštění v samostatném vlákně
        threading.Thread(target=self._async_compute, daemon=True).start()

    def _async_compute(self):
        elevation = fetch_elevation(self.lat, self.lon)
        self.elevation = elevation

        sun, moon, earth = self.eph['sun'], self.eph['moon'], self.eph['earth']
        observer = earth + wgs84.latlon(self.lat, self.lon, elevation_m=elevation)

        t0 = self.ts.utc(self.start_year, 1, 1)
        t1 = self.ts.utc(self.end_year, 12, 31)

        f_phases = almanac.moon_phases(self.eph)
        times, phases = almanac.find_discrete(t0, t1, f_phases)

        new_moons = times[phases == 0]
        results = []
        total_nm = len(new_moons)

        for i, nm_time in enumerate(new_moons):
            pct = int(((i + 1) / total_nm) * 100)
            dt_nm = nm_time.utc_strftime('%Y-%m')
            self.after(0, self._update_progress, pct, f"Analýza novu {i+1}/{total_nm} ({dt_nm})...")

            t_search = self.ts.utc(
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
                        "time_utc": dt_utc.strftime("%H:%M:%S"),
                        "type": e_type,
                        "obscuration_val": max_obscuration,
                        "obscuration": f"{max_obscuration:.2f} %",
                        "alt": f"{alt.degrees:.1f}°",
                        "raw_time": dt_utc
                    })

        self.after(0, self._finish_computation, results)

    def _update_progress(self, val, text):
        self.progress['value'] = val
        self.lbl_status.config(text=text)

    def _finish_computation(self, results):
        self.results = results
        self.btn_compute.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Hotovo! Nalezeno {len(results)} zatmění.", foreground="green")

        # Vyčištění a naplnění tabulky
        for item in self.tree.get_children():
            self.tree.delete(item)

        for r in results:
            self.tree.insert("", tk.END, values=(r["date"], r["time_utc"], r["type"], r["obscuration"], r["alt"]))

        if results:
            first_child = self.tree.get_children()[0]
            self.tree.selection_set(first_child)

    def on_eclipse_selected(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return

        item_idx = self.tree.index(selected_items[0])
        selected_eclipse = self.results[item_idx]

        self._prepare_simulation_data(selected_eclipse)

    def _prepare_simulation_data(self, eclipse):
        sun, moon, earth = self.eph['sun'], self.eph['moon'], self.eph['earth']
        observer = earth + wgs84.latlon(self.lat, self.lon, elevation_m=self.elevation)

        peak_dt = eclipse['raw_time']
        start_dt = peak_dt - timedelta(hours=2)
        
        minutes = range(241)
        t_range = self.ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute + np.array(minutes))

        obs_sun_range = observer.at(t_range).observe(sun).apparent()
        obs_moon_range = observer.at(t_range).observe(moon).apparent()

        sep_range = obs_sun_range.separation_from(obs_moon_range).radians
        d_sun_range = obs_sun_range.distance().km
        d_moon_range = obs_moon_range.distance().km

        r_sun_rad = np.arcsin(R_SUN_KM / d_sun_range)
        r_moon_rad = np.arcsin(R_MOON_KM / d_moon_range)
        self.altitudes = obs_sun_range.altaz()[0].degrees

        self.frames_data = []
        self.dt_list = []

        for i in range(len(t_range)):
            dt = t_range[i].utc_datetime()
            self.dt_list.append(dt)

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

            self.frames_data.append({
                "time_str": dt.strftime("%H:%M:%S UTC"),
                "dt": dt,
                "obscuration": obs_pct,
                "alt": self.altitudes[i],
                "delta_ra": delta_ra_arcmin,
                "delta_dec": delta_dec_arcmin,
                "sun_r": np.degrees(s_rad) * 60.0,
                "moon_r": np.degrees(m_rad) * 60.0
            })

        self.slider.config(to=len(self.frames_data) - 1)
        self.slider.set(120)  # Nastavení na střed (vrchol zatmění)
        self.update_simulation_plots(120)

    def on_slider_move(self, val):
        if not self.frames_data:
            return
        idx = int(float(val))
        self.update_simulation_plots(idx)

    def update_simulation_plots(self, idx):
        if idx >= len(self.frames_data):
            return

        frame = self.frames_data[idx]
        self.lbl_current_time.config(text=frame["time_str"])

        # 1. Graf zakrytí a výšky
        self.ax_graph.clear()
        obscuration_vals = [f["obscuration"] for f in self.frames_data]
        color_obs = '#ff8c00'
        
        self.ax_graph.set_xlabel('Čas (UTC)', fontsize=8)
        self.ax_graph.set_ylabel('Zakrytí Slunce (%)', color=color_obs, fontsize=8)
        self.ax_graph.plot(self.dt_list, obscuration_vals, color=color_obs, linewidth=2)
        self.ax_graph.fill_between(self.dt_list, obscuration_vals, color=color_obs, alpha=0.2)
        self.ax_graph.set_ylim(-2, 105)

        # Druhá osa Y pro výšku
        ax_graph_alt = self.ax_graph.twinx()
        color_alt = '#1f77b4'
        ax_graph_alt.set_ylabel('Výška Slunce (°)', color=color_alt, fontsize=8)
        ax_graph_alt.plot(self.dt_list, self.altitudes, color=color_alt, linestyle='--', linewidth=1.2)
        ax_graph_alt.axhline(0, color='gray', linestyle=':', linewidth=1)

        self.ax_graph.axvline(frame["dt"], color='red', linestyle='-', linewidth=1.5)
        self.ax_graph.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax_graph.tick_params(labelsize=8)
        ax_graph_alt.tick_params(labelsize=8)
        self.ax_graph.set_title("Průběh zakrytí a výška Slunce", fontsize=9, fontweight='bold')

        # 2. 2D Simulace
        self.ax_sim.clear()
        self.ax_sim.set_facecolor('#0d1117')
        
        lim = frame["sun_r"] * 2.2
        self.ax_sim.set_xlim(-lim, lim)
        self.ax_sim.set_ylim(-lim, lim)
        self.ax_sim.set_aspect('equal')

        sun_patch = Circle((0, 0), frame["sun_r"], color='#ffcc00', ec='#ff9900', lw=1.5, zorder=1)
        moon_patch = Circle((frame["delta_ra"], frame["delta_dec"]), frame["moon_r"], color='#1c2128', ec='#8b949e', lw=1.2, zorder=2)

        self.ax_sim.add_patch(sun_patch)
        self.ax_sim.add_patch(moon_patch)

        self.ax_sim.set_xlabel("Úhlový posun (')", fontsize=8, color='white')
        self.ax_sim.set_ylabel("Úhlový posun (')", fontsize=8, color='white')
        self.ax_sim.tick_params(colors='white', labelsize=8)
        for spine in self.ax_sim.spines.values():
            spine.set_color('#30363d')

        self.ax_sim.set_title(
            f"2D Pohled | Zakrytí: {frame['obscuration']:.2f}% | Výška: {frame['alt']:.1f}°", 
            fontsize=9, fontweight='bold', color='white'
        )

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = EclipseApp()
    app.mainloop()
