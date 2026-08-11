import re
import datetime
from datetime import timedelta
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

from skyfield.api import load, wgs84
from skyfield import almanac

# Konstanta pro poloměry
R_SUN_KM = 696340.0
R_MOON_KM = 1737.4


def parse_coordinates(coord_str):
    parts = [p.strip() for p in coord_str.split(',')]
    if len(parts) != 2:
        raise ValueError("Zadejte souřadnice ve tvaru 'Šířka, Délka' (např. 50.0835, 14.4341)")

    def parse_part(part, pos_dirs, neg_dirs):
        match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*([NSEWnsew]?)', part)
        if not match:
            raise ValueError(f"Neplatný formát: {part}")
        val = float(match.group(1))
        dir_char = match.group(2).upper()
        if dir_char in neg_dirs:
            val = -abs(val)
        elif dir_char in pos_dirs:
            val = abs(val)
        return val

    lat = parse_part(parts[0], ['N'], ['S'])
    lon = parse_part(parts[1], ['E'], ['W'])

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError("Souřadnice jsou mimo platný rozsah.")

    return lat, lon


def fetch_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=4).json()
        if "elevation" in res and len(res["elevation"]) > 0:
            return float(res["elevation"][0])
    except Exception:
        pass
    return 0.0


def calculate_overlap_area(r_s, r_m, d):
    if d >= r_s + r_m:
        return 0.0
    if d <= abs(r_s - r_m):
        return np.pi * (min(r_s, r_m) ** 2)

    r_s2, r_m2, d2 = r_s**2, r_m**2, d**2
    alpha = np.arccos(np.clip((d2 + r_s2 - r_m2) / (2 * d * r_s), -1.0, 1.0))
    beta = np.arccos(np.clip((d2 + r_m2 - r_s2) / (2 * d * r_m), -1.0, 1.0))

    return (r_s2 * alpha + r_m2 * beta - 
            0.5 * np.sqrt(max(0.0, (-d + r_s + r_m) * (d + r_s - r_m) * (d - r_s + r_m) * (d + r_s + r_m))))


class EclipseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Zatmění Slunce – Výpočet a vizualizace")
        self.geometry("1100x750")
        self.minsize(900, 600)

        self.ts = None
        self.eph = None
        self.results = []
        self.current_frames = []

        self._build_ui()
        self._load_ephemeris_async()

    def _build_ui(self):
        # Horní panel s parametry
        input_frame = ttk.LabelFrame(self, text=" Parametry pozorování ", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(input_frame, text="Souřadnice (Šířka, Délka):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.ent_coords = ttk.Entry(input_frame, width=28)
        self.ent_coords.insert(0, "50.0835, 14.4341")
        self.ent_coords.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Rok od:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.ent_start = ttk.Entry(input_frame, width=8)
        self.ent_start.insert(0, "2024")
        self.ent_start.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(input_frame, text="Rok do:").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        self.ent_end = ttk.Entry(input_frame, width=8)
        self.ent_end.insert(0, "2040")
        self.ent_end.grid(row=0, column=5, padx=5, pady=5)

        self.btn_calc = ttk.Button(input_frame, text="Vyhledat zatmění", command=self._start_calculation)
        self.btn_calc.grid(row=0, column=6, padx=15, pady=5)

        # Hlavní rozdělené okno (Tabulka vlevo / Graf a simulace vpravo)
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Leví panel: Seznam událostí
        left_frame = ttk.LabelFrame(paned, text=" Nalezená zatmění ", padding=5)
        paned.add(left_frame, weight=1)

        columns = ("date", "time", "type", "obs", "alt")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("date", text="Datum")
        self.tree.heading("time", text="Místní čas (UTC)")
        self.tree.heading("type", text="Typ")
        self.tree.heading("obs", text="Zakrytí")
        self.tree.heading("alt", text="Výška")

        self.tree.column("date", width=90)
        self.tree.column("time", width=90)
        self.tree.column("type", width=90)
        self.tree.column("obs", width=70)
        self.tree.column("alt", width=60)

        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_eclipse_select)

        # Pravý panel: Graf a simulace
        right_frame = ttk.Frame(paned, padding=5)
        paned.add(right_frame, weight=2)

        # Posuvník času
        slider_frame = ttk.LabelFrame(right_frame, text=" Průběh v čase ", padding=5)
        slider_frame.pack(fill="x", pady=(0, 5))

        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal", command=self._on_slider_move)
        self.slider.pack(fill="x", padx=5, pady=2)
        
        self.lbl_slider_time = ttk.Label(slider_frame, text="Čas: --:--:-- UTC")
        self.lbl_slider_time.pack()

        # Matplotlib Plátno
        self.fig, (self.ax_chart, self.ax_sim) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stavová lišta
        self.status_label = ttk.Label(self, text="Načítání astronomických dat...", relief="sunken", anchor="w", padding=3)
        self.status_label.pack(side="bottom", fill="x")

    def _load_ephemeris_async(self):
        def worker():
            self.ts = load.timescale()
            self.eph = load('de440.bsp')
            self.status_label.config(text="Připraveno k výpočtu.")

        threading.Thread(target=worker, daemon=True).start()

    def _start_calculation(self):
        if not self.eph:
            messagebox.showwarning("Upozornění", "Astronomická data se stále načítají. Chvíli vyčkejte.")
            return

        try:
            lat, lon = parse_coordinates(self.ent_coords.get())
            start_y = int(self.ent_start.get())
            end_y = int(self.ent_end.get())
        except ValueError as e:
            messagebox.showerror("Chyba vstupu", str(e))
            return

        self.btn_calc.config(state="disabled")
        self.status_label.config(text="Probíhá výpočet zatmění...")

        def worker():
            elev = fetch_elevation(lat, lon)
            earth, sun, moon = self.eph['earth'], self.eph['sun'], self.eph['moon']
            observer = earth + wgs84.latlon(lat, lon, elevation_m=elev)

            t0 = self.ts.utc(start_y, 1, 1)
            t1 = self.ts.utc(end_y, 12, 31)

            f_phases = almanac.moon_phases(self.eph)
            times, phases = almanac.find_discrete(t0, t1, f_phases)

            new_moons = times[phases == 0]
            results = []

            for nm in new_moons:
                nm_dt = nm.utc_datetime()
                t_search = self.ts.utc(nm_dt.year, nm_dt.month, nm_dt.day, nm_dt.hour - 4, range(0, 480, 3))

                obs_sun = observer.at(t_search).observe(sun).apparent()
                obs_moon = observer.at(t_search).observe(moon).apparent()

                sep = obs_sun.separation_from(obs_moon).radians
                d_sun = obs_sun.distance().km
                d_moon = obs_moon.distance().km

                r_sun_rad = np.arcsin(R_SUN_KM / d_sun)
                r_moon_rad = np.arcsin(R_MOON_KM / d_moon)

                max_obs = 0.0
                best_t = None
                type_str = "Částečné"

                for idx in range(len(t_search)):
                    s_r, m_r, d_v = r_sun_rad[idx], r_moon_rad[idx], sep[idx]
                    if d_v < (s_r + m_r):
                        area_s = np.pi * (s_r ** 2)
                        overlap = calculate_overlap_area(s_r, m_r, d_v)
                        obs_pct = (overlap / area_s) * 100.0

                        if obs_pct > max_obs:
                            max_obs = obs_pct
                            best_t = t_search[idx]
                            if obs_pct >= 99.8:
                                type_str = "Úplné" if m_r >= s_r else "Prstencové"

                if max_obs > 0.1 and best_t is not None:
                    alt = observer.at(best_t).observe(sun).apparent().altaz()[0].degrees
                    if alt > 0:
                        dt = best_t.utc_datetime()
                        results.append({
                            "date": dt.strftime("%Y-%m-%d"),
                            "time": dt.strftime("%H:%M UTC"),
                            "type": type_str,
                            "obs_val": max_obs,
                            "obs": f"{max_obs:.1f} %",
                            "alt": f"{alt:.1f}°",
                            "raw_dt": dt,
                            "lat": lat, "lon": lon, "elev": elev
                        })

            self.results = results
            self.after(0, self._update_table_results)

        threading.Thread(target=worker, daemon=True).start()

    def _update_table_results(self):
        self.tree.delete(*self.tree.get_children())
        for idx, r in enumerate(self.results):
            self.tree.insert("", "end", iid=str(idx), values=(r["date"], r["time"], r["type"], r["obs"], r["alt"]))

        self.btn_calc.config(state="normal")
        count = len(self.results)
        self.status_label.config(text=f"Výpočet dokončen. Nalezeno {count} viditelných událostí.")

    def _on_eclipse_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return

        idx = int(selected[0])
        item = self.results[idx]

        self.status_label.config(text="Generování simulace...")
        
        def worker():
            peak_dt = item["raw_dt"]
            start_dt = peak_dt - timedelta(hours=2)
            
            t_range = self.ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, 
                                  start_dt.minute + np.arange(0, 241, 2))

            earth, sun, moon = self.eph['earth'], self.eph['sun'], self.eph['moon']
            observer = earth + wgs84.latlon(item["lat"], item["lon"], elevation_m=item["elev"])

            obs_sun = observer.at(t_range).observe(sun).apparent()
            obs_moon = observer.at(t_range).observe(moon).apparent()

            sep = obs_sun.separation_from(obs_moon).radians
            d_sun = obs_sun.distance().km
            d_moon = obs_moon.distance().km

            r_sun_rad = np.arcsin(R_SUN_KM / d_sun)
            r_moon_rad = np.arcsin(R_MOON_KM / d_moon)

            frames = []
            for i in range(len(t_range)):
                dt = t_range[i].utc_datetime()
                s_r, m_r, d_v = r_sun_rad[i], r_moon_rad[i], sep[i]
                
                obs_pct = 0.0
                if d_v < (s_r + m_r):
                    area_s = np.pi * (s_r ** 2)
                    overlap = calculate_overlap_area(s_r, m_r, d_v)
                    obs_pct = (overlap / area_s) * 100.0

                ra_s, dec_s, _ = obs_sun[i].radec()
                ra_m, dec_m, _ = obs_moon[i].radec()

                delta_ra = (ra_m.hours - ra_s.hours) * 15.0 * 60.0 * np.cos(dec_s.radians)
                delta_dec = (dec_m.degrees - dec_s.degrees) * 60.0

                frames.append({
                    "dt": dt,
                    "time_str": dt.strftime("%H:%M:%S UTC"),
                    "obs": float(obs_pct),
                    "delta_ra": float(delta_ra),
                    "delta_dec": float(delta_dec),
                    "sun_r": float(np.degrees(s_r) * 60.0),
                    "moon_r": float(np.degrees(m_r) * 60.0)
                })

            self.current_frames = frames
            self.after(0, self._setup_simulation_view)

        threading.Thread(target=worker, daemon=True).start()

    def _setup_simulation_view(self):
        if not self.current_frames:
            return

        self.slider.config(from_=0, to=len(self.current_frames) - 1)
        self.slider.set(len(self.current_frames) // 2)
        
        self._render_frame(int(self.slider.get()))
        self.status_label.config(text="Simulace připravena.")

    def _on_slider_move(self, val):
        idx = int(float(val))
        self._render_frame(idx)

    def _render_frame(self, idx):
        if not self.current_frames or idx >= len(self.current_frames):
            return

        frame = self.current_frames[idx]
        self.lbl_slider_time.config(text=f"Čas: {frame['time_str']} | Zakrytí: {frame['obs']:.1f} %")

        # 1. Graf zakrytí
        self.ax_chart.clear()
        times = [f["dt"] for f in self.current_frames]
        obs = [f["obs"] for f in self.current_frames]

        self.ax_chart.plot(times, obs, color='#d9534f', linewidth=1.5)
        self.ax_chart.axvline(frame["dt"], color='black', linestyle='--', linewidth=1)
        self.ax_chart.set_title("Průběh zakrytí (%)", fontsize=9)
        self.ax_chart.tick_params(axis='x', rotation=30, labelsize=7)
        self.ax_chart.tick_params(axis='y', labelsize=8)

        # 2. 2D Simulace
        self.ax_sim.clear()
        self.ax_sim.set_facecolor('#111111')

        lim = frame["sun_r"] * 2.2
        self.ax_sim.set_xlim(-lim, lim)
        self.ax_sim.set_ylim(-lim, lim)
        self.ax_sim.set_aspect('equal')

        sun = Circle((0, 0), frame["sun_r"], color='#ffcc00', ec='#e67e22', lw=1)
        moon = Circle((frame["delta_ra"], frame["delta_dec"]), frame["moon_r"], color='#222222', ec='#aaaaaa', lw=1)

        self.ax_sim.add_patch(sun)
        self.ax_sim.add_patch(moon)
        self.ax_sim.set_title("Pohled na disk Slunce", fontsize=9)
        self.ax_sim.axis('off')

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = EclipseApp()
    app.mainloop()
