"""Visualisation of the battery-swap station (pygame).

The top of the window is a live scene of the station on a Kigali street: the sky
follows the time of day, the hills light up at night, riders queue on their motos,
and the battery cabinet shows every bay's charge level and charger mode.
The bottom is the operator's dashboard: today's results, grid load, the agent's
last decision and a 24-hour timeline of electricity prices, demand and service.

Colour code used everywhere:
  coral  = depleted battery / lost rider
  grey   = partly charged battery whose charger is off
  volt   = charging (yellow) / fast charging (orange)
  leaf   = ready to swap / rider served
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from environment import custom_env as E

W, H = 1280, 780
SCENE_H = 430

# ---------------------------------------------------------------- palette
INK = (20, 33, 47)          # dashboard background
INK_2 = (30, 46, 63)        # raised surfaces on the dashboard
INK_3 = (48, 67, 88)        # tracks, dividers
MIST = (226, 234, 238)      # main text on ink
SLATE = (146, 163, 178)     # secondary text
VOLT = (247, 201, 46)       # charging, the station's brand colour
FAST = (255, 139, 44)       # fast charging
LEAF = (66, 190, 124)       # ready / served
CORAL = (233, 98, 82)       # depleted / lost
SKYBLUE = (96, 170, 230)    # time marker
IDLE_GREY = (124, 142, 158)  # partly charged, charger off
CABINET = (236, 240, 236)
CABINET_EDGE = (178, 190, 186)

TARIFF_FILL = {E.TARIFF_OFF_PEAK: (36, 84, 77), E.TARIFF_SHOULDER: (92, 82, 44),
               E.TARIFF_PEAK: (112, 50, 52)}
TARIFF_NAME = {E.TARIFF_OFF_PEAK: "Off-peak", E.TARIFF_SHOULDER: "Shoulder", E.TARIFF_PEAK: "Peak"}

ACTION_LABEL = {
    "HOLD": "Hold chargers as they are",
    "START_NORMAL": "Start a normal charge",
    "START_FAST": "Start a fast charge",
    "BOOST": "Boost a battery to fast",
    "SLOW_DOWN": "Slow a fast charge down",
    "PAUSE": "Pause the fullest battery",
}

# Sky colours (top, horizon) through the day
SKY_KEYS = [
    (0.0, (10, 20, 38), (26, 40, 64)),
    (5.0, (16, 26, 52), (52, 52, 86)),
    (6.0, (64, 90, 150), (240, 160, 110)),
    (7.5, (92, 160, 214), (238, 214, 170)),
    (12.0, (72, 150, 218), (186, 222, 240)),
    (16.5, (84, 146, 206), (240, 206, 150)),
    (18.3, (60, 58, 112), (236, 120, 80)),
    (19.5, (22, 30, 62), (70, 58, 96)),
    (21.0, (10, 20, 38), (26, 40, 64)),
    (24.0, (10, 20, 38), (26, 40, 64)),
]

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


def _sky(hour):
    for (h0, t0, b0), (h1, t1, b1) in zip(SKY_KEYS, SKY_KEYS[1:]):
        if h0 <= hour <= h1:
            t = (hour - h0) / (h1 - h0)
            return _lerp(t0, t1, t), _lerp(b0, b1, t)
    return SKY_KEYS[0][1], SKY_KEYS[0][2]


def _daylight(hour):
    """0 at night, 1 at midday."""
    return max(0.0, min(1.0, math.sin((hour - 5.5) / 14.0 * math.pi) * 1.4))


class StationRenderer:
    def __init__(self, render_mode: str, fps: int = 8):
        self.render_mode = render_mode
        self.fps = fps
        pygame.init()
        if render_mode == "human":
            pygame.display.set_caption("Kigali battery-swap station: smart charging agent")
            self.screen = pygame.display.set_mode((W, H))
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": self._font("Barlow-Bold.ttf", 26, bold=True),
            "h": self._font("Barlow-SemiBold.ttf", 18, bold=True),
            "body": self._font("Barlow-Medium.ttf", 16),
            "small": self._font("Barlow-Regular.ttf", 14),
            "tiny": self._font("Barlow-Medium.ttf", 12),
            "num": self._font("BarlowCondensed-SemiBold.ttf", 34),
            "clock": self._font("BarlowCondensed-SemiBold.ttf", 40),
            "soc": self._font("BarlowCondensed-SemiBold.ttf", 20),
        }
        rng = np.random.default_rng(7)
        self.stars = [(int(x), int(y), float(s)) for x, y, s in
                      zip(rng.integers(0, W, 90), rng.integers(0, 230, 90), rng.uniform(0.4, 1, 90))]
        self.city_lights = [(int(x), float(p)) for x, p in zip(rng.integers(40, W - 40, 70), rng.uniform(0, 1, 70))]
        self.history = []     # per step: (served, lost) — for the timeline
        self._prev = (0, 0, -1)

    def _font(self, file, size, bold=False):
        path = FONT_DIR / file
        if path.exists():
            return pygame.font.Font(str(path), size)
        return pygame.font.SysFont("arial", size, bold=bold)

    def _text(self, s, font, color, pos, anchor="topleft"):
        surf = self.fonts[font].render(str(s), True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    # =============================================================== scene
    def _hill(self, base_y, amp, freq, phase, color):
        pts = [(0, SCENE_H)]
        for x in range(0, W + 20, 20):
            y = base_y - amp * (0.6 * math.sin(x * freq + phase) + 0.4 * math.sin(x * freq * 2.3 + phase * 1.7))
            pts.append((x, y))
        pts.append((W, SCENE_H))
        pygame.draw.polygon(self.screen, color, pts)

    def _scene(self, env):
        hour = env.hour
        top, horizon = _sky(hour)
        light = _daylight(hour)
        # sky gradient
        for y in range(0, SCENE_H, 4):
            pygame.draw.rect(self.screen, _lerp(top, horizon, min(y / 300, 1)), (0, y, W, 4))
        # stars at night
        if light < 0.5:
            for x, y, s in self.stars:
                c = _shade((230, 235, 255), s * (1 - light * 2))
                self.screen.set_at((x, y), c)
                if s > 0.85:
                    self.screen.set_at((x + 1, y), c)
        # sun or moon on an arc
        if 6 <= hour <= 18.5:
            t = (hour - 6) / 12.5
            sx, sy = 420 + t * (W - 500), 230 - math.sin(t * math.pi) * 180
            pygame.draw.circle(self.screen, _lerp((255, 214, 120), (255, 244, 214), light), (int(sx), int(sy)), 26)
        else:
            t = ((hour - 18.5) % 24) / 11.5
            mx, my = 420 + t * (W - 500), 200 - math.sin(t * math.pi) * 150
            pygame.draw.circle(self.screen, (226, 232, 244), (int(mx), int(my)), 18)
            pygame.draw.circle(self.screen, _lerp(top, horizon, min((my - 5) / 300, 1)), (int(mx) + 8, int(my) - 5), 16)

        # Kigali's hills: three layers, darker at night
        k = 0.35 + 0.65 * light
        self._hill(250, 30, 0.004, 1.0, _shade((92, 140, 120), k))
        if light < 0.6:   # city lights on the far hill
            for x, p in self.city_lights:
                y = int(250 - 30 * (0.6 * math.sin(x * 0.004 + 1.0) + 0.4 * math.sin(x * 0.0092 + 1.7)) + 8 + p * 30)
                pygame.draw.circle(self.screen, _shade((255, 214, 140), 1 - light), (x, y), 2)
        self._hill(300, 34, 0.0055, 2.4, _shade((62, 120, 86), k))
        self._hill(340, 22, 0.007, 4.0, _shade((48, 98, 68), k))

        # road
        road_y = 360
        pygame.draw.rect(self.screen, _shade((70, 74, 80), 0.55 + 0.45 * light), (0, road_y, W, SCENE_H - road_y))
        pygame.draw.rect(self.screen, _shade((150, 140, 120), 0.5 + 0.5 * light), (0, road_y - 6, W, 6))
        for x in range(0, W, 60):
            pygame.draw.rect(self.screen, _shade((230, 226, 210), 0.5 + 0.5 * light), (x, 404, 32, 4))

        self._station(env, light)
        self._riders(env, light)
        self._scene_overlay(env)

    def _bolt(self, color, cx, cy, s=1.0):
        pts = [(1, -10), (-6, 2), (-1, 2), (-3, 10), (6, -3), (1, -3), (3, -10)]
        pygame.draw.polygon(self.screen, color, [(cx + x * s, cy + y * s) for x, y in pts])

    def _station(self, env, light):
        x0, y0, w, h = 560, 120, 690, 244
        # canopy and its sign
        pygame.draw.rect(self.screen, _shade((34, 52, 70), 0.6 + 0.4 * light), (x0 - 20, y0 - 34, w + 40, 26),
                         border_radius=6)
        pygame.draw.rect(self.screen, VOLT, (x0 - 20, y0 - 12, w + 40, 4))
        self._text("Kigali battery-swap station", "h", VOLT, (x0, y0 - 31))
        self._bolt(VOLT, x0 + w + 4, y0 - 21, 0.9)
        # cabinet (glows at night)
        if light < 0.5:
            glow = pygame.Surface((w + 80, h + 80), pygame.SRCALPHA)
            strength = 1 - light * 2
            for g in range(8):   # soft halo: stacked translucent rings
                pygame.draw.rect(glow, (*VOLT, int(7 * strength)), (g * 5, g * 5, w + 80 - g * 10, h + 80 - g * 10),
                                 border_radius=40 - g * 3)
            self.screen.blit(glow, (x0 - 40, y0 - 40))
        pygame.draw.rect(self.screen, CABINET, (x0, y0, w, h), border_radius=10)
        pygame.draw.rect(self.screen, CABINET_EDGE, (x0, y0, w, h), 2, border_radius=10)

        cols, cw, ch = 5, 124, 104
        for i in range(E.N_SLOTS):
            r, c = divmod(i, cols)
            cx, cy = x0 + 18 + c * (cw + 6), y0 + 14 + r * (ch + 12)
            soc, mode = float(env.soc[i]), int(env.mode[i])
            ready = soc >= E.READY_SOC
            fill = LEAF if ready else (FAST if mode == E.FAST else VOLT if mode == E.NORMAL
                                       else CORAL if soc < 0.3 else IDLE_GREY)
            # locker door
            pygame.draw.rect(self.screen, (214, 221, 218), (cx, cy, cw, ch), border_radius=8)
            # battery, horizontal
            bx, by, bw, bh = cx + 12, cy + 30, cw - 34, 38
            pygame.draw.rect(self.screen, (52, 64, 74), (bx + bw, by + 12, 6, 14), border_radius=2)
            pygame.draw.rect(self.screen, (52, 64, 74), (bx, by, bw, bh), border_radius=6)
            fw = int((bw - 6) * soc)
            if fw > 0:
                pygame.draw.rect(self.screen, fill, (bx + 3, by + 3, fw, bh - 6), border_radius=4)
            self._text(f"{soc * 100:.0f}%", "soc", (255, 255, 255) if soc > 0.35 else MIST,
                       (bx + bw // 2, by + bh // 2), "center")
            # bay label and charger state
            self._text(f"Bay {i + 1}", "tiny", (88, 100, 110), (cx + 10, cy + 8))
            state = "Ready" if ready else {E.IDLE: "Idle", E.NORMAL: "Charging", E.FAST: "Fast charging"}[mode]
            sc = (40, 140, 90) if ready else {E.IDLE: (170, 80, 70) if soc < 0.3 else (100, 112, 124), E.NORMAL: (160, 120, 10),
                                              E.FAST: (200, 96, 20)}[mode]
            self._text(state, "tiny", sc, (cx + 10, cy + ch - 22))
            if mode != E.IDLE:
                self._bolt(FAST if mode == E.FAST else (205, 160, 20), cx + cw - 16, cy + 16, 0.8)
        # power cable to the grid pole
        pole_x = x0 + w + 12
        pygame.draw.rect(self.screen, _shade((90, 80, 70), 0.6 + 0.4 * light), (pole_x, 60, 6, 300))
        pygame.draw.line(self.screen, (40, 40, 40), (pole_x, 70), (W, 90), 2)

    def _rider(self, x, y, color, light):
        dark = _shade((40, 44, 52), 0.7 + 0.3 * light)
        pygame.draw.circle(self.screen, dark, (x - 18, y + 20), 11, 4)       # back wheel
        pygame.draw.circle(self.screen, dark, (x + 20, y + 20), 11, 4)       # front wheel
        pygame.draw.polygon(self.screen, _shade((60, 110, 160), 0.7 + 0.3 * light),
                            [(x - 22, y + 10), (x + 14, y + 10), (x + 24, y), (x + 18, y + 14), (x - 22, y + 16)])
        pygame.draw.line(self.screen, dark, (x + 18, y - 2), (x + 22, y - 14), 3)       # handlebar
        pygame.draw.rect(self.screen, _shade((70, 80, 96), 0.7 + 0.3 * light), (x - 6, y - 22, 12, 24),
                         border_radius=5)                                              # body
        pygame.draw.circle(self.screen, color, (x, y - 30), 9)                          # helmet

    def _riders(self, env, light):
        if not env.queue:
            return
        # queue faces the station; the first rider is closest to it
        for k, wait in enumerate(env.queue[:8]):
            x, y = 520 - k * 62, 378
            col = LEAF if wait == 0 else VOLT if wait == 1 else CORAL
            self._rider(x, y, col, light)
            tag = f"{wait * E.STEP_MINUTES} min"
            r = self._text(tag, "tiny", INK, (x, y - 54), "center")
            pygame.draw.rect(self.screen, (*MIST,), r.inflate(10, 4), border_radius=6)
            self._text(tag, "tiny", INK, (x, y - 54), "center")
        if len(env.queue) > 8:
            self._text(f"+{len(env.queue) - 8} more", "body", MIST, (20, 340))

    def _scene_overlay(self, env):
        hour = env.hour
        hh, mm = int(hour), int(round((hour - int(hour)) * 60)) % 60
        price = E.tariff_at(hour)
        panel = pygame.Surface((340, 92), pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 22, 34, 190), panel.get_rect(), border_radius=14)
        self.screen.blit(panel, (20, 18))
        self._text(f"{hh:02d}:{mm:02d}", "clock", MIST, (36, 22))
        pygame.draw.rect(self.screen, TARIFF_FILL[price], (140, 32, 204, 30), border_radius=15)
        self._text(f"{TARIFF_NAME[price]}, {price:.0f} RWF per kWh", "small", MIST, (242, 47), "center")
        self._text(f"Riders waiting: {len(env.queue)}   Ready batteries: {env.ready_count()}/{E.N_SLOTS}",
                   "small", MIST, (36, 76))

    # =============================================================== dashboard
    def _metric(self, x, y, label, value, color):
        self._text(value, "num", color, (x, y))
        self._text(label, "small", SLATE, (x, y + 40))

    def _dashboard(self, env):
        y0 = SCENE_H
        pygame.draw.rect(self.screen, INK, (0, y0, W, H - y0))
        pygame.draw.rect(self.screen, VOLT, (0, y0, W, 3))
        t = env.totals
        avg_wait = t["wait_steps"] * E.STEP_MINUTES / max(t["served"] + t["lost"], 1)
        metrics = [
            ("Riders served", f"{t['served']}", LEAF),
            ("Riders lost", f"{t['lost']}", CORAL if t["lost"] else MIST),
            ("Average wait", f"{avg_wait:.1f} min", MIST),
            ("Electricity", f"{t['energy_cost'] / 1000:.1f}k RWF", MIST),
            ("Fast-charged", f"{t['fast_kwh']:.0f} kWh", FAST if t["fast_kwh"] else MIST),
            ("Reward today", f"{t['reward']:+.1f}", VOLT),
        ]
        for i, (lab, val, col) in enumerate(metrics):
            self._metric(30 + i * 130, y0 + 20, lab, val, col)

        # grid load meter
        gx, gy = 830, y0 + 24
        p, cap = env.power_kw(), E.GRID_CAP_KW
        self._text("Grid connection", "small", SLATE, (gx, gy))
        self._text(f"{p:.1f} of {cap:.0f} kW", "h", MIST, (gx + 196, gy - 2), "topright")
        pygame.draw.rect(self.screen, INK_3, (gx, gy + 26, 196, 12), border_radius=6)
        frac = min(p / cap, 1.0)
        if frac > 0:
            pygame.draw.rect(self.screen, FAST if frac > 0.85 else VOLT, (gx, gy + 26, int(196 * frac), 12),
                             border_radius=6)
        # last decision
        ax, ay, aw = 1050, y0 + 16, 210
        pygame.draw.rect(self.screen, INK_2, (ax, ay, aw, 64), border_radius=10)
        name = E.ACTION_NAMES[env.last_action]
        ok = env.last_action_valid
        self._text("Agent decision" if ok else "Agent decision, not possible now", "small",
                   SLATE if ok else CORAL, (ax + 12, ay + 8))
        self._text(ACTION_LABEL[name], "body", MIST if ok else SLATE, (ax + 12, ay + 32))

        self._timeline(env, y0 + 106)

    def _timeline(self, env, y0):
        x0, w = 30, W - 60
        self._text("Today at a glance", "h", MIST, (x0, y0))
        self._text("Background shows the electricity price, the line shows expected riders per hour, "
                   "dots show riders served (green) and lost (coral).", "small", SLATE, (x0 + 150, y0 + 3))
        gx, gy, gw, gh = x0, y0 + 32, w, 150
        for hr in range(24):
            pygame.draw.rect(self.screen, TARIFF_FILL[E.tariff_at(hr)],
                             (gx + hr * gw / 24, gy, math.ceil(gw / 24), gh))
        for hr in range(0, 25, 3):
            x = gx + hr * gw / 24
            pygame.draw.line(self.screen, INK, (x, gy), (x, gy + gh), 1)
            self._text(f"{hr:02d}:00", "tiny", SLATE, (min(max(x, gx + 16), gx + gw - 16), gy + gh + 10), "center")
        # demand curve
        demand = E.HOURLY_ARRIVALS * env.demand_scale
        top = E.MAX_HOURLY_ARRIVALS * 1.35
        pts = [(gx + (hr + 0.5) * gw / 24, gy + gh - 10 - demand[hr] / top * (gh - 20)) for hr in range(24)]
        pygame.draw.lines(self.screen, MIST, False, pts, 3)
        # service history
        step_w = gw / E.STEPS_PER_DAY
        for s, (served, lost) in enumerate(self.history):
            x = int(gx + (s + 0.5) * step_w)
            for j in range(served):
                pygame.draw.circle(self.screen, LEAF, (x, gy + gh - 8 - j * 7), 3)
            for j in range(lost):
                pygame.draw.circle(self.screen, CORAL, (x, gy + 10 + j * 7), 3)
        # now
        nx = gx + env.hour / 24 * gw
        pygame.draw.line(self.screen, SKYBLUE, (nx, gy - 4), (nx, gy + gh + 2), 3)
        pygame.draw.circle(self.screen, SKYBLUE, (int(nx), gy - 4), 5)

    # =============================================================== public
    def _track(self, env):
        if env.t == 0:
            self.history = []
            self._prev = (0, 0, 0)
            return
        served, lost, t_prev = self._prev
        if env.t != t_prev:
            self.history.append((env.totals["served"] - served, env.totals["lost"] - lost))
            self._prev = (env.totals["served"], env.totals["lost"], env.t)

    def draw(self, env):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
        self._track(env)
        self._scene(env)
        self._dashboard(env)
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)).copy()

    def close(self):
        pygame.display.quit()
        pygame.quit()
