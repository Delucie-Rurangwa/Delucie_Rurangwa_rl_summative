"""2D visualisation of the battery-swap station (pygame).

Layout
  * Header: station name, clock, current electricity price band.
  * Left: the 8 charging bays. Each battery shows its charge level, colour
    (red = depleted, amber = charging, green = ready to swap) and charger mode.
  * Middle: grid-power gauge and the queue of riders waiting for a battery.
  * Right: live statistics (served, lost, energy, cost, reward, last action).
  * Bottom: 24-hour timeline with tariff bands, the demand curve and a marker
    for the current time.
"""

from __future__ import annotations

import numpy as np
import pygame

from environment import custom_env as E

W, H = 1180, 720
BG = (18, 22, 30)
PANEL = (30, 36, 48)
PANEL_LIGHT = (42, 50, 66)
TEXT = (232, 236, 242)
MUTED = (150, 160, 178)
RED = (230, 80, 80)
AMBER = (245, 175, 60)
GREEN = (70, 200, 120)
BLUE = (80, 160, 245)
PURPLE = (170, 120, 245)
TARIFF_COLORS = {E.TARIFF_OFF_PEAK: (40, 90, 70), E.TARIFF_SHOULDER: (110, 95, 40),
                 E.TARIFF_PEAK: (120, 45, 45)}


class StationRenderer:
    def __init__(self, render_mode: str, fps: int = 8):
        self.render_mode = render_mode
        self.fps = fps
        pygame.init()
        if render_mode == "human":
            pygame.display.set_caption("Kigali Battery-Swap Station - RL Smart Charging")
            self.screen = pygame.display.set_mode((W, H))
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("arial", 28, bold=True)
        self.f_big = pygame.font.SysFont("arial", 22, bold=True)
        self.f = pygame.font.SysFont("arial", 17)
        self.f_small = pygame.font.SysFont("arial", 14)

    # ------------------------------------------------------------- helpers
    def _text(self, s, font, color, pos, center=False):
        surf = font.render(str(s), True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def _panel(self, rect, title=None):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=12)
        if title:
            self._text(title, self.f_big, TEXT, (rect[0] + 16, rect[1] + 10))

    @staticmethod
    def _bolt(surface, color, cx, cy, s=1.0):
        pts = [(0, -11), (-6, 1), (-1, 1), (-3, 11), (6, -2), (1, -2), (3, -11)]
        pygame.draw.polygon(surface, color, [(cx + x * s, cy + y * s) for x, y in pts])

    # ------------------------------------------------------------- sections
    def _header(self, env):
        hour = env.hour
        hh, mm = int(hour), int(round((hour - int(hour)) * 60)) % 60
        price = E.tariff_at(hour)
        band = {E.TARIFF_OFF_PEAK: "OFF-PEAK", E.TARIFF_SHOULDER: "SHOULDER", E.TARIFF_PEAK: "PEAK"}[price]
        self._text("Kigali E-Moto Battery-Swap Station", self.f_title, TEXT, (24, 18))
        self._text("RL smart charging controller", self.f, MUTED, (26, 52))
        pygame.draw.rect(self.screen, PANEL, (700, 16, 200, 56), border_radius=10)
        self._text(f"{hh:02d}:{mm:02d}", self.f_title, TEXT, (800, 44), center=True)
        pygame.draw.rect(self.screen, TARIFF_COLORS[price], (915, 16, 240, 56), border_radius=10)
        self._text(f"{band}  {price:.0f} RWF/kWh", self.f_big, TEXT, (1035, 44), center=True)

    def _bays(self, env):
        x0, y0 = 20, 90
        self._panel((x0, y0, 520, 400), "Charging bays")
        bw, bh, gap_x = 78, 130, 20
        for i in range(E.N_SLOTS):
            row, col = divmod(i, 5)
            x = x0 + 24 + col * (bw + gap_x)
            y = y0 + 56 + row * (bh + 46)
            soc, mode = float(env.soc[i]), int(env.mode[i])
            color = GREEN if soc >= E.READY_SOC else (AMBER if mode != E.IDLE else RED if soc < 0.3 else MUTED)
            # battery body + terminal
            pygame.draw.rect(self.screen, PANEL_LIGHT, (x, y, bw, bh), border_radius=8)
            pygame.draw.rect(self.screen, PANEL_LIGHT, (x + bw // 2 - 14, y - 8, 28, 10), border_radius=3)
            fill_h = int((bh - 8) * soc)
            pygame.draw.rect(self.screen, color, (x + 4, y + bh - 4 - fill_h, bw - 8, fill_h), border_radius=6)
            pygame.draw.rect(self.screen, (90, 100, 120), (x, y, bw, bh), 2, border_radius=8)
            self._text(f"{soc * 100:.0f}%", self.f_big, TEXT, (x + bw // 2, y + bh // 2), center=True)
            label = {E.IDLE: "idle", E.NORMAL: "normal", E.FAST: "FAST"}[mode]
            lc = {E.IDLE: MUTED, E.NORMAL: BLUE, E.FAST: PURPLE}[mode]
            self._text(f"Bay {i + 1} · {label}", self.f_small, lc, (x + bw // 2, y + bh + 14), center=True)
            if mode != E.IDLE:
                self._bolt(self.screen, lc, x + bw - 14, y + 16, 1.0 if mode == E.NORMAL else 1.3)

    def _power_and_queue(self, env):
        x0, y0 = 560, 90
        self._panel((x0, y0, 320, 400), "Grid & riders")
        # power gauge
        p, cap = env.power_kw(), E.GRID_CAP_KW
        self._text(f"Grid load  {p:.1f} / {cap:.1f} kW", self.f, TEXT, (x0 + 16, y0 + 48))
        pygame.draw.rect(self.screen, PANEL_LIGHT, (x0 + 16, y0 + 74, 288, 20), border_radius=8)
        frac = p / cap
        pc = GREEN if frac < 0.6 else AMBER if frac < 0.9 else RED
        pygame.draw.rect(self.screen, pc, (x0 + 16, y0 + 74, int(288 * frac), 20), border_radius=8)
        self._text(f"Ready to swap: {env.ready_count()} / {E.N_SLOTS}", self.f, GREEN, (x0 + 16, y0 + 106))
        self._text(f"Riders waiting: {len(env.queue)}", self.f, TEXT, (x0 + 16, y0 + 132))
        # rider icons (colour shows how long they have waited)
        for k, wait in enumerate(env.queue):
            row, col = divmod(k, 5)
            cx = x0 + 44 + col * 56
            cy = y0 + 190 + row * 80
            c = GREEN if wait == 0 else AMBER if wait == 1 else RED
            pygame.draw.circle(self.screen, c, (cx, cy), 11)                       # helmet
            pygame.draw.rect(self.screen, c, (cx - 9, cy + 12, 18, 18), border_radius=4)  # body
            pygame.draw.rect(self.screen, (120, 130, 150), (cx - 22, cy + 30, 44, 8), border_radius=3)  # moto
            pygame.draw.circle(self.screen, (200, 205, 215), (cx - 16, cy + 42), 7, 2)
            pygame.draw.circle(self.screen, (200, 205, 215), (cx + 16, cy + 42), 7, 2)
            self._text(f"{wait * E.STEP_MINUTES}m", self.f_small, MUTED, (cx, cy + 58), center=True)
        if not env.queue:
            self._text("No queue", self.f, MUTED, (x0 + 160, y0 + 260), center=True)

    def _stats(self, env):
        x0, y0 = 900, 90
        self._panel((x0, y0, 260, 400), "Today")
        t = env.totals
        rows = [
            ("Riders served", f"{t['served']}", GREEN),
            ("Riders lost", f"{t['lost']}", RED),
            ("Avg wait", f"{(t['wait_steps'] * E.STEP_MINUTES / max(t['served'] + t['lost'], 1)):.1f} min", TEXT),
            ("Energy used", f"{t['energy_kwh']:.1f} kWh", TEXT),
            ("Electricity cost", f"{t['energy_cost']:,.0f} RWF", AMBER),
            ("Fast-charge energy", f"{t['fast_kwh']:.1f} kWh", PURPLE),
            ("Step reward", f"{env.last_reward:+.2f}", TEXT),
            ("Total reward", f"{t['reward']:+.1f}", BLUE),
        ]
        for k, (name, val, c) in enumerate(rows):
            y = y0 + 52 + k * 34
            self._text(name, self.f, MUTED, (x0 + 16, y))
            self._text(val, self.f, c, (x0 + 244 - self.f.size(val)[0], y))
        act = E.ACTION_NAMES[env.last_action]
        pygame.draw.rect(self.screen, PANEL_LIGHT, (x0 + 14, y0 + 334, 232, 50), border_radius=8)
        self._text("Last action", self.f_small, MUTED, (x0 + 24, y0 + 338))
        self._text(act + ("" if env.last_action_valid else " (invalid)"), self.f_big,
                   TEXT if env.last_action_valid else RED, (x0 + 24, y0 + 356))

    def _timeline(self, env):
        x0, y0, w, h = 20, 506, 1140, 196
        self._panel((x0, y0, w, h), "24-hour view: tariff bands and rider demand")
        gx, gy, gw, gh = x0 + 40, y0 + 50, w - 70, 110
        for hr in range(24):
            c = TARIFF_COLORS[E.tariff_at(hr)]
            pygame.draw.rect(self.screen, c, (gx + hr * gw / 24, gy, gw / 24 + 1, gh))
        demand = E.HOURLY_ARRIVALS * env.demand_scale
        pts = [(gx + (hr + 0.5) * gw / 24, gy + gh - (demand[hr] / (E.MAX_HOURLY_ARRIVALS * 1.3)) * gh)
               for hr in range(24)]
        pygame.draw.lines(self.screen, TEXT, False, pts, 3)
        for hr in range(0, 25, 3):
            self._text(f"{hr:02d}h", self.f_small, MUTED, (gx + hr * gw / 24, gy + gh + 12), center=True)
        nx = gx + env.hour / 24 * gw
        pygame.draw.line(self.screen, BLUE, (nx, gy - 6), (nx, gy + gh + 4), 3)
        self._text("demand (riders/h)", self.f_small, TEXT, (gx + gw - 130, gy + 4))

    # ------------------------------------------------------------- public
    def draw(self, env):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
        self.screen.fill(BG)
        self._header(env)
        self._bays(env)
        self._power_and_queue(env)
        self._stats(env)
        self._timeline(env)
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)).copy()

    def close(self):
        pygame.display.quit()
        pygame.quit()
