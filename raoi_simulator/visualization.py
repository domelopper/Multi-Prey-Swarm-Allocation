# -*- coding: utf-8 -*-
"""
**************************************************************************
 *   Copyright (C) 2023 Erick Ordaz                                      *
 *   erick.ordazrv@uanl.edu.mx                                           *
 *                                                                       *
 *   Módulo de visualización — Prey-Predator task                        *
 *                                                                       *
 *   Language: Python                                                    *
 *   Rev: 5.0  (Pygame + OpenCV)                                         *
 **************************************************************************

Estrategia v5.0:
  - Pygame para renderizado: robots como polígonos rotados correctamente
  - OpenCV para grabación de video (.mp4)
  - Robot diferencial: cuerpo + 2 ruedas + flecha de orientación
    Todo construido como vértices rotados → sin artefactos de transformación
  - Capas opcionales desde config.py:
      SHOW_ZONES  → sectores de percepción RAOI
      SHOW_TRAIL  → rastro de los últimos TRAIL_LENGTH pasos
"""

import math
import os
import numpy as np
import pygame
import cv2
from collections import deque
from typing import Optional

from . import config


# ── Paleta ────────────────────────────────────────────────────────────────────

def _hex(h: str) -> tuple:
    """Convierte color hex a tupla RGB para pygame."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# Colores RGB
BG_COLOR       = (255, 255, 255)
BORDER_COLOR   = ( 40,  40,  40)
GRID_COLOR     = (220, 220, 220)
SPAWN_COLOR    = (230, 230, 230)
SPAWN_BORDER   = (160, 160, 160)
INFLUENCE_COL  = (200,  40,  50)
TEXT_COLOR     = ( 30,  30,  30)
TRAIL_ALPHA    = 120   # 0-255

STATE_RGB = {
    0: (160, 160, 160),   # Sin vecinos   — gris
    1: (220,  50,  60),   # Repulsión     — rojo
    2: ( 60, 120, 170),   # Atracción     — azul
    3: ( 38, 160, 140),   # Orientación   — verde
    4: (220, 185,  80),   # Influencia    — dorado
}
STATE_LABELS = {
    0: "Free exploration",
    1: "Repulsion",
    2: "Attraction",
    3: "Orientation",
    4: "Influence",
}
ZONE_RGBA = {
    "repulsion":   (220,  50,  60, 35),
    "orientation": ( 38, 160, 140, 20),
    "attraction":  ( 60, 120, 170, 15),
}


# ── Geometría del robot ───────────────────────────────────────────────────────

def _rotate_points(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rota un array de puntos (N,2) alrededor del origen."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _robot_polygons(cx: float, cy: float, theta: float, r: float) -> dict:
    """
    Calcula los vértices del icono de robot diferencial en píxeles.

    v6.1 — Cuerpo circular con ruedas gruesas y nariz frontal triangular.
    La nariz actúa como indicador de dirección integrado al cuerpo.
    La flecha sale desde la punta de la nariz hacia adelante.

    Args:
        cx, cy: Centro en píxeles.
        theta:  Orientación en radianes (coordenadas de simulación).
        r:      Radio visual del cuerpo en píxeles.

    Returns:
        Dict con:
          'body_center'  — (cx, cy, r) para pygame.draw.circle
          'nose'         — triángulo de nariz frontal (polígono)
          'wheel_l'      — rectángulo rueda izquierda
          'wheel_r'      — rectángulo rueda derecha
          'arrow'        — ((x0,y0), (x1,y1)) flecha desde nariz
    """
    # ── Nariz frontal: triángulo que sale del círculo hacia adelante ──────────
    nose_pts = np.array([
        [ r * 1.45,  0.0      ],   # punta
        [ r * 0.80,  r * 0.38],   # base derecha  (tangente al círculo)
        [ r * 0.80, -r * 0.38],   # base izquierda
    ])

    # ── Ruedas: gruesas y prominentes ─────────────────────────────────────────
    wl  = r * 1.10   # largo (cubre longitud del cuerpo)
    wh  = r * 0.45   # ancho — prominente y visible
    wy  = r * 1.00   # distancia lateral al borde interno

    wl_pts = np.array([          # rueda izquierda (+y local)
        [-wl/2,  wy      ],
        [ wl/2,  wy      ],
        [ wl/2,  wy + wh ],
        [-wl/2,  wy + wh ],
    ])
    wr_pts = np.array([          # rueda derecha (-y local)
        [-wl/2, -wy - wh],
        [ wl/2, -wy - wh],
        [ wl/2, -wy     ],
        [-wl/2, -wy     ],
    ])

    # ── Flecha: desde punta de nariz hacia adelante ───────────────────────────
    arrow_start = np.array([[r * 1.45, 0.0]])
    arrow_end   = np.array([[r * 2.10, 0.0]])

    # Rotar todo con -theta (pygame Y invertido)
    angle  = -theta
    center = np.array([cx, cy])

    nose_r  = _rotate_points(nose_pts,    angle) + center
    wl_r    = _rotate_points(wl_pts,      angle) + center
    wr_r    = _rotate_points(wr_pts,      angle) + center
    arr_s   = (_rotate_points(arrow_start, angle) + center)[0]
    arr_e   = (_rotate_points(arrow_end,   angle) + center)[0]

    return {
        "body_center": (cx, cy, r),
        "nose":        nose_r.tolist(),
        "wheel_l":     wl_r.tolist(),
        "wheel_r":     wr_r.tolist(),
        "arrow":       (arr_s, arr_e),
    }


# ── Conversión mundo → pantalla ───────────────────────────────────────────────

class WorldToScreen:
    """
    Convierte coordenadas de simulación (metros) a píxeles de pantalla.
    Pygame tiene Y=0 arriba, la simulación tiene Y=0 abajo.
    """
    def __init__(self, area_m: float, screen_px: int, margin_px: int = 40):
        self.area_m    = area_m
        self.screen_px = screen_px
        self.margin    = margin_px
        self.drawable  = screen_px - 2 * margin_px
        self.scale     = self.drawable / area_m   # px/m

    def xy(self, xm: float, ym: float) -> tuple[int, int]:
        """Metro → píxel (con Y invertido)."""
        px = int(self.margin + xm * self.scale)
        py = int(self.margin + (self.area_m - ym) * self.scale)
        return px, py

    def r(self, rm: float) -> int:
        """Radio en metros → radio en píxeles."""
        return max(1, int(rm * self.scale))


# ══════════════════════════════════════════════════════════════════════════════
# Animación — Prey-Predator task
# ══════════════════════════════════════════════════════════════════════════════

PREY_ALIVE_COL    = (220,  30, 130)   # magenta vivo  — presa viva, huyendo
PREY_CAPTURED_COL = (140, 140, 140)   # gris apagado  — presa congelada/atrapada


def _draw_pp_hud(
    surf: pygame.Surface,
    font_sm,
    font_lg,
    font_title,
    frame_idx: int,
    total_frames: int,
    n_predators: int,
    n_preys: int,
    n_captured: int,
) -> None:
    """
    Dibuja el HUD específico de prey-predator: título, iteración y contador
    de presas capturadas.

    Args:
        surf:         Superficie pygame.
        font_*:       Fuentes pygame.
        frame_idx:    Índice del frame actual.
        total_frames: Total de frames.
        n_predators:  Número de predadores.
        n_preys:      Número total de presas.
        n_captured:   Presas capturadas hasta el frame actual.
    """
    W = surf.get_width()
    H = surf.get_height()

    hud_surf = pygame.Surface((W, 36), pygame.SRCALPHA)
    hud_surf.fill((240, 240, 240, 210))
    surf.blit(hud_surf, (0, 0))

    title_txt = font_title.render(
        "RAOI Swarm Simulator — Prey-Predator Task", True, TEXT_COLOR)
    surf.blit(title_txt, (10, 6))

    status = f"Iteration {frame_idx+1:>4} / {total_frames}   |   " \
             f"Predators {n_predators}   |   Captured {n_captured}/{n_preys}"
    iter_txt = font_lg.render(status, True, (80, 80, 80))
    surf.blit(iter_txt, (W - iter_txt.get_width() - 12, 10))

    bar_h = 6
    bar_y = H - bar_h - 2
    bar_w = int(W * n_captured / max(n_preys, 1))
    pygame.draw.rect(surf, (210, 210, 210), (0, bar_y, W, bar_h))
    pygame.draw.rect(surf, PREY_CAPTURED_COL, (0, bar_y, bar_w, bar_h))


def animate_prey_predator(
    report: np.ndarray,
    alive_report: np.ndarray,
    env: dict,
    interval: int = 100,
    show_zones: bool = False,
    show_trail: bool = False,
    trail_length: int = 15,
    save_path: str = "prey_predator.mp4",
    screen_size: int = 800,
) -> None:
    """
    Anima la tarea de prey-predator con Pygame y graba el video con OpenCV.

    Los predadores se colorean según su estado RAOI (igual que en las demás
    tareas: gris=libre, rojo=repulsión, azul=atracción, verde=orientación,
    dorado=influencia/persiguiendo). Las presas usan una paleta independiente
    del estado para distinguir su rol: magenta mientras están vivas, gris
    apagado una vez capturadas (congeladas).

    Args:
        report:       Estado de todos los robots, shape (T, N, 8), con
                      predadores en [0, n_predators) y presas en
                      [n_predators, N).
        alive_report: Estado vivo/capturada de cada presa por iteración,
                      shape (T, n_preys), booleano.
        env:          Dict del escenario con claves 'area_limits',
                      'n_predators', 'n_preys'.
        interval:     ms entre frames (default 100 ms ≈ 10 fps).
        show_zones:   Mostrar radios de percepción RAOI (solo predadores).
        show_trail:   Mostrar rastro de trayectoria.
        trail_length: Pasos del rastro.
        save_path:    Ruta del video de salida (.mp4). None para no guardar.
        screen_size:  Tamaño de la ventana en píxeles (cuadrada).
    """
    iterations  = report.shape[0]
    n_predators = env["n_predators"]
    n_preys     = env["n_preys"]
    n_robots    = n_predators + n_preys

    os.environ.setdefault("SDL_VIDEODRIVER", "")
    pygame.init()
    pygame.display.set_caption("RAOI Swarm Simulator — Prey-Predator")

    try:
        screen   = pygame.display.set_mode((screen_size, screen_size))
        headless = False
    except Exception:
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        pygame.init()
        screen   = pygame.Surface((screen_size, screen_size))
        headless = True

    w2s   = WorldToScreen(env["area_limits"], screen_size, margin_px=50)
    clock = pygame.time.Clock()

    pygame.font.init()
    try:
        font_sm    = pygame.font.SysFont("DejaVu Sans", 13)
        font_lg    = pygame.font.SysFont("DejaVu Sans", 14)
        font_title = pygame.font.SysFont("DejaVu Sans Bold", 15)
    except Exception:
        font_sm = font_lg = font_title = pygame.font.Font(None, 16)

    trails = [deque(maxlen=trail_length) for _ in range(n_robots)]

    writer = None
    if save_path:
        fps    = max(1, int(1000 / interval))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps,
                                 (screen_size, screen_size))
        print(f"Recording video → '{save_path}'  ({fps} fps)")

    r_rep_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_repulsion"])
    r_ori_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_orientation"])
    r_att_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_attraction"])
    body_px  = w2s.r(config.ROBOT_BODY_RADIUS * config.ROBOT_VISUAL_SCALE)

    running = True
    paused  = False
    frame   = 0

    while running and frame < iterations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for e2 in pygame.event.get():
                            if e2.type == pygame.KEYDOWN and e2.key == pygame.K_SPACE:
                                paused = False
                            if e2.type == pygame.QUIT:
                                paused = False
                                running = False
                        clock.tick(10)

        screen.fill(BG_COLOR)

        step_g = 1.0
        v = 0.0
        while v <= env["area_limits"]:
            x0, y0 = w2s.xy(v, 0);            x1, y1 = w2s.xy(v, env["area_limits"])
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            x0, y0 = w2s.xy(0, v);            x1, y1 = w2s.xy(env["area_limits"], v)
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            v += step_g

        bx0, by0 = w2s.xy(0, env["area_limits"])
        bx1, by1 = w2s.xy(env["area_limits"], 0)
        pygame.draw.rect(screen, BORDER_COLOR,
                         (bx0, by0, bx1 - bx0, by1 - by0), 2)

        positions    = report[frame, :, :2]
        orientations = report[frame, :, 3]
        states       = report[frame, :, 7]
        n_captured   = int(np.sum(~alive_report[frame]))

        for i in range(n_robots):
            xm, ym = positions[i]
            theta  = orientations[i]
            is_predator = i < n_predators

            if is_predator:
                state = int(states[i])
                color = STATE_RGB.get(state, (128, 128, 128))
            else:
                prey_idx = i - n_predators
                color = (PREY_ALIVE_COL if alive_report[frame, prey_idx]
                         else PREY_CAPTURED_COL)

            cx, cy = w2s.xy(xm, ym)
            trails[i].append((cx, cy))

            if show_zones and is_predator:
                for (r_px, rgba, fov_key) in [
                    (r_rep_px, ZONE_RGBA["repulsion"],   "fov_repulsion"),
                    (r_ori_px, ZONE_RGBA["orientation"], "fov_orientation"),
                    (r_att_px, ZONE_RGBA["attraction"],  "fov_attraction"),
                ]:
                    fov_v  = config.RAOI_FOV[fov_key]
                    zone_s = pygame.Surface((r_px*2+2, r_px*2+2), pygame.SRCALPHA)
                    if fov_v >= 2*math.pi - 0.01:
                        pygame.draw.circle(zone_s, rgba, (r_px+1, r_px+1), r_px)
                    else:
                        start_a = -theta - fov_v / 2
                        pts = [(r_px+1, r_px+1)]
                        steps = max(20, int(math.degrees(fov_v)))
                        for k in range(steps + 1):
                            a = start_a + fov_v * k / steps
                            pts.append((r_px+1 + r_px*math.cos(a),
                                        r_px+1 + r_px*math.sin(a)))
                        if len(pts) > 2:
                            pygame.draw.polygon(zone_s, rgba, pts)
                    screen.blit(zone_s, (cx - r_px - 1, cy - r_px - 1))

            if show_trail and len(trails[i]) >= 2:
                pts_t = list(trails[i])
                n_pts = len(pts_t)
                for k in range(n_pts - 1):
                    alpha = int((k + 1) / n_pts * TRAIL_ALPHA)
                    tr_s  = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(tr_s, (130, 130, 130, alpha),
                                     pts_t[k], pts_t[k+1], 2)
                    screen.blit(tr_s, (0, 0))

            polys  = _robot_polygons(cx, cy, theta, body_px)
            to_int = lambda pts: [(int(x), int(y)) for x, y in pts]
            bx, by, br = polys["body_center"]

            pygame.draw.circle(screen, (185, 185, 185), (bx + 2, by + 2), br)

            wheel_color = (35, 35, 35)
            for wkey in ("wheel_l", "wheel_r"):
                wpts = polys[wkey]
                pygame.draw.polygon(screen, wheel_color, to_int(wpts))
                pygame.draw.polygon(screen, (60, 60, 60), to_int(wpts), 1)

            pygame.draw.circle(screen, color, (bx, by), br)
            pygame.draw.circle(screen, (25, 25, 25), (bx, by), br, 1)

            r_c, g_c, b_c = color
            nose_color = (min(255, r_c+45), min(255, g_c+45), min(255, b_c+45))
            pygame.draw.polygon(screen, nose_color, to_int(polys["nose"]))
            pygame.draw.polygon(screen, (25, 25, 25), to_int(polys["nose"]), 1)

            hub_r = max(2, br // 5)
            pygame.draw.circle(screen, (245, 245, 245), (bx, by), hub_r)
            pygame.draw.circle(screen, (40,  40,  40),  (bx, by), hub_r, 1)

            arr_s, arr_e = polys["arrow"]
            pygame.draw.line(screen, (15, 15, 15),
                             (int(arr_s[0]), int(arr_s[1])),
                             (int(arr_e[0]), int(arr_e[1])), 2)
            tip_x, tip_y = int(arr_e[0]), int(arr_e[1])
            tip_ang  = -theta
            tip_size = max(5, br // 2)
            tip_pts  = [
                (tip_x, tip_y),
                (int(tip_x - tip_size*math.cos(tip_ang - 0.42)),
                 int(tip_y - tip_size*math.sin(tip_ang - 0.42))),
                (int(tip_x - tip_size*math.cos(tip_ang + 0.42)),
                 int(tip_y - tip_size*math.sin(tip_ang + 0.42))),
            ]
            pygame.draw.polygon(screen, (15, 15, 15), tip_pts)

            if config.SHOW_ROBOT_IDS:
                label   = str(i) if is_predator else f"P{i - n_predators}"
                id_surf = font_sm.render(label, True, (20, 20, 20))
                id_x    = cx - id_surf.get_width() // 2
                id_y    = cy - br - id_surf.get_height() - 1
                bg_w    = id_surf.get_width()  + 4
                bg_h    = id_surf.get_height() + 2
                bg_     = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
                bg_.fill((255, 255, 255, 160))
                screen.blit(bg_,     (id_x - 2, id_y - 1))
                screen.blit(id_surf, (id_x,     id_y))

        # ── Leyenda: estados de predador + roles de presa ──────────────────
        leg_x, leg_y = 10, 44
        leg_entries = list(STATE_LABELS.items()) + [
            (-1, "Prey (alive)"), (-2, "Prey (captured)"),
        ]
        leg_surf = pygame.Surface((200, len(leg_entries)*22 + 10), pygame.SRCALPHA)
        leg_surf.fill((255, 255, 255, 200))
        screen.blit(leg_surf, (leg_x - 4, leg_y - 4))
        for state_id, label in leg_entries:
            if state_id == -1:
                rgb = PREY_ALIVE_COL
            elif state_id == -2:
                rgb = PREY_CAPTURED_COL
            else:
                rgb = STATE_RGB[state_id]
            pygame.draw.circle(screen, rgb, (leg_x + 8, leg_y + 8), 7)
            pygame.draw.circle(screen, (30, 30, 30), (leg_x + 8, leg_y + 8), 7, 1)
            txt = font_sm.render(label, True, TEXT_COLOR)
            screen.blit(txt, (leg_x + 20, leg_y + 1))
            leg_y += 22

        _draw_pp_hud(
            screen, font_sm, font_lg, font_title,
            frame, iterations, n_predators, n_preys, n_captured,
        )

        if not headless:
            pygame.display.flip()

        if writer is not None:
            px_array  = pygame.surfarray.array3d(screen)
            frame_bgr = cv2.cvtColor(
                np.transpose(px_array, (1, 0, 2)), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr)

        clock.tick(1000 // max(1, interval))
        frame += 1

        if frame % max(1, iterations // 10) == 0:
            pct = int(frame / iterations * 100)
            print(f"  Animating... {pct}%", end="\r")

    print(f"\nAnimation complete ({frame} frames rendered).")

    if writer is not None:
        writer.release()
        print(f"Video saved: {save_path}")

    pygame.quit()