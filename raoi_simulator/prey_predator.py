# -*- coding: utf-8 -*-
"""
Simulación de tarea de prey-predator con el modelo RAOI.

En esta variante el enjambre se divide en dos roles:
  - Predadores: comportamiento RAOI completo (R, O, A entre ellos). La
    influencia (I) apunta hacia la presa viva más cercana detectada,
    tratada como un estímulo móvil con el modelo ri+rs (la presa "brilla"
    omnidireccionalmente — análogo a la fuente de luz LDR del paper 2021b).
  - Presas: solo repulsión activa (huyen de predadores, paredes y otras
    presas). Sin orientación, atracción ni influencia. Sin repulsión activa,
    exploran libremente con el mismo mecanismo de random-walk usado en el
    resto del simulador.

Condición de captura — "sectores de escape":
  Alrededor de cada presa viva se acumulan arcos angulares bloqueados:
    - cada predador dentro del radio de cooperación bloquea un arco de
      ancho ENCIRCLEMENT_BLOCK_ANGLE centrado en su dirección hacia la presa;
    - cada pared cercana bloquea un arco de ancho WALL_BLOCK_ANGLE centrado
      en la dirección que apunta hacia afuera del área.
  Se fusionan todos los arcos sobre el círculo de 360°. Si el hueco libre
  máximo es menor a ENCIRCLEMENT_GAP_THRESHOLD y hay al menos
  MIN_PREDATORS_FOR_CAPTURE predadores cooperando, la presa queda atrapada.
  Esta única fórmula cubre tanto "rodeada en el centro" (solo predadores)
  como "arrinconada" (predadores + pared).

Una presa capturada se congela en su posición y pasa a actuar como
obstáculo fijo de repulsión para el resto del enjambre (predadores y
presas vivas), pero deja de ser elegible como objetivo de influencia.

La tarea termina cuando todas las presas han sido capturadas o se alcanza
el límite de iteraciones.

Referencia:
  Ordaz-Rivas et al. (2021). Flock of Robots with Self-Cooperation for
  Prey-Predator Task. Journal of Intelligent & Robotic Systems.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math
import random
import time
from typing import Optional, Callable

import numpy as np
from tqdm import tqdm

from . import config
from . import metrics as mtr
from . import visualization as viz
from .behavior import (
    wrap_angle,
    repulsion_vector,
    combined_direction,
    detect_neighbors,
    detect_influence,
    select_voltage,
)
from .dynamics import DynamicsConstants, integrate_robot


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades internas — geometría angular de encierro
# ══════════════════════════════════════════════════════════════════════════════

def _wall_block_sectors(
    pos: np.ndarray,
    margin: float,
    area_limits: float,
) -> list:
    """
    Detecta paredes cercanas a una posición y devuelve sus sectores de bloqueo.

    A diferencia de environment.detect_walls() (que devuelve puntos virtuales
    para repulsión), esta función devuelve el ángulo de bloqueo de escape:
    la dirección que apunta hacia afuera del área desde cada pared detectada,
    junto con el ancho de bloqueo config.WALL_BLOCK_ANGLE.

    Args:
        pos:         Posición [x, y] de la presa (m).
        margin:      Distancia a la pared a partir de la cual se considera
                     "cerca" (m). Típicamente el radio de repulsión de la presa.
        area_limits: Lado del área cuadrada (m).

    Returns:
        Lista de tuplas (center_angle, half_width) en radianes, una por
        pared detectada. Vacía si ninguna pared está dentro del margen.
    """
    x, y = float(pos[0]), float(pos[1])
    half = config.WALL_BLOCK_ANGLE / 2.0
    sectors = []

    if y < margin:                          # pared sur — afuera es -y
        sectors.append((-math.pi / 2, half))
    if y > area_limits - margin:            # pared norte — afuera es +y
        sectors.append((math.pi / 2, half))
    if x < margin:                          # pared oeste — afuera es -x
        sectors.append((math.pi, half))
    if x > area_limits - margin:            # pared este — afuera es 0
        sectors.append((0.0, half))

    return sectors


def _max_free_gap(blocked_sectors: list) -> float:
    """
    Calcula el hueco angular libre más grande tras fusionar arcos bloqueados.

    Cada sector se expresa como (center_angle, half_width) en radianes.
    Los intervalos se normalizan a [0, 2π), se dividen si cruzan el origen,
    se fusionan por orden, y se considera el wraparound circular al final.

    Args:
        blocked_sectors: Lista de (center_angle, half_width) en radianes.

    Returns:
        Hueco libre máximo en radianes. 2π si no hay sectores bloqueados.
    """
    if not blocked_sectors:
        return 2 * math.pi

    intervals = []
    for center, half in blocked_sectors:
        start = wrap_angle(center - half)
        end   = wrap_angle(center + half)
        if start <= end:
            intervals.append((start, end))
        else:
            # Cruza el origen 0/2π: dividir en dos intervalos
            intervals.append((start, 2 * math.pi))
            intervals.append((0.0, end))

    intervals.sort(key=lambda iv: iv[0])

    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    # Nota: no es necesario fusionar explícitamente a través del límite 0/2π
    # aquí — el cómputo de huecos más abajo usa módulo circular y maneja
    # correctamente el wraparound entre el último y el primer intervalo.

    if len(merged) == 1 and merged[0][0] <= 1e-9 and merged[0][1] >= 2 * math.pi - 1e-9:
        return 0.0  # un solo arco cubre todo el círculo

    max_gap = 0.0
    for k in range(len(merged)):
        nxt = merged[(k + 1) % len(merged)]
        if k + 1 < len(merged):
            gap = nxt[0] - merged[k][1]
        else:
            gap = (merged[0][0] + 2 * math.pi) - merged[k][1]
        max_gap = max(max_gap, gap)

    return max_gap


# ══════════════════════════════════════════════════════════════════════════════
# Simulación principal
# ══════════════════════════════════════════════════════════════════════════════

def run(
    n_predators: int,
    n_preys:     int,
    r_r:         float,
    o_r:         float,
    a_r:         float,
    i_r:         float,
    prey_r_r:    float,
    animation:   bool,
    allocation_mode:            str                                    = 'emergent',
    gap_threshold_deg:          Optional[float]                        = None,
    seed:        Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Ejecuta la simulación de tarea de prey-predator con el modelo RAOI.

    Los predadores usan R/O/A entre ellos (tratando paredes y presas —vivas
    y congeladas— como fuentes de repulsión adicionales, evitando que se
    superpongan físicamente con el cuerpo de la presa) e I apunta a la
    presa viva más cercana detectada mediante el modelo ri+rs. Las presas
    vivas solo usan repulsión (huyen de predadores, paredes y otras presas),
    exploran libremente cuando no hay nada que evadir, y se mueven con
    voltajes escalados por config.PREY_SPEED_FACTOR — ligeramente más
    rápidas que los predadores.

    Estrategias de asignación de objetivo (allocation_mode):
      'emergent' (EA) — cada predador persigue de forma independiente la presa
                        viva más cercana dentro de su radio de influencia. El
                        patrón de asignación emerge de las reglas RAOI locales
                        sin coordinación explícita entre predadores.
      'focused'  (FA) — todos los predadores convergen en una sola presa foco
                        (la primera presa viva en orden de índice). Al ser
                        capturada, el foco se desplaza automáticamente a la
                        siguiente presa viva. Garantiza presión máxima
                        cooperativa sobre un único objetivo en todo momento.

    Args:
        n_predators: Número de robots predadores.
        n_preys:     Número de robots presa.
        r_r:  Radio adicional de repulsión de los predadores (m, se suma
              a ROBOT_BODY_RADIUS).
        o_r:  Radio adicional de orientación de los predadores (m).
        a_r:  Radio adicional de atracción de los predadores (m). También
              se usa como radio de cooperación para la detección de encierro.
        i_r:  Radio sensorial r_I de los predadores hacia la presa (m).
              El rango efectivo de detección es i_r + PREY_STIMULUS_RADIUS.
        prey_r_r: Radio adicional de repulsión de las presas (m, se suma
              a ROBOT_BODY_RADIUS). Recomendado mayor que r_r de predadores
              — ver config.PREY_REPULSION_RADIUS_RECOMMENDED.
        animation: Si True, reproduce la animación Pygame al terminar.
        allocation_mode: Estrategia de asignación de objetivo — 'emergent'
              (EA, comportamiento natural RAOI) o 'focused' (FA, foco
              colectivo secuencial). Default 'emergent'.
        gap_threshold_deg: Hueco angular libre máximo (grados) por debajo
              del cual la presa se considera sin ruta de escape — a menor
              valor, el cerco exigido es más cerrado (más difícil de
              lograr). None → usa config.ENCIRCLEMENT_GAP_THRESHOLD.
        seed: Semilla aleatoria. None → usa config.SEED.
        progress_callback: Función opcional f(t, max_iter, n_captured)
              llamada al final de cada iteración. None → modo silencioso.

    Returns:
        report       : Estado de todos los robots, shape (T, N, 8), con
                       N = n_predators + n_preys. Predadores en índices
                       [0, n_predators), presas en [n_predators, N).
                       Columnas: [x, y, z, theta, theta_deg, v, omega, state].
        alive_report : Estado vivo/capturada de cada presa por iteración,
                       shape (T, n_preys), booleano.
        metrics      : Dict de métricas (ver metrics.prey_predator_metrics).
    """
    _seed = seed if seed is not None else config.SEED
    random.seed(_seed)
    np.random.seed(_seed)
    rng = np.random.default_rng(_seed)

    # ── Parámetros del escenario ──────────────────────────────────────────────
    area_limits = config.AREA_LIMITS
    weights     = config.RAOI_WEIGHTS
    fov         = config.RAOI_FOV
    voltages    = config.VOLTAGE
    # La presa recibe voltajes escalados — ligeramente más rápida que los
    # predadores, que usan los voltajes base sin modificar.
    voltages_prey = {
        k: v * config.PREY_SPEED_FACTOR for k, v in config.VOLTAGE.items()
    }

    r_repulsion_pred = config.ROBOT_BODY_RADIUS + r_r
    r_orientation     = config.ROBOT_BODY_RADIUS + o_r
    r_attraction      = config.ROBOT_BODY_RADIUS + a_r
    r_repulsion_prey  = config.ROBOT_BODY_RADIUS + prey_r_r
    r_s_prey          = config.PREY_STIMULUS_RADIUS

    # La presa "brilla" omnidireccionalmente — el sensor del predador no
    # depende de orientación para detectarla, a diferencia de la influencia
    # frontal usada en aggregation/foraging/farming.
    fov_prey = dict(fov)
    fov_prey["fov_influence"] = 2 * math.pi

    cooperation_radius = r_attraction  # ver config.COOPERATION_RADIUS_USES_ATTRACTION

    gap_threshold = (
        math.radians(gap_threshold_deg)
        if gap_threshold_deg is not None
        else config.ENCIRCLEMENT_GAP_THRESHOLD
    )
    min_preds_for_capture = config.MIN_PREDATORS_FOR_CAPTURE

    N = n_predators + n_preys

    max_iter = max(
        config.PREY_PREDATOR_MIN_ITER,
        n_preys * config.PREY_PREDATOR_ITERS_PER_CAPTURE,
    )

    # ── Constantes dinámicas precalculadas ────────────────────────────────────
    dyn = DynamicsConstants()

    # ── Arrays de estado ──────────────────────────────────────────────────────
    C            = np.zeros((N, 6))
    report_buf   = np.zeros((max_iter, N, 8))
    alive_buf    = np.ones((max_iter, n_preys), dtype=bool)
    state_now_arr= np.zeros(N)
    free_iters   = np.zeros(N, dtype=int)
    state_prev   = np.full(N, -1, dtype=int)

    alive          = np.ones(n_preys, dtype=bool)
    capture_time   = np.full(n_preys, max_iter, dtype=int)
    predator_target= np.full(n_predators, -1, dtype=int)  # índice de presa perseguida

    # ── Posicionamiento inicial ───────────────────────────────────────────────
    # Predadores: cuadrado de spawn en la esquina suroeste, igual criterio
    # que el resto de las tareas del simulador (aggregation/foraging/farming).
    # Presas: cuadrado de spawn centrado en el área, para que la persecución
    # comience desde el centro hacia afuera.
    spawn_sep_pred = max(config.SPAWN_MIN_SEPARATION, 2.0 * r_repulsion_pred)
    spawn_area_min_pred = n_predators * (spawn_sep_pred ** 2) / 0.5
    spawn_side_pred = max(
        config.PREY_PREDATOR_PRED_SPAWN_SIDE,
        math.sqrt(spawn_area_min_pred),
    )
    spawn_side_pred = min(spawn_side_pred, area_limits - r_repulsion_pred)
    spawn_min_pred  = r_repulsion_pred

    for i in range(n_predators):
        if i == 0:
            C[i, 0] = random.uniform(spawn_min_pred, spawn_side_pred)
            C[i, 1] = random.uniform(spawn_min_pred, spawn_side_pred)
        else:
            placed = False
            for _ in range(config.SPAWN_MAX_ATTEMPTS):
                cx = random.uniform(spawn_min_pred, spawn_side_pred)
                cy = random.uniform(spawn_min_pred, spawn_side_pred)
                if all(
                    math.sqrt((cx - C[j, 0])**2 + (cy - C[j, 1])**2) > spawn_sep_pred
                    for j in range(i)
                ):
                    C[i, 0], C[i, 1] = cx, cy
                    placed = True
                    break
            if not placed:
                raise ValueError(
                    f"No se pudo ubicar el predador {i} con separación mínima "
                    f"de {spawn_sep_pred:.3f} m en {config.SPAWN_MAX_ATTEMPTS} "
                    f"intentos. Reduce n_predators, aumenta el área de spawn "
                    f"o reduce r_r."
                )

    spawn_sep_prey = max(config.SPAWN_MIN_SEPARATION, 2.0 * r_repulsion_prey)
    spawn_area_min_prey = n_preys * (spawn_sep_prey ** 2) / 0.5
    spawn_side_prey = max(
        config.PREY_PREDATOR_PREY_SPAWN_SIDE,
        math.sqrt(spawn_area_min_prey),
    )
    spawn_side_prey = min(spawn_side_prey, area_limits - r_repulsion_prey)
    center      = area_limits / 2.0
    half_prey   = spawn_side_prey / 2.0
    prey_lo     = center - half_prey
    prey_hi     = center + half_prey

    for p in range(n_preys):
        gi = n_predators + p
        if p == 0:
            C[gi, 0] = random.uniform(prey_lo, prey_hi)
            C[gi, 1] = random.uniform(prey_lo, prey_hi)
        else:
            placed = False
            for _ in range(config.SPAWN_MAX_ATTEMPTS):
                cx = random.uniform(prey_lo, prey_hi)
                cy = random.uniform(prey_lo, prey_hi)
                if all(
                    math.sqrt((cx - C[n_predators + j, 0])**2 +
                               (cy - C[n_predators + j, 1])**2) > spawn_sep_prey
                    for j in range(p)
                ):
                    C[gi, 0], C[gi, 1] = cx, cy
                    placed = True
                    break
            if not placed:
                raise ValueError(
                    f"No se pudo ubicar la presa {p} con separación mínima "
                    f"de {spawn_sep_prey:.3f} m en {config.SPAWN_MAX_ATTEMPTS} "
                    f"intentos. Reduce n_preys, aumenta el área de spawn "
                    f"o reduce prey_r_r."
                )

    C[:, 2] = 0.0
    C[:, 3] = rng.uniform(0.0, 2 * math.pi, size=N)
    C[:, 4] = 0.0
    C[:, 5] = 0.0

    dir_explore = C[:, 3].copy()

    # ── Tracking de asignación de objetivo (EA vs FA) ─────────────────────────
    # predator_target_prev: objetivo de cada predador en la iteración anterior.
    # Se usa para detectar cambios de presa en EA (swarm_split_ratio).
    predator_target_prev = np.full(n_predators, -1, dtype=int)
    total_target_changes = 0   # acumula cambios de objetivo en EA

    # ── Loop principal ────────────────────────────────────────────────────────
    t = 0
    while np.any(alive) and t < max_iter:
        desired_voltages = np.zeros((N, 2))
        desired_thetas   = np.zeros(N)

        # ── Presa foco para FA (calculada una vez por iteración) ──────────────
        # En FA todos los predadores convergen en la primera presa viva
        # (orden de índice). Al ser capturada, el foco avanza a la siguiente.
        # next() con default=-1 evita StopIteration cuando no quedan presas.
        if allocation_mode == 'focused':
            focus_prey = next((p for p in range(n_preys) if alive[p]), -1)

        for i in range(N):
            is_predator = i < n_predators

            if not is_predator and not alive[i - n_predators]:
                # Presa capturada: congelamiento total. No se recalcula nada
                # (ni vecinos, ni orientación, ni voltaje) — la posición,
                # theta, v y omega quedan exactamente como estaban en el
                # instante de la captura, para siempre.
                continue

            noise        = abs(np.random.normal(0.0, 0.005))
            theta_before = C[i, 3]
            r_repulsion_i = r_repulsion_pred if is_predator else r_repulsion_prey

            # ── Sensado de paredes y vecinos ───────────────────────────────────
            walls = _wall_points(C[i, :2], r_repulsion_i, area_limits)

            if is_predator:
                # Vecinos: otros predadores (R/O/A entre predadores)
                neighbors = detect_neighbors(
                    i, C[:n_predators], r_repulsion_pred, r_orientation,
                    r_attraction, fov, len(walls),
                ) if n_predators > 1 else {
                    "rep_neighbors": [], "ox": [], "oy": [], "ax": [], "ay": [],
                    "n_rep": 0, "n_ori": 0, "n_att": 0,
                }
                # Presas (vivas y congeladas) también repelen a los predadores
                # — evita que el predador se superponga físicamente con el
                # cuerpo de la presa, sea esta capturada o no. CRÍTICO: se
                # filtra por distancia igual que cualquier otra fuente de
                # repulsión — una presa congelada lejana NO debe mantener al
                # predador en zona R indefinidamente (bug detectado: antes se
                # agregaban TODAS las presas congeladas sin filtro de
                # distancia, dejando a todo el enjambre permanentemente en
                # "R" tras la primera captura, sin importar qué tan lejos
                # estuviera cada predador del cadáver).
                extra_repulsion = []
                for p in range(n_preys):
                    pi = n_predators + p
                    d = math.sqrt(
                        (C[i, 0] - C[pi, 0])**2 + (C[i, 1] - C[pi, 1])**2
                    )
                    if d < r_repulsion_pred:
                        extra_repulsion.append([C[pi, 0], C[pi, 1]])
            else:
                # Presa viva: repulsión de predadores + otras presas (vivas y
                # congeladas). Sin orientación ni atracción.
                p_idx = i - n_predators
                rep_sources = []
                for k in range(n_predators):
                    d = math.sqrt(
                        (C[i,0]-C[k,0])**2 + (C[i,1]-C[k,1])**2
                    )
                    if d < r_repulsion_prey:
                        rep_sources.append([C[k, 0], C[k, 1]])
                for q in range(n_preys):
                    if q == p_idx:
                        continue
                    qi = n_predators + q
                    d = math.sqrt(
                        (C[i,0]-C[qi,0])**2 + (C[i,1]-C[qi,1])**2
                    )
                    if d < r_repulsion_prey:
                        rep_sources.append([C[qi, 0], C[qi, 1]])
                neighbors = {
                    "rep_neighbors": rep_sources,
                    "ox": [], "oy": [], "ax": [], "ay": [],
                    "n_rep": len(rep_sources), "n_ori": 0, "n_att": 0,
                }
                extra_repulsion = []

            # ── Construcción de vectores RAOI activos ─────────────────────────
            active = {}
            all_repulsion = neighbors["rep_neighbors"] + walls + extra_repulsion
            if all_repulsion:
                rvx, rvy = repulsion_vector(C[i, :2], all_repulsion)
                r_norm   = max(math.sqrt(rvx**2 + rvy**2), 1e-9)
                active["R"] = (rvx / r_norm, rvy / r_norm)

            inf_dist, inf_angle, inf_detected = 0.0, 0.0, 0

            if is_predator:
                if neighbors["n_ori"] > 0:
                    ovx = sum(neighbors["ox"]); ovy = sum(neighbors["oy"])
                    o_n = max(math.sqrt(ovx**2 + ovy**2), 1e-9)
                    active["O"] = (ovx / o_n, ovy / o_n)

                if neighbors["n_att"] > 0:
                    avx = sum(neighbors["ax"]); avy = sum(neighbors["ay"])
                    a_n = max(math.sqrt(avx**2 + avy**2), 1e-9)
                    active["A"] = (avx / a_n, avy / a_n)

                # ── Asignar presa objetivo según estrategia de asignación ──
                # EA (Emergent Allocation): cada predador persigue de forma
                # independiente la presa viva más cercana que detecta dentro
                # de su radio de influencia. El patrón de asignación surge
                # exclusivamente de las reglas RAOI locales.
                #
                # FA (Focused Allocation): todos los predadores intentan
                # detectar únicamente la presa foco (focus_prey), calculada
                # una vez por iteración antes del loop. Si la presa foco no
                # está dentro del radio de influencia, el predador no activa
                # la zona I (explora/usa O o A hasta detectarla).
                #
                # Inicialización por defecto: ninguna presa detectada.
                # Necesario porque ambas ramas pueden no asignar valores
                # (FA: focus_prey fuera de rango; EA: ninguna presa visible).
                target_idx   = -1
                inf_dist     = 0.0
                inf_angle    = 0.0
                inf_detected = 0

                if allocation_mode == 'focused':
                    if focus_prey >= 0:
                        pi = n_predators + focus_prey
                        d, ang, det = detect_influence(
                            C[i, :2], C[i, 3],
                            C[pi, :2].tolist(), i_r,
                            fov_prey, 0, len(walls),
                            stimulus_radius=r_s_prey,
                        )
                        if det:
                            target_idx     = focus_prey
                            inf_dist       = d
                            inf_angle      = ang
                            inf_detected   = 1
                else:
                    # EA: presa viva más cercana detectada
                    best_dist = math.inf
                    for p in range(n_preys):
                        if not alive[p]:
                            continue
                        pi = n_predators + p
                        # No se suprime por n_rep entre predadores: el momento en
                        # que varios predadores se acercan entre sí es justo
                        # cuando deben mantener la influencia activa para cerrar
                        # el cerco. Suprimir aquí (como en aggregation/foraging)
                        # rompería la cooperación al dispersarlos por repulsión
                        # mutua justo antes de completar la captura.
                        d, ang, det = detect_influence(
                            C[i, :2], C[i, 3],
                            C[pi, :2].tolist(), i_r,
                            fov_prey, 0, len(walls),
                            stimulus_radius=r_s_prey,
                        )
                        if det and d < best_dist:
                            best_dist    = d
                            target_idx   = p
                            inf_dist     = d
                            inf_angle    = ang
                            inf_detected = det

                predator_target[i] = target_idx
                if inf_detected:
                    active["I"] = (math.cos(inf_angle), math.sin(inf_angle))

            # ── Estado dominante para reporte/visualización ───────────────────
            # A diferencia de aggregation/foraging/farming, aquí R e I pueden
            # estar simultáneamente activos por diseño: la cooperación entre
            # predadores requiere que sigan detectando a la presa incluso
            # mientras se evitan entre sí (ver detect_influence con n_rep=0
            # más arriba). Si R tuviera prioridad absoluta para el color,
            # un grupo de predadores muy próximos entre sí se vería "atascado
            # en rojo" aunque internamente sigan persiguiendo a la siguiente
            # presa — por eso aquí I tiene prioridad sobre R para reflejar
            # fielmente que el robot está en modo de persecución.
            if active.get("I"):
                state_now = 4
            elif active.get("R"):
                state_now = 1
            elif active.get("O"):
                state_now = 3
            elif active.get("A"):
                state_now = 2
            else:
                state_now = 0
            state_now_arr[i] = state_now

            # ── Dirección resultante RAOI o exploración libre ─────────────────
            if active:
                desired_thetas[i] = combined_direction(C[i, 3], active, weights)
            else:
                if free_iters[i] < config.EXPLORE_FREE_ITERS:
                    free_iters[i]    += 1
                    desired_thetas[i] = C[i, 3]
                else:
                    dir_explore[i]    = wrap_angle(
                        dir_explore[i] + np.random.normal(0.0, config.EXPLORE_TURN_NOISE)
                    )
                    desired_thetas[i] = dir_explore[i]

            if active and state_prev[i] == 0:
                free_iters[i]  = 0
                dir_explore[i] = wrap_angle(
                    C[i, 3] + np.random.normal(0.0, config.DIREXP_RESET_NOISE)
                )

            if active:
                xT = 0.5 * math.cos(C[i, 3]) + 0.5 * math.cos(desired_thetas[i])
                yT = 0.5 * math.sin(C[i, 3]) + 0.5 * math.sin(desired_thetas[i])
                C[i, 3] = wrap_angle(math.atan2(yT, xT))
            else:
                xT = 0.5 * math.cos(C[i, 3]) + 0.5 * math.cos(dir_explore[i])
                yT = 0.5 * math.sin(C[i, 3]) + 0.5 * math.sin(dir_explore[i])
                C[i, 3]           = wrap_angle(math.atan2(yT, xT))
                desired_thetas[i] = dir_explore[i]

            state_prev[i] = state_now

            desired_voltages[i] = select_voltage(
                active,
                desired_thetas[i], theta_before,
                inf_dist, i_r if is_predator else 0.0, noise,
                voltages if is_predator else voltages_prey,
                stimulus_radius=r_s_prey if is_predator else 0.0,
            )

        # ── Tracking de cambios de objetivo (EA únicamente) ──────────────────
        # Cuenta predadores que tenían un objetivo válido y lo cambiaron por
        # otro objetivo válido distinto. Excluye: predadores sin objetivo en
        # alguno de los dos instantes (ganancia/pérdida de detección no es un
        # "split") y FA (siempre 0 por construcción — todos apuntan al mismo).
        if allocation_mode == 'emergent' and t > 0:
            for k in range(n_predators):
                if (predator_target_prev[k] >= 0
                        and predator_target[k] >= 0
                        and predator_target_prev[k] != predator_target[k]):
                    total_target_changes += 1
        predator_target_prev[:] = predator_target

        # ── Integración dinámica ────────────────────────────────────────────
        for i in range(N):
            p_idx = i - n_predators
            if i >= n_predators and not alive[p_idx]:
                continue  # presa capturada: congelada, no se integra
            C[i], bounced = integrate_robot(
                C[i], desired_voltages[i], dyn,
                r_repulsion_pred if i < n_predators else r_repulsion_prey,
                area_limits,
            )
            if bounced:
                dir_explore[i] = C[i, 3]

        # ── Evaluación de captura (sectores de escape) ─────────────────────
        # El ancho angular que bloquea cada predador depende de su distancia
        # a la presa (geometría real de un obstáculo circular): un predador
        # que está tocando a la presa bloquea hasta 180°, uno lejano bloquea
        # un arco angosto. half_width = asin(r_repulsion_pred / distancia).
        for p in range(n_preys):
            if not alive[p]:
                continue
            pi = n_predators + p
            prey_xy = C[pi, :2]

            blocked = []
            n_cooperating = 0
            for k in range(n_predators):
                d = math.sqrt(
                    (prey_xy[0]-C[k,0])**2 + (prey_xy[1]-C[k,1])**2
                )
                if d < cooperation_radius:
                    n_cooperating += 1
                    ang = wrap_angle(math.atan2(
                        C[k, 1] - prey_xy[1], C[k, 0] - prey_xy[0],
                    ))
                    half_w = math.asin(min(1.0, r_repulsion_pred / max(d, 1e-9)))
                    blocked.append((ang, half_w))

            blocked += _wall_block_sectors(prey_xy, r_repulsion_prey, area_limits)

            free_gap = _max_free_gap(blocked)
            if (free_gap < gap_threshold
                    and n_cooperating >= min_preds_for_capture):
                alive[p]        = False
                capture_time[p] = t

        # ── Registro del estado post-integración ────────────────────────────
        report_buf[t, :, 0] = C[:, 0]
        report_buf[t, :, 1] = C[:, 1]
        report_buf[t, :, 2] = C[:, 2]
        report_buf[t, :, 3] = C[:, 3]
        report_buf[t, :, 4] = np.degrees(C[:, 3])
        report_buf[t, :, 5] = C[:, 4]
        report_buf[t, :, 6] = C[:, 5]
        report_buf[t, :, 7] = state_now_arr
        alive_buf[t]         = alive.copy()

        if progress_callback is not None:
            progress_callback(t, max_iter, int(np.sum(~alive)))

        t += 1

    # ── Recortar buffers al número real de iteraciones ──────────────────────
    report       = report_buf[:t]
    alive_report = alive_buf[:t]

    # ── swarm_split_ratio final (EA: fracción de pred-iters con cambio de presa)
    # t-1 porque el tracking empieza en t=1 (primera iteración no tiene "prev")
    effective_iters  = max(n_predators * max(t - 1, 1), 1)
    swarm_split_ratio = total_target_changes / effective_iters

    # ── Métricas ──────────────────────────────────────────────────────────────
    pp_result = mtr.prey_predator_metrics(
        report            = report,
        alive_report      = alive_report,
        n_predators       = n_predators,
        n_preys           = n_preys,
        capture_time      = capture_time,
        iterations        = t,
        swarm_split_ratio = swarm_split_ratio,
        allocation_mode   = allocation_mode,
    )

    # ── Animación ─────────────────────────────────────────────────────────────
    if animation:
        env = {
            "area_limits": area_limits,
            "n_predators": n_predators,
            "n_preys":     n_preys,
        }
        viz.animate_prey_predator(
            report, alive_report, env,
            interval     = config.ANIMATION_INTERVAL,
            show_zones   = config.SHOW_ZONES,
            show_trail   = config.SHOW_TRAIL,
            trail_length = config.TRAIL_LENGTH,
            save_path    = config.VIDEO_SAVE_PATH,
            screen_size  = config.SCREEN_SIZE,
        )

    return report, alive_report, pp_result


def _wall_points(
    pos: np.ndarray,
    repulsion_radius: float,
    area_limits: float,
) -> list:
    """
    Detecta puntos virtuales de pared dentro del radio de repulsión.

    Idéntico en propósito a environment.detect_walls(); se reimplementa
    localmente para no introducir una dependencia adicional, ya que la
    lógica es identica a la usada en aggregation.py y foraging.py.

    Args:
        pos:              Posición [x, y] del robot (m).
        repulsion_radius: Radio de repulsión efectivo (m).
        area_limits:      Lado del área cuadrada (m).

    Returns:
        Lista de posiciones virtuales [[x, y], ...] de paredes detectadas.
    """
    x, y = float(pos[0]), float(pos[1])
    wall_points = []
    if y < repulsion_radius:
        wall_points.append([x, 0.0])
    if y > area_limits - repulsion_radius:
        wall_points.append([x, area_limits])
    if x < repulsion_radius:
        wall_points.append([0.0, y])
    if x > area_limits - repulsion_radius:
        wall_points.append([area_limits, y])
    return wall_points


# ══════════════════════════════════════════════════════════════════════════════
# Reporte de resultados en consola
# ══════════════════════════════════════════════════════════════════════════════

def _print_results_prey_predator(
    metrics: dict,
    elapsed: float,
    real_iters: int,
    max_iter: int,
    n_predators: int,
    n_preys: int,
    r_r: float, o_r: float, a_r: float, i_r: float, prey_r_r: float,
    allocation_mode: str = 'emergent',
) -> None:
    """Imprime tabla resumen de resultados para una corrida de prey-predator."""
    mode_label = "Emergent (EA)" if allocation_mode == 'emergent' else "Focused  (FA)"
    print("\n" + "═" * 56)
    print("  RESULTS — Prey-Predator task")
    print("═" * 56)
    print(f"  Allocation mode              : {mode_label}")
    print(f"  Predators                    : {n_predators}")
    print(f"  Preys                        : {n_preys}")
    print(f"  Captured                     : {metrics['n_captured']}/{n_preys}")
    print(f"  completion_time              : {metrics['completion_time']} iter "
          f"(of {max_iter} max)")
    print(f"  success_fraction             : {metrics['success_fraction']:.3f}")
    print(f"  Real iters                   : {real_iters}")
    print(f"  r_repulsion   : {r_r:.3f} m    r_orientation: {o_r:.3f} m")
    print(f"  r_attraction  : {a_r:.3f} m    r_influence  : {i_r:.3f} m")
    print(f"  prey_r_r                     : {prey_r_r:.3f} m")
    print(f"  cohesion_mean                : {metrics['cohesion_mean']:.3f} m")
    print(f"  swarm_area_mean              : {metrics['swarm_area_mean']:.3f} m²")
    print(f"  mean_speed                   : {metrics['mean_speed']:.3f} m/s")
    print(f"  capture_time_std             : {metrics['capture_time_std']:.1f} iter")
    print(f"  predator_engagement_fraction : {metrics['predator_engagement_fraction']:.3f}")
    print(f"  dispersion                   : {metrics['dispersion']:.3f} m")
    print("  ── Métricas EA vs FA ───────────────────────────────")
    print(f"  inter_capture_interval       : {metrics['inter_capture_interval']:.1f} iter")
    print(f"  cohesion_at_capture          : {metrics['cohesion_at_capture']:.3f} m")
    print(f"  swarm_split_ratio            : {metrics['swarm_split_ratio']:.4f}")
    print(f"  Elapsed                      : {elapsed:.2f} s")
    print("═" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# API de alto nivel — single_run / statistical_run
# ══════════════════════════════════════════════════════════════════════════════

def single_run(
    n_predators:     Optional[int]   = None,
    n_preys:         Optional[int]   = None,
    r_r:             Optional[float] = None,
    o_r:             Optional[float] = None,
    a_r:             Optional[float] = None,
    i_r:             Optional[float] = None,
    prey_r_r:        Optional[float] = None,
    animation:       Optional[bool]  = None,
    allocation_mode: Optional[str]   = None,
    gap_threshold_deg:               Optional[float] = None,
) -> tuple:
    """
    Ejecuta una simulación de prey-predator con barra de progreso y resumen.

    Todos los parámetros son opcionales: si no se proporcionan, se
    solicitan por consola. El radio de repulsión de la presa muestra el
    valor recomendado de config.PREY_REPULSION_RADIUS_RECOMMENDED. El
    umbral de captura (gap_threshold_deg) se pregunta antes de decidir si
    hay animación, muestra el default de config.py, y acepta Enter para
    conservarlo sin tener que escribirlo cada vez.

    Args:
        n_predators, n_preys: Número de robots por rol.
        r_r, o_r, a_r, i_r:   Radios RAOI de los predadores (m).
        prey_r_r:             Radio adicional de repulsión de la presa (m).
        gap_threshold_deg: Hueco angular libre máximo (grados) para
              considerar a la presa capturada — ver prey_predator.run().
        animation:            True para mostrar animación al terminar.

    Returns:
        (report, alive_report, metrics) — ídem prey_predator.run().
    """
    def _int(prompt, val):
        if val is not None:
            return val
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("  ✗ Ingresa un número entero.")

    def _float(prompt, val):
        if val is not None:
            return val
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("  ✗ Ingresa un número (e.g. 0.5).")

    def _bool(prompt, val):
        if val is not None:
            return val
        while True:
            ans = input(prompt).strip().upper()
            if ans in ("YES", "NO"):
                return ans == "YES"

    def _choice(prompt, val, options: dict, default: str):
        """Presenta opciones numeradas; acepta Enter para el default."""
        if val is not None and val in options.values():
            return val
        opts_str = "  ".join(f"({k}) {v}" for k, v in options.items())
        default_label = next(k for k, v in options.items() if v == default)
        while True:
            raw = input(
                f"{prompt} [{opts_str}]  [default {default_label}, Enter]: "
            ).strip()
            if raw == "":
                return default
            if raw in options:
                return options[raw]
            print(f"  ✗ Opción inválida. Elige: {list(options.keys())}")

    def _float_default(prompt, val, default):
        if val is not None:
            return val
        raw = input(f"{prompt} [default {default:.2f}, Enter para conservarlo]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"  ✗ Valor inválido, usando default {default:.2f}.")
            return default

    n_predators = _int  ("Predators: ",          n_predators)
    n_preys     = _int  ("Preys: ",               n_preys)
    r_r         = _float("Repulsion radius (m): ", r_r)
    o_r         = _float("Orientation radius (m): ", o_r)
    a_r         = _float("Attraction radius (m): ", a_r)
    i_r         = _float("Influence radius (m): ",  i_r)
    prey_r_r    = _float(
        f"Prey repulsion radius (m) "
        f"[recommended ≥ {config.PREY_REPULSION_RADIUS_RECOMMENDED:.2f}]: ",
        prey_r_r,
    )
    allocation_mode = _choice(
        "Allocation mode: ",
        allocation_mode,
        {"1": "emergent", "2": "focused"},
        default="emergent",
    )
    gap_threshold_deg = _float_default(
        "Gap threshold (deg)", gap_threshold_deg,
        math.degrees(config.ENCIRCLEMENT_GAP_THRESHOLD),
    )
    animation   = _bool ("Animation? (YES/NO): ",  animation)

    max_iter = max(
        config.PREY_PREDATOR_MIN_ITER,
        n_preys * config.PREY_PREDATOR_ITERS_PER_CAPTURE,
    )
    print(f"  Max iterations (dynamic): {max_iter}")

    bar = tqdm(
        total  = max_iter,
        desc   = "Prey-Pred ",
        unit   = "iter",
        ncols  = 72,
        colour = "red",
    )
    bar.set_postfix(captured=f"0/{n_preys}")
    last_t = [0]

    def _cb(t: int, total: int, n_captured: int) -> None:
        bar.update(t - last_t[0])
        bar.set_postfix(captured=f"{n_captured}/{n_preys}")
        last_t[0] = t

    t0 = time.time()
    report, alive_report, metrics = run(
        n_predators, n_preys, r_r, o_r, a_r, i_r, prey_r_r, animation,
        allocation_mode   = allocation_mode,
        gap_threshold_deg = gap_threshold_deg,
        progress_callback = _cb,
    )
    bar.update(report.shape[0] - last_t[0])
    bar.set_postfix(captured=f"{metrics['n_captured']}/{n_preys}")
    bar.update(report.shape[0] - last_t[0])
    bar.set_postfix(captured=f"{metrics['n_captured']}/{n_preys}")
    bar.close()
    elapsed = time.time() - t0

    _print_results_prey_predator(
        metrics, elapsed, report.shape[0], max_iter,
        n_predators, n_preys, r_r, o_r, a_r, i_r, prey_r_r,
        allocation_mode=allocation_mode,
    )

    np.save("prey_predator_report",       report)
    np.save("prey_predator_alive_report", alive_report)

    return report, alive_report, metrics


def statistical_run(
    replicas:        int,
    n_predators:     Optional[int]   = None,
    n_preys:         Optional[int]   = None,
    r_r:             Optional[float] = None,
    o_r:             Optional[float] = None,
    a_r:             Optional[float] = None,
    i_r:             Optional[float] = None,
    prey_r_r:        Optional[float] = None,
    allocation_mode: Optional[str]   = None,
    gap_threshold_deg:               Optional[float] = None,
) -> tuple:
    """
    Ejecuta múltiples réplicas de prey-predator con tabla estadística.

    Cada réplica usa config.SEED + replica_index como semilla. El umbral
    de captura (gap_threshold_deg) se fija una sola vez para todas las
    réplicas (ver prey_predator.run() para el detalle).

    Args:
        replicas: Número de réplicas a ejecutar.
        n_predators, n_preys, r_r, o_r, a_r, i_r, prey_r_r: ídem single_run().
        gap_threshold_deg: Hueco angular libre máximo (grados) para
              considerar a la presa capturada, constante durante todas
              las réplicas.

    Returns:
        metrics_report : Métricas [capture_time, cohesion, swarm_area,
                         n_captured] por réplica, shape (R, 4).
        running_mean   : Promedio acumulado, shape (R, 4).
        final_mean     : Media global, shape (4,).
    """
    def _int(p, v):
        if v is not None:
            return v
        while True:
            try:
                return int(input(p))
            except ValueError:
                print("  ✗ Ingresa un número entero.")

    def _float(p, v):
        if v is not None:
            return v
        while True:
            try:
                return float(input(p))
            except ValueError:
                print("  ✗ Ingresa un número (e.g. 0.5).")

    def _float_default(prompt, val, default):
        if val is not None:
            return val
        raw = input(f"{prompt} [default {default:.2f}, Enter para conservarlo]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"  ✗ Valor inválido, usando default {default:.2f}.")
            return default

    def _choice(prompt, val, options: dict, default: str):
        """Presenta opciones numeradas; acepta Enter para el default."""
        if val is not None and val in options.values():
            return val
        opts_str = "  ".join(f"({k}) {v}" for k, v in options.items())
        default_label = next(k for k, v in options.items() if v == default)
        while True:
            raw = input(
                f"{prompt} [{opts_str}]  [default {default_label}, Enter]: "
            ).strip()
            if raw == "":
                return default
            if raw in options:
                return options[raw]
            print(f"  ✗ Opción inválida. Elige: {list(options.keys())}")

    n_predators = _int  ("Predators: ",          n_predators)
    n_preys     = _int  ("Preys: ",               n_preys)
    r_r         = _float("Repulsion radius (m): ", r_r)
    o_r         = _float("Orientation radius (m): ", o_r)
    a_r         = _float("Attraction radius (m): ", a_r)
    i_r         = _float("Influence radius (m): ",  i_r)
    prey_r_r    = _float(
        f"Prey repulsion radius (m) "
        f"[recommended ≥ {config.PREY_REPULSION_RADIUS_RECOMMENDED:.2f}]: ",
        prey_r_r,
    )
    allocation_mode = _choice(
        "Allocation mode: ",
        allocation_mode,
        {"1": "emergent", "2": "focused"},
        default="emergent",
    )
    gap_threshold_deg = _float_default(
        "Gap threshold (deg)", gap_threshold_deg,
        math.degrees(config.ENCIRCLEMENT_GAP_THRESHOLD),
    )

    max_iter = max(
        config.PREY_PREDATOR_MIN_ITER,
        n_preys * config.PREY_PREDATOR_ITERS_PER_CAPTURE,
    )

    metrics_report = np.zeros((replicas, 7))   # +3 métricas EA/FA vs original 4
    running_mean   = np.zeros((replicas, 7))

    rep_bar = tqdm(
        total  = replicas,
        desc   = "Replicas  ",
        unit   = "rep",
        ncols  = 72,
        colour = "cyan",
    )
    rep_bar.set_postfix(captured=f"0/{n_preys}")

    t0 = time.time()

    for r in range(replicas):
        iter_bar = tqdm(
            total  = max_iter,
            desc   = f"  Rep {r+1:>3}/{replicas}",
            unit   = "iter",
            ncols  = 72,
            leave  = False,
            colour = "red",
        )
        last_t = [0]

        def _cb(t: int, total: int, n_captured: int, _b=iter_bar, _l=last_t) -> None:
            _b.update(t - _l[0])
            _b.set_postfix(captured=f"{n_captured}/{n_preys}")
            _l[0] = t

        _, _, metrics = run(
            n_predators, n_preys, r_r, o_r, a_r, i_r, prey_r_r,
            animation         = False,
            allocation_mode   = allocation_mode,
            gap_threshold_deg = gap_threshold_deg,
            seed              = config.SEED + r,
            progress_callback = _cb,
        )
        iter_bar.close()

        metrics_report[r] = [
            metrics["completion_time"],
            metrics["success_fraction"],
            metrics["cohesion_mean"],
            metrics["n_captured"],
            metrics["inter_capture_interval"],
            metrics["cohesion_at_capture"],
            metrics["swarm_split_ratio"],
        ]
        rep_bar.set_postfix(captured=f"{metrics['n_captured']}/{n_preys}")
        rep_bar.update(1)

    rep_bar.close()
    elapsed = time.time() - t0

    final_mean = np.mean(metrics_report, axis=0)
    for r in range(replicas):
        running_mean[r] = np.mean(metrics_report[: r + 1], axis=0)

    print(f"\n  Replicas completed in {elapsed:.2f} s")
    print(f"  Allocation mode          : {'Emergent (EA)' if allocation_mode == 'emergent' else 'Focused (FA)'}")
    print(f"  Mean completion_time     : {final_mean[0]:.2f} iter")
    print(f"  Mean success_fraction    : {final_mean[1]:.3f}")
    print(f"  Mean cohesion_mean       : {final_mean[2]:.3f} m")
    print(f"  Mean n_captured          : {final_mean[3]:.2f}/{n_preys}")
    print(f"  Mean inter_capture_int   : {final_mean[4]:.2f} iter")
    print(f"  Mean cohesion_at_capture : {final_mean[5]:.3f} m")
    print(f"  Mean swarm_split_ratio   : {final_mean[6]:.4f}\n")

    np.save("prey_predator_metrics_report", metrics_report)

    return metrics_report, running_mean, final_mean