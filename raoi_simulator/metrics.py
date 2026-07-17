# -*- coding: utf-8 -*-
"""
Métricas de desempeño del enjambre RAOI — Prey-Predator task.

Implementa las funciones objetivo para evaluar el desempeño del enjambre
en la tarea de prey-predator.

Convención de nombres:
  Todas las métricas usan nombres descriptivos (sin codigos f1-f6).
  El núcleo común comparte 5 métricas con la misma definición exacta
  usada en el resto de tareas del framework RAOI:
    completion_time  — iteración de finalización de la tarea
    success_fraction — fracción de unidades de tarea completadas
    cohesion_mean    — distancia media al centroide durante la corrida
    swarm_area_mean  — área elíptica media del enjambre durante la corrida
    mean_speed       — velocidad lineal media de todos los robots

Las funciones son stateless: reciben el reporte completo de la simulación
y devuelven diccionarios. Esto permite su uso desde cualquier script de
análisis sin dependencia del estado interno del simulador.

Referencias:
  Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
  Using Influence Factor. J. Intelligent & Robotic Systems.

  Ordaz-Rivas et al. (2021). Flock of Robots with Self-Cooperation for
  Prey-Predator Task. Journal of Intelligent & Robotic Systems.

  Ordaz-Rivas & Torres-Treviño (2024). Improving performance in swarm
  robots using multi-objective optimization. Mathematics and Computers
  in Simulation, 223, 433–457.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

import numpy as np

from . import config


# ══════════════════════════════════════════════════════════════════════════════
# Funciones auxiliares compartidas
# ══════════════════════════════════════════════════════════════════════════════

def cohesion(positions: np.ndarray) -> float:
    """
    Calcula la cohesión del enjambre como distancia media al centroide.

    Una cohesión baja indica enjambre compacto; alta indica dispersión.
    Definición según Ordaz-Rivas et al. (2018).

    Args:
        positions: Posiciones de los robots, shape (N, 2).

    Returns:
        Distancia media al centroide (m).
    """
    centroid  = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return float(np.mean(distances))


def swarm_area(positions: np.ndarray) -> float:
    """
    Estima el área ocupada por el enjambre como elipse de desviaciones estándar.

    Área = π · σ_x · σ_y (aproximación elíptica, Ordaz-Rivas et al. 2018).
    Acotada a config.AREA_COVERAGE_MAX_FRACTION del área total del escenario.

    Args:
        positions: Posiciones de los robots, shape (N, 2).

    Returns:
        Área estimada (m²).
    """
    centroid = np.mean(positions, axis=0)
    std_x    = math.sqrt(float(np.mean((positions[:, 0] - centroid[0])**2)))
    std_y    = math.sqrt(float(np.mean((positions[:, 1] - centroid[1])**2)))
    area     = math.pi * std_x * std_y
    max_area = (config.AREA_LIMITS**2) * config.AREA_COVERAGE_MAX_FRACTION
    return min(area, max_area)


def snapshot_stats(positions: np.ndarray) -> dict:
    """
    Calcula estadísticas de posición para un instante de tiempo único.

    Usado por la simulación para los 3 snapshots canónicos
    (t=0, t_medio, t_final).

    Args:
        positions: Posiciones de los robots, shape (N, 2).

    Returns:
        Dict con 'mean_x', 'mean_y', 'std_x', 'std_y', 'area'.
    """
    mean_x = float(np.mean(positions[:, 0]))
    mean_y = float(np.mean(positions[:, 1]))
    std_x  = float(np.std(positions[:, 0]))
    std_y  = float(np.std(positions[:, 1]))
    area   = swarm_area(positions)
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x":  std_x,
        "std_y":  std_y,
        "area":   area,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de la tarea de PREY-PREDATOR
# ══════════════════════════════════════════════════════════════════════════════

def prey_predator_metrics(
    report:            np.ndarray,
    alive_report:      np.ndarray,
    n_predators:       int,
    n_preys:           int,
    capture_time:      np.ndarray,
    iterations:        int,
    swarm_split_ratio: float = 0.0,
    allocation_mode:   str   = 'emergent',
) -> dict:
    """
    Calcula las métricas de desempeño de la tarea de prey-predator.

    Métricas del núcleo común:
      completion_time  — iteración en que la última presa fue capturada,
                         o `iterations` si no se capturaron todas.
      success_fraction — fracción de presas capturadas al finalizar.
      cohesion_mean    — distancia media al centroide de los predadores.
      swarm_area_mean  — área elíptica media del enjambre de predadores (m²).
      mean_speed       — velocidad lineal media de todos los robots (m/s).

    Métricas específicas de prey-predator (originales):
      n_captured       — número absoluto de presas capturadas (int).
      capture_time_std — desviación estándar de los tiempos de captura
                         individuales de cada presa. Con una presa vale 0.
                         Con varias, un valor bajo indica capturas casi
                         simultáneas (enjambre disperso y paralelo); un valor
                         alto indica capturas escalonadas (enjambre secuencial).
                         Cuantifica empíricamente el tradeoff de repulsión /
                         dispersión documentado en el paper 2021b.
      predator_engagement_fraction — fracción de iteraciones-predador en estado
                         de persecución activa (estado I = influencia). Un valor
                         alto indica que el enjambre pasó la mayor parte del
                         tiempo cazando, no explorando ni dispersándose.
      dispersion       — raíz cuadrada de la varianza total de posición de los
                         predadores (√(σx² + σy²)). Complementa cohesion_mean
                         (que usa distancia al centroide) con una medida de
                         dispersión espacial isotrópica, equivalente a las
                         δx, δy del paper 2021b combinadas en un solo escalar.

    Métricas nuevas para estudio EA vs FA (paper HRFEST 2026):
      inter_capture_interval — intervalo medio entre capturas consecutivas
                         (iteraciones). Con n_preys=1 equivale a completion_time.
                         Un valor bajo indica que el enjambre mantiene presión
                         continua; uno alto revela periodos de reorganización
                         entre capturas. Minimizar.
      cohesion_at_capture — cohesión media de los predadores en el instante
                         exacto de cada captura (m). Mide qué tan compacto
                         estaba el enjambre en el momento del encierro. Un valor
                         bajo indica captura coordinada y concentrada; alto
                         indica captura por robot aislado o grupo pequeño.
      swarm_split_ratio  — fracción de iteraciones-predador con cambio de presa
                         objetivo (solo significativa en EA). Cuantifica la
                         inestabilidad de asignación emergente: en FA siempre
                         vale 0.0. Un valor alto en EA indica que los predadores
                         cambian frecuentemente de presa, lo que puede diluir
                         la presión de encierro sobre cualquier objetivo.

    Args:
        report:            Estado completo, shape (T, N, 8). Predadores en
                           [0, n_predators), presas en [n_predators, N).
        alive_report:      Estado vivo/capturada de cada presa, shape (T, n_preys).
        n_predators:       Número de robots predadores.
        n_preys:           Número de robots presa.
        capture_time:      Iteración de captura de cada presa, shape (n_preys,).
                           Vale `iterations` si nunca fue capturada.
        iterations:        Número real de iteraciones ejecutadas.
        swarm_split_ratio: Fracción de cambios de objetivo calculada en run()
                           (solo EA; FA pasa 0.0 por defecto).
        allocation_mode:   'emergent' | 'focused' — para documentación del dict.

    Returns:
        Dict con las métricas descritas arriba.
    """
    pred_pos = report[:, :n_predators, :2]

    cohs  = np.array([cohesion(pred_pos[t]) for t in range(iterations)])
    areas = np.array([swarm_area(pred_pos[t]) for t in range(iterations)])
    mean_speed = float(np.mean(report[:, :, 5]))

    n_captured = int(np.sum(~alive_report[-1])) if n_preys > 0 else 0

    # completion_time: iteración de la última captura si se completó
    if n_captured == n_preys:
        completion_time = int(np.max(capture_time))
    else:
        completion_time = int(iterations)

    success_fraction = n_captured / max(n_preys, 1)

    # capture_time_std: solo sobre presas efectivamente capturadas
    captured_times = capture_time[capture_time < iterations]
    capture_time_std = float(np.std(captured_times)) if len(captured_times) > 1 else 0.0

    # predator_engagement_fraction: estado I = 4 en predadores
    pred_states = report[:, :n_predators, 7]
    total_pred_iters = iterations * n_predators
    engagement = float(np.sum(pred_states == 4)) / max(total_pred_iters, 1)

    # dispersion: sqrt(var_x + var_y) sobre todas las iteraciones y predadores
    all_pred_x = pred_pos[:, :, 0].flatten()
    all_pred_y = pred_pos[:, :, 1].flatten()
    dispersion = float(math.sqrt(float(np.var(all_pred_x)) + float(np.var(all_pred_y))))

    # ── Métricas nuevas EA vs FA (paper HRFEST 2026) ─────────────────────────

    # inter_capture_interval: tiempo medio entre capturas consecutivas.
    # Se ordena capture_time por orden de captura real (no por índice de presa)
    # para obtener la secuencia temporal verdadera.
    captured_times = np.sort(capture_time[capture_time < iterations])
    if len(captured_times) >= 2:
        inter_capture_interval = float(np.mean(np.diff(captured_times)))
    elif len(captured_times) == 1:
        inter_capture_interval = float(captured_times[0])   # solo hubo 1 captura
    else:
        inter_capture_interval = float(iterations)           # ninguna captura

    # cohesion_at_capture: cohesión del enjambre de predadores en el instante
    # exacto de cada captura. Mide cuán compacto estaba el enjambre al encerrar.
    capture_cohesions = []
    for p in range(n_preys):
        if capture_time[p] < iterations:
            t_cap = int(min(capture_time[p], iterations - 1))
            capture_cohesions.append(cohesion(pred_pos[t_cap]))
    cohesion_at_capture = (
        float(np.mean(capture_cohesions)) if capture_cohesions
        else float(np.mean(cohs))
    )

    return {
        "completion_time":               completion_time,
        "success_fraction":              float(success_fraction),
        "cohesion_mean":                 float(np.mean(cohs)),
        "swarm_area_mean":               float(np.mean(areas)),
        "mean_speed":                    mean_speed,
        "n_captured":                    n_captured,
        "capture_time_std":              capture_time_std,
        "predator_engagement_fraction":  engagement,
        "dispersion":                    dispersion,
        # nuevas métricas HRFEST 2026
        "inter_capture_interval":        inter_capture_interval,
        "cohesion_at_capture":           cohesion_at_capture,
        "swarm_split_ratio":             float(swarm_split_ratio),
        "allocation_mode":               allocation_mode,
    }