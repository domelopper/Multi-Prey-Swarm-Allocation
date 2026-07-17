# -*- coding: utf-8 -*-
"""
raoi_simulator — Simulador de enjambre de robots con modelo RAOI.

Paquete principal. Expone la API pública del simulador para uso
desde código externo o desde scripts de análisis.

Este build del paquete está adaptado exclusivamente a la tarea de
Prey-Predator.

Uso básico:
    from raoi_simulator.prey_predator import single_run, statistical_run
    from raoi_simulator import config

Autores:
    Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
    FIME — Universidad Autónoma de Nuevo León

Referencia principal:
    Ordaz-Rivas et al. (2021). Flock of Robots with Self-Cooperation for
    Prey-Predator Task. Journal of Intelligent & Robotic Systems.
"""

from .prey_predator import (
    run             as prey_predator_run,
    single_run      as prey_predator_single_run,
    statistical_run as prey_predator_statistical_run,
)
from . import config, metrics, behavior, dynamics, environment, visualization

__all__ = [
    # Prey-Predator
    "prey_predator_run",
    "prey_predator_single_run",
    "prey_predator_statistical_run",
    # Módulos
    "config",
    "metrics",
    "behavior",
    "dynamics",
    "environment",
    "visualization",
]
