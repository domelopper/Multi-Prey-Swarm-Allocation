# -*- coding: utf-8 -*-
"""
Punto de entrada interactivo del simulador RAOI — Prey-Predator task.

Ejecutar con:
    python main.py

O desde código externo:
    from raoi_simulator.prey_predator import single_run, statistical_run

    # Simulación de prey-predator directa
    report, alive_report, metrics = single_run(
        n_predators=6, n_preys=1,
        r_r=0.1, o_r=0.5, a_r=1.5, i_r=2.0, prey_r_r=0.6,
        animation=True,
    )
"""

import os
import pathlib

from raoi_simulator.prey_predator import single_run as pp_single, statistical_run as pp_stat


# ── Directorio raíz del proyecto (donde está este archivo) ──────────────────
_ROOT = pathlib.Path(__file__).parent.resolve()
_RESULTS_BASE = _ROOT / "Results"


def _ask_folder() -> pathlib.Path:
    """Pregunta al usuario en qué carpeta guardar los resultados y la crea."""
    while True:
        raw = input("Carpeta: ").strip()
        if not raw:
            print("  ✗ Ingresa un número o nombre de carpeta.")
            continue
        folder = _RESULTS_BASE / raw
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Los archivos .npy se guardarán en: {folder}")
        return folder


def _stat_run(fn, replicas: int) -> None:
    """Cambia al directorio de destino, ejecuta fn y restaura el cwd."""
    folder = _ask_folder()
    original_cwd = pathlib.Path.cwd()
    os.chdir(folder)
    try:
        fn(replicas)
    finally:
        os.chdir(original_cwd)


def main() -> None:
    """Menú interactivo del simulador RAOI — Prey-Predator task."""
    menu = """
    ╔══════════════════════════════════════════════════╗
    ║       RAOI Swarm Simulator — Main Menu           ║
    ╠══════════════════════════════════════════════════╣
    ║  Prey-Predator task                              ║
    ║    1. Single simulation                          ║
    ║    2. Statistical run (multiple replicas)        ║
    ╠══════════════════════════════════════════════════╣
    ║    3. Exit                                       ║
    ╚══════════════════════════════════════════════════╝
    """
    print(menu)

    while True:
        try:
            choice = int(input("Option: "))
        except ValueError:
            print("Please enter a number.")
            continue

        if choice == 1:
            pp_single()
            break
        elif choice == 2:
            try:
                replicas = int(input("Number of replicas: "))
            except ValueError:
                print("Invalid number.")
                continue
            _stat_run(pp_stat, replicas)
            break
        elif choice == 3:
            break
        else:
            print("Invalid option. Choose 1–3.")


if __name__ == "__main__":
    main()
