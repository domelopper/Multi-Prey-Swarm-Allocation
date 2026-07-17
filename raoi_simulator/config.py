# -*- coding: utf-8 -*-
"""
Configuración global del simulador RAOI — Prey-Predator task.

Todos los parámetros del modelo, del escenario y de la visualización
residen aquí. Ningún módulo debe contener valores literales —
siempre importar desde este archivo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

# ── Reproducibilidad ──────────────────────────────────────────────────────────

SEED: int = 42
"""Semilla global. Cada réplica usa SEED + replica_index para independencia estadística."""

# ── Simulación ────────────────────────────────────────────────────────────────

DT: float = 1.0
"""
Paso de tiempo en segundos.

Los voltajes del modelo dinámico están calibrados para DT=1.0 s.
Reducir DT sin recalibrar los voltajes produce desplazamientos
imperceptibles. Si se requiere un paso más fino, escalar los voltajes
proporcionalmente o recalibrar el modelo.
"""

RK4_SUBSTEPS: int = 10
"""
Subdivisiones internas del integrador RK4.

10 subdivisiones dan precisión equivalente al integrador odeint original.
Reducir a 4–5 para mayor velocidad con menor precisión numérica.
"""

# ── Parámetros RAOI ───────────────────────────────────────────────────────────

RAOI_WEIGHTS: dict = {
    "w_r": 0.8,   # Peso de repulsión  (dominante)
    "w_o": 0.5,   # Peso de orientación
    "w_a": 0.3,   # Peso de atracción
    "w_I": 0.2,   # Peso de influencia
}
"""
Pesos de cada zona de percepción.

Con w_r=1.0 y el resto en 0 se reproduce la prioridad absoluta del paper 2018.
Con pesos intermedios se obtiene comportamiento combinado con inercia adaptativa:
la inercia del robot escala como (1 - max_weight), de modo que el robot con
w_r=0.8 gira más suavemente que con w_r=1.0.
"""

ROBOT_BODY_RADIUS: float = 0.075
"""Radio físico del robot (m). Ajustable según el diseño del robot."""

RAOI_RADII: dict = {
    "r_repulsion":   0.075,
    "r_orientation": 1.0,
    "r_attraction":  2.0,
}
"""
Radios de zona RAOI en metros.

Estos valores se suman a ROBOT_BODY_RADIUS en tiempo de ejecución.
Rangos típicos según la literatura: r_r ∈ [0, 0.2], r_o ∈ [0.4, 0.6], r_a ∈ [1, 1.2].
"""

RAOI_FOV: dict = {
    "fov_repulsion":   math.pi,        # ±90°  — frontal
    "fov_orientation": 2 * math.pi,    # 360°  — omnidireccional
    "fov_attraction":  math.pi,        # ±90°  — frontal
    "fov_influence":   math.pi,        # ±90°  — frontal
}
"""
Campos de visión por zona (radianes).

math.pi  → 180° (semicírculo frontal).
2*math.pi → 360° (omnidireccional).
"""

# ── Modelo dinámico — robot diferencial ──────────────────────────────────────

ROBOT_MASS: float       = 0.38      # Masa total (kg)
ROBOT_INERTIA: float    = 0.005     # Momento de inercia (kg·m²)
ROBOT_D: float          = 0.02      # Distancia centroide → eje de ruedas (m)
ROBOT_WHEEL_R: float    = 0.03      # Radio de rueda (m)
ROBOT_WHEEL_SEP: float  = 0.05      # Semiseparación entre ruedas (m)

MOTOR_Ts: float = 0.434             # Constante de tiempo del motor (s)
MOTOR_Ks: float = 2.745             # Ganancia de velocidad (rad / s·N·m)
MOTOR_Kl: float = 1460.2705         # Ganancia de corriente (rad / s·V)

VOLTAGE: dict = {
    "repulsion":   2.0,   # ~15 cm/s
    "orientation": 2.7,   # ~20 cm/s
    "attraction":  3.7,   # ~30 cm/s
}
"""
Voltajes de referencia por estado RAOI (V).

La influencia usa interpolación lineal entre V_repulsion y V_attraction
en función de la distancia normalizada a la fuente.
"""

# ── Límites físicos del Khepera III ───────────────────────────────────────────

OMEGA_MAX: float    = 10.0   # Velocidad angular máxima (rad/s)
V_MAX_LINEAR: float = 0.5    # Velocidad lineal máxima (m/s)

# ── Controlador de giro proporcional ─────────────────────────────────────────

KP_TURN: float = 0.8
"""
Ganancia del controlador de voltaje diferencial.

v_diff = KP_TURN * theta_error  →  voltajes = [v_base + v_diff, v_base - v_diff]

KP_TURN = 0.0 → voltajes simétricos (sin giro activo)
KP_TURN = 0.5 → giro suave
KP_TURN = 2.0 → giro agresivo
"""

# ── Escenario ─────────────────────────────────────────────────────────────────

AREA_LIMITS: float = 10.0
"""Lado del área cuadrada de simulación (m)."""

# ── Zona de spawn ─────────────────────────────────────────────────────────────

SPAWN_FRACTION: float     = 0.22
"""
Fracción del lado del área usada como zona de spawn inicial.

Usada para derivar PREY_PREDATOR_PRED_SPAWN_SIDE y
PREY_PREDATOR_PREY_SPAWN_SIDE (ver sección Prey-Predator más abajo).
"""

SPAWN_MIN_SEPARATION: float = 0.3
"""
Separación mínima entre robots en el spawn (m).

En tiempo de ejecución se eleva automáticamente a 2*r_repulsion si
este valor resulta menor, para evitar que los robots arranquen en
zona de repulsión mutua.
"""

SPAWN_MAX_ATTEMPTS: int = 10_000
"""Intentos máximos para ubicar un robot con la separación requerida."""

# ── Ruido del sensor de influencia ───────────────────────────────────────────

INFLUENCE_NOISE_AMP: float = 0.05
"""
Amplitud del ruido gaussiano añadido al ángulo percibido de la fuente (rad).

El ruido modela incertidumbre en la dirección del estímulo, no en la
detección (radio de detección permanece determinístico).
"""

# ── Exploración libre ─────────────────────────────────────────────────────────

EXPLORE_FREE_ITERS: int   = 10
"""
Iteraciones consecutivas en estado libre antes de activar el random walk.

Durante las primeras EXPLORE_FREE_ITERS iteraciones sin vecinos el robot
mantiene su última dirección activa. Después de ese umbral, la dirección
acumula un giro gaussiano por iteración (EXPLORE_TURN_NOISE).
"""

EXPLORE_TURN_NOISE: float = 0.15
"""
Amplitud del giro gaussiano por iteración en estado de exploración activa (rad).

0.15 rad ≈ ±8.6° — exploración en arco suave.
Aumentar para cobertura más agresiva del área.
"""

DIREXP_RESET_NOISE: float = 0.1
"""
Perturbación gaussiana aplicada a dirExp al entrar en estado libre (rad).

Rompe la inercia de la última dirección activa sin redirigir bruscamente.
"""

# ── Métricas ──────────────────────────────────────────────────────────────────

AREA_COVERAGE_MAX_FRACTION: float = 0.8
"""Cota superior del área ocupada como fracción del área total."""

# ── Visualización ─────────────────────────────────────────────────────────────

ROBOT_VISUAL_SCALE: float = 1.5
"""
Multiplicador del radio visual del robot en la animación Pygame.

1.5 es apropiado para 20 robots en un área de 10×10 m con pantalla de 800 px.
Aumentar para pocos robots, reducir para enjambres grandes.
"""

SHOW_ROBOT_IDS: bool  = True
"""Mostrar número de ID sobre cada robot en la animación."""

SHOW_ZONES: bool      = False
"""Mostrar radios de percepción RAOI alrededor de cada robot."""

SHOW_TRAIL: bool      = False
"""Mostrar rastro de trayectoria de los últimos TRAIL_LENGTH pasos."""

TRAIL_LENGTH: int     = 15
"""Número de pasos mostrados en el rastro de trayectoria."""

ANIMATION_INTERVAL: int = 100
"""
Milisegundos entre frames en la animación (Pygame).

100 ms = 10 fps (reproducción acelerada).
1000 ms = 1 fps = tiempo real (1 iteración ≡ 1 segundo físico simulado).
"""

SCREEN_SIZE: int       = 800
"""Tamaño de la ventana Pygame en píxeles (cuadrada)."""

VIDEO_SAVE_PATH: str   = "prey_predator.mp4"
"""
Ruta del archivo de video grabado con OpenCV.

Cambiar a None para desactivar la grabación.
Requiere ffmpeg instalado para el codec mp4v.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Parámetros exclusivos de la tarea de PREY-PREDATOR (paper 2021b)
# ══════════════════════════════════════════════════════════════════════════════
# Predadores: RAOI completo (R, O, A entre ellos; I apunta a la presa más
# cercana detectada, tratada como estímulo móvil con el modelo ri+rs).
# Presas: solo repulsión (huyen de predadores, paredes y otras presas);
# sin orientación, atracción ni influencia. Sin repulsión activa, exploran
# libremente con el mismo mecanismo de random-walk que el resto del simulador.
#
# Condición de captura — "sectores de escape": cada predador dentro del
# radio de cooperación bloquea un arco angular alrededor de la presa; cada
# pared cercana bloquea el arco que apunta hacia afuera del área. Si el hueco
# libre máximo entre los arcos bloqueados es menor al umbral, y hay al menos
# MIN_PREDATORS_FOR_CAPTURE predadores cooperando, la presa queda atrapada
# (cubre tanto "rodeada en el centro" como "arrinconada en una pared" con la
# misma fórmula).

PREY_STIMULUS_RADIUS: float = 0.3
"""
Radio de "brillo" de la presa como estímulo de influencia (m). Componente r_s.

La presa emite señal omnidireccional (FOV 360°) — los predadores la detectan
cuando su sensor (i_r, parámetro de run()) toca este radio:
    dist(predador, presa) ≤ i_r + PREY_STIMULUS_RADIUS
Análogo a la baliza de aggregation.
"""

PREY_REPULSION_RADIUS_RECOMMENDED: float = 0.6
"""
Valor recomendado (no forzado) para el radio adicional de repulsión de la
presa (prey_r_r en run()), mostrado como sugerencia en el prompt de consola.

Se recomienda que sea mayor que el r_r típico de los predadores para que la
presa empiece a huir desde más lejos que la distancia de colisión física
entre predadores. El valor final lo decide el usuario en cada corrida.
"""

COOPERATION_RADIUS_USES_ATTRACTION: bool = True
"""
Si True, el radio de cooperación para la detección de encierro (cuántos
predadores "cuentan" para bloquear un sector angular) reutiliza r_attraction
(a_r + ROBOT_BODY_RADIUS) — el mismo radio donde los predadores ya se
consideran agrupados. Evita introducir un parámetro arbitrario adicional.
"""

ENCIRCLEMENT_BLOCK_MODEL: str = "geometric"
"""
Modelo usado para el ancho angular que bloquea cada predador alrededor de
la presa. "geometric" calcula el ancho dinámicamente según la distancia
real: half_width = asin(r_repulsion_predator / distancia), igual que el
ángulo subtendido por un obstáculo circular. Un predador que toca a la
presa bloquea hasta 180°; uno lejano bloquea un arco angosto. Este modelo
reemplazó a un ángulo fijo (40°) que en pruebas resultó insuficiente para
producir capturas incluso con varios predadores en contacto directo —
el bloqueo angular real de un robot depende físicamente de cuán cerca
está, no es una constante independiente de la distancia.
"""

WALL_BLOCK_ANGLE: float = math.radians(180)
"""
Ancho angular (rad) que bloquea una pared detectada cerca de la presa.
Centrado en la dirección que apunta hacia afuera del área (perpendicular
a la pared). Una pared bloquea efectivamente medio círculo de escape.
"""

ENCIRCLEMENT_GAP_THRESHOLD: float = math.radians(20)
"""
Hueco angular libre máximo (rad) por debajo del cual la presa se considera
sin ruta de escape. Si el espacio libre más grande entre todos los arcos
bloqueados (predadores + paredes) es menor a este umbral, la presa es
capturada en esa iteración.

Reducido de 30° a 20° para exigir un cerco genuinamente cerrado — con 30°
una presa todavía tenía un hueco angular notorio (equivalente a ~0.5 m de
ancho libre a 1.5 m de distancia) y se consideraba "atrapada" de forma
permisiva.
"""

MIN_PREDATORS_FOR_CAPTURE: int = 3
"""
Número mínimo de predadores dentro del radio de cooperación requerido para
que la captura sea válida. Evita que una presa arrinconada sin suficientes
predadores cerca se marque como capturada solo por geometría de pared.

Aumentado de 2 a 3 — con solo 2 predadores cooperando el cerco rara vez es
genuinamente inescapable; el paper describe la captura como un esfuerzo
colectivo de varios robots, no de un par.
"""

PREY_SPEED_FACTOR: float = 1.15
"""
Factor multiplicativo aplicado a config.VOLTAGE para las presas, dándoles
una ventaja de velocidad moderada sobre los predadores (que usan los
voltajes base sin escalar). Con los voltajes de referencia actuales
(15-30 cm/s) el resultado se mantiene muy por debajo de V_MAX_LINEAR
(0.5 m/s), así que el incremento se refleja realmente en la velocidad
lineal sin saturar el límite físico.
"""

PREY_PREDATOR_ITERS_PER_CAPTURE: int = 600
"""
Iteraciones estimadas para que el enjambre de predadores capture una presa,
usado en el cálculo dinámico de max_iter:
    max_iter = max(PREY_PREDATOR_MIN_ITER, n_preys * ITERS_PER_CAPTURE)
"""

PREY_PREDATOR_MIN_ITER: int = 1500
"""Piso mínimo de iteraciones para prey-predator."""

PREY_PREDATOR_PRED_SPAWN_SIDE: float = AREA_LIMITS * SPAWN_FRACTION
"""
Lado (m) del cuadrado de spawn de los predadores, en la esquina suroeste
del área — misma proporción SPAWN_FRACTION usada en el resto de las
tareas históricas del simulador.
"""

PREY_PREDATOR_PREY_SPAWN_SIDE: float = AREA_LIMITS * SPAWN_FRACTION * 0.6
"""
Lado (m) del cuadrado de spawn de las presas, centrado en el área
(área_limits/2, área_limits/2). Más pequeño que el de los predadores ya
que normalmente hay menos presas que predadores.
"""