# RAOI Swarm Simulator — Prey-Predator Task

Simulador de enjambre de robots basado en el modelo **RAOI** (Repulsión, Atracción, Orientación, Influencia), desarrollado en el grupo de investigación del **Dr. Erick Ordaz-Rivas** en la Facultad de Ingeniería Mecánica y Eléctrica (FIME) de la Universidad Autónoma de Nuevo León (UANL).

El modelo extiende las reglas de Reynolds (1987) y Couzin et al. (2002) incorporando una componente de **Influencia** que permite dirigir al enjambre hacia tareas específicas sin comprometer la autonomía individual ni la naturaleza descentralizada del sistema.

Este build del simulador está adaptado exclusivamente a la tarea de **Prey-Predator**: caza cooperativa de predadores contra presas huidizas, con condición de captura basada en encierro angular.

---

## Contenido

- [Características](#características)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Modelo RAOI](#modelo-raoi)
- [Tarea de Prey-Predator](#tarea-de-prey-predator)
- [Parámetros de configuración](#parámetros-de-configuración)
- [Métricas de desempeño](#métricas-de-desempeño)
- [Visualización](#visualización)
- [Referencias](#referencias)

---

## Características

- **Modelo RAOI completo** con cuatro zonas de percepción: Repulsión, Orientación, Atracción e Influencia.
- **Modelo dinámico realista**: robot diferencial con masa, inercia y motores DC.
- **Integrador RK4 vectorizado** con subdivisiones configurables para precisión numérica.
- **Tarea de Prey-Predator**: predadores con RAOI completo (I apunta a la presa viva más cercana); presas con repulsión pura (huyen de predadores, paredes y otras presas).
- **Condición de captura por "sectores de escape"**: cada predador cercano y cada pared bloquean un arco angular alrededor de la presa; si el hueco libre restante cae bajo un umbral, la presa queda atrapada.
- **Dos estrategias de asignación de objetivo**: `emergent` (cada predador persigue de forma independiente a la presa más cercana) y `focused` (todos los predadores convergen en un único foco).
- **Detección ri+rs**: modelo de sensor + radio físico del estímulo (la presa "brilla" omnidireccionalmente).
- **Gestión de paredes** como vecinos virtuales RAOI, sin lógica especial.
- **Campo de visión (FOV) configurable** por zona de percepción.
- **Inercia adaptativa**: el peso de inercia escala automáticamente con el peso dominante.
- **Animación en tiempo real** con Pygame y grabación de video con OpenCV.
- **Métricas descriptivas** del núcleo común más métricas específicas de captura cooperativa.
- **Barra de progreso** en consola para simulaciones simples y estadísticas.
- **API pura** (`run()`) sin I/O, lista para conectar con optimizadores externos.
- **Semillas independientes por réplica** para reproducibilidad estadística.

---

## Estructura del proyecto

```
raoi_simulator/
├── raoi_simulator/
│   ├── __init__.py          API pública: prey_predator_run/single_run/statistical_run
│   ├── config.py            Todos los parámetros (modelo, escenario, visualización)
│   ├── dynamics.py          Modelo dinámico RK4 vectorizado + DynamicsConstants
│   ├── behavior.py          Reglas RAOI stateless (vectores, dirección, voltaje)
│   ├── environment.py       Detección de paredes como vecinos virtuales
│   ├── metrics.py           Métricas descriptivas compartidas + de prey-predator
│   ├── prey_predator.py     Tarea de caza cooperativa depredador-presa
│   └── visualization.py     Animación Pygame + grabación OpenCV
├── main.py                  Menú interactivo
├── generar_reporte_excel.py Genera un .xlsx a partir de las corridas en Results/
├── Requirements.txt
└── README.md
```

### Responsabilidad de cada módulo

| Módulo | Responsabilidad |
|--------|----------------|
| `config.py` | Único punto de configuración. Ningún módulo contiene valores literales. |
| `behavior.py` | Cálculo de vectores RAOI y selección de voltajes. Completamente stateless. |
| `dynamics.py` | Integración RK4 del modelo diferencial. Stateless, vectorizado sobre N robots. |
| `environment.py` | Detección de paredes como puntos virtuales de repulsión. |
| `metrics.py` | Métricas descriptivas compartidas + específicas de prey-predator. Stateless. |
| `prey_predator.py` | Loop principal de caza cooperativa con encierro angular + runners. |
| `visualization.py` | Toda la lógica de renderizado, separada de la lógica de simulación. |

---

## Instalación

### Requisitos

- Python 3.10 o superior
- Conda (recomendado) o pip

### Con Conda

```bash
conda create -n SwarmRobotics python=3.11
conda activate SwarmRobotics
pip install -r Requirements.txt
```

### Con pip

```bash
pip install -r Requirements.txt
```

### Verificar instalación

```bash
python main.py
```

Debe aparecer el menú interactivo del simulador.

---

## Uso rápido

### Menú interactivo

```bash
python main.py
```

```
╔══════════════════════════════════════════════════╗
║       RAOI Swarm Simulator — Main Menu           ║
╠══════════════════════════════════════════════════╣
║  Prey-Predator task                              ║
║    1. Single simulation                          ║
║    2. Statistical run (multiple replicas)        ║
╠══════════════════════════════════════════════════╣
║    3. Exit                                       ║
╚══════════════════════════════════════════════════╝
```

### Desde código

```python
from raoi_simulator.prey_predator import single_run as pp_run

report, alive_report, metrics = pp_run(
    n_predators=6, n_preys=2,
    r_r=0.15, o_r=0.5, a_r=1.5, i_r=2.0, prey_r_r=0.6,
    animation=True,
)
print(f"Capturadas: {metrics['n_captured']}/{2}")

# API pura para optimizadores externos (sin I/O)
from raoi_simulator.prey_predator import run

report, alive_report, metrics = run(
    n_predators=6, n_preys=2,
    r_r=0.15, o_r=0.5, a_r=1.5, i_r=2.0, prey_r_r=0.6,
    animation=False,
    allocation_mode="emergent",
    seed=42,
)
```

---

## Modelo RAOI

Cada robot percibe su entorno en cuatro zonas concéntricas definidas por distancia euclidiana:

| Zona | Condición | Comportamiento |
|------|-----------|---------------|
| Repulsión | `d_ij < r_r` | Alejarse — prioridad máxima absoluta |
| Orientación | `r_r ≤ d_ij < r_o` | Alinearse con la dirección de vecinos |
| Atracción | `r_o ≤ d_ij < r_a` | Acercarse a vecinos lejanos |
| Influencia | señal ambiental | Modular hacia la tarea específica |

### Vectores de comportamiento

```
R_i = Σ (p_i - p_j) / ||p_i - p_j||    # repulsión
O_i = Σ v_j / ||v_j||                   # orientación
A_i = Σ (p_j - p_i) / ||p_j - p_i||    # atracción
I_i = f(señal_ambiente)                  # influencia
```

Todos los vectores se normalizan antes de componer. Se usa `max(norm, 1e-9)` para evitar división por cero.

### Prioridad de zonas

```python
if repulsion_neighbors:
    delta_v = r_vector                       # solo repulsión
elif orientation_neighbors or attraction_neighbors:
    delta_v = w_o * o_vector + w_a * a_vector
    delta_v += w_I * influence_vector        # influencia siempre suma
else:
    delta_v = influence_vector               # exploración libre
```

### Inercia adaptativa

El robot no cambia de dirección instantáneamente. La inercia se calcula como `1 - max_weight`, donde `max_weight` es el peso normalizado de la zona dominante. Con `w_r = 1.0` la prioridad de repulsión es absoluta (inercia = 0); con pesos intermedios, el giro es gradual.

### Modelo dinámico

Robot diferencial con motores DC (Khepera III). Integración RK4 con `config.RK4_SUBSTEPS` subdivisiones internas:

```
v_c     = (r/2) * (ω_r + ω_l)
ω_c     = (r / 2R) * (ω_r - ω_l)
x(t+1)  = x(t) + v_c · cos(θ) · dt
y(t+1)  = y(t) + v_c · sin(θ) · dt
θ(t+1)  = θ(t) + ω_c · dt
```

---

## Tarea de Prey-Predator

El enjambre se divide en dos roles:

- **Predadores**: comportamiento RAOI completo (R, O, A entre ellos). La influencia (I) apunta hacia la presa viva más cercana detectada, tratada como un estímulo móvil con el modelo `ri+rs` (la presa "brilla" omnidireccionalmente).
- **Presas**: solo repulsión activa (huyen de predadores, paredes y otras presas). Sin orientación, atracción ni influencia. Cuando no hay nada que evadir, exploran libremente con el mismo mecanismo de random-walk del resto del simulador. Se mueven con voltajes escalados por `PREY_SPEED_FACTOR`, ligeramente más rápidas que los predadores.

### Condición de captura — "sectores de escape"

Alrededor de cada presa viva se acumulan arcos angulares bloqueados:

- cada predador dentro del radio de cooperación bloquea un arco centrado en su dirección hacia la presa (ancho geométrico, según qué tan cerca está);
- cada pared cercana bloquea un arco de ancho `WALL_BLOCK_ANGLE` centrado en la dirección que apunta hacia afuera del área.

Se fusionan todos los arcos sobre el círculo de 360°. Si el hueco libre máximo es menor a `ENCIRCLEMENT_GAP_THRESHOLD` y hay al menos `MIN_PREDATORS_FOR_CAPTURE` predadores cooperando, la presa queda atrapada. Esta única fórmula cubre tanto "rodeada en el centro" (solo predadores) como "arrinconada" (predadores + pared).

Una presa capturada se congela en su posición y pasa a actuar como obstáculo fijo de repulsión para el resto del enjambre, pero deja de ser elegible como objetivo de influencia. La tarea termina cuando todas las presas han sido capturadas o se alcanza el límite de iteraciones.

### Estrategias de asignación de objetivo

| Modo | Comportamiento |
|------|-----------------|
| `emergent` (EA) | Cada predador persigue de forma independiente la presa viva más cercana dentro de su radio de influencia. El patrón de asignación emerge de las reglas RAOI locales sin coordinación explícita. |
| `focused` (FA) | Todos los predadores convergen en una sola presa foco (la primera presa viva en orden de índice). Al ser capturada, el foco se desplaza automáticamente a la siguiente presa viva. |

### Límite de tiempo dinámico

```
max_iter = max(PREY_PREDATOR_MIN_ITER, n_preys * PREY_PREDATOR_ITERS_PER_CAPTURE)
```

**Métrica principal:** `n_captured` / `completion_time`.

---

## Parámetros de configuración

Todos los parámetros residen en `raoi_simulator/config.py`. Los más relevantes:

### Modelo RAOI

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `RAOI_RADII["r_repulsion"]` | `0.075 m` | Radio de zona de repulsión |
| `RAOI_RADII["r_orientation"]` | `1.0 m` | Radio de zona de orientación |
| `RAOI_RADII["r_attraction"]` | `2.0 m` | Radio de zona de atracción |
| `RAOI_WEIGHTS["w_r"]` | `0.8` | Peso de repulsión |
| `RAOI_WEIGHTS["w_o"]` | `0.5` | Peso de orientación |
| `RAOI_WEIGHTS["w_a"]` | `0.3` | Peso de atracción |
| `RAOI_WEIGHTS["w_I"]` | `0.2` | Peso de influencia |

### Simulación

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `DT` | `1.0 s` | Paso de tiempo |
| `RK4_SUBSTEPS` | `10` | Subdivisiones del integrador |
| `SEED` | `42` | Semilla aleatoria global |
| `AREA_LIMITS` | `10.0 m` | Lado del área cuadrada de simulación |

### Prey-Predator

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `PREY_STIMULUS_RADIUS` | `0.3 m` | Radio de "brillo" de la presa como estímulo |
| `PREY_REPULSION_RADIUS_RECOMMENDED` | `0.6 m` | Valor sugerido para `prey_r_r` en el prompt de consola |
| `MIN_PREDATORS_FOR_CAPTURE` | `3` | Predadores mínimos cooperando para validar una captura |
| `ENCIRCLEMENT_GAP_THRESHOLD` | `20°` | Hueco angular libre máximo antes de considerar a la presa atrapada |
| `WALL_BLOCK_ANGLE` | `180°` | Ancho angular que bloquea una pared cercana |
| `PREY_SPEED_FACTOR` | `1.15` | Ventaja de velocidad de las presas sobre los predadores |
| `PREY_PREDATOR_MIN_ITER` | `1500` | Piso mínimo de iteraciones |
| `PREY_PREDATOR_ITERS_PER_CAPTURE` | `600` | Iteraciones estimadas por captura, usado en el límite dinámico |

### Visualización

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `SHOW_ZONES` | `False` | Mostrar radios de percepción RAOI |
| `SHOW_TRAIL` | `False` | Mostrar rastro de trayectoria |
| `SHOW_ROBOT_IDS` | `True` | Mostrar ID numérico sobre cada robot |
| `SCREEN_SIZE` | `800 px` | Tamaño de la ventana Pygame |
| `ANIMATION_INTERVAL` | `100 ms` | Tiempo entre frames (≈10 fps) |
| `VIDEO_SAVE_PATH` | `"prey_predator.mp4"` | Ruta de grabación. `None` desactiva |

---

## Métricas de desempeño

Todas las métricas usan nombres descriptivos.

### Núcleo común

| Métrica | Descripción | Objetivo |
|---|---|---|
| `completion_time` | Iteración en que la última presa fue capturada, o `max_iter` si no se completó | Minimizar |
| `success_fraction` | Fracción de presas capturadas al finalizar | Maximizar |
| `cohesion_mean` | Distancia media al centroide de los predadores durante la corrida (m) | Minimizar |
| `swarm_area_mean` | Área elíptica media del enjambre de predadores: π·σx·σy (m²) | Minimizar |
| `mean_speed` | Velocidad lineal media de todos los robots durante la corrida (m/s) | Minimizar |

### Específicas de Prey-Predator

| Métrica | Descripción | Objetivo |
|---|---|---|
| `n_captured` | Número absoluto de presas capturadas | Maximizar |
| `capture_time_std` | Desviación estándar de los tiempos de captura individuales — bajo: capturas en paralelo; alto: capturas secuenciales | Minimizar |
| `predator_engagement_fraction` | Fracción de iteraciones-predador en estado de persecución activa (estado I) | Maximizar |
| `dispersion` | √(σx² + σy²) de posiciones de predadores — dispersión espacial isotrópica del enjambre | — |
| `inter_capture_interval` | Intervalo medio entre capturas consecutivas (iteraciones) | Minimizar |
| `cohesion_at_capture` | Cohesión media de los predadores en el instante exacto de cada captura (m) | — |
| `swarm_split_ratio` | Fracción de iteraciones-predador con cambio de presa objetivo (solo significativa en modo `emergent`) | Minimizar |

---

## Visualización

### Colores por estado RAOI (predadores)

| Color | Estado |
|-------|--------|
| ⚫ Gris | Sin vecinos — exploración libre |
| 🔴 Rojo | Repulsión activa |
| 🔵 Azul | Atracción activa |
| 🟢 Verde | Orientación activa |
| 🟡 Dorado | Influencia / persiguiendo |

### Colores de las presas

| Color | Estado |
|-------|--------|
| 🟣 Magenta | Viva, huyendo |
| ⚫ Gris apagado | Capturada / congelada |

### Grabación de video

La animación se graba automáticamente en `prey_predator.mp4` si `VIDEO_SAVE_PATH` no es `None`. Requiere `ffmpeg` instalado. Para desactivar la grabación:

```python
# En config.py
VIDEO_SAVE_PATH = None
```

---

## Diagnóstico de comportamientos

| Síntoma | Causa probable | Parámetro a revisar |
|---------|---------------|---------------------|
| Robots se aglutinan | `r_repulsion` muy pequeño o `w_a >> w_r` | `RAOI_RADII`, `RAOI_WEIGHTS` |
| Enjambre se dispersa | `r_attraction` insuficiente para N robots | `RAOI_RADII["r_attraction"]` |
| No responde a influencia | `w_I` muy bajo o robots en repulsión constante | `RAOI_WEIGHTS["w_I"]` |
| Oscilaciones / vibración | `dt` muy grande o `KP_TURN` alto | `DT`, `KP_TURN` |
| Robots estáticos | Ningún vecino en ninguna zona | Densidad de robots vs. radios |
| Presas nunca son capturadas | `ENCIRCLEMENT_GAP_THRESHOLD` muy exigente o pocos predadores | `ENCIRCLEMENT_GAP_THRESHOLD`, `MIN_PREDATORS_FOR_CAPTURE` |
| Capturas casi instantáneas | `prey_r_r` insuficiente frente a `r_r` de los predadores | `prey_r_r`, `PREY_REPULSION_RADIUS_RECOMMENDED` |

---

## Referencias

```bibtex
@article{ordaz2018collective,
  title   = {Collective Tasks for a Flock of Robots Using Influence Factor},
  author  = {Ordaz-Rivas, Erick and Rodriguez-Liñan, America and Torres-Treviño, Luis},
  journal = {Journal of Intelligent \& Robotic Systems},
  year    = {2018}
}

@article{ordaz2021flock,
  title   = {Flock of Robots with Self-Cooperation for Prey-Predator Task},
  author  = {Ordaz-Rivas, Erick and Torres-Treviño, Luis},
  journal = {Journal of Intelligent \& Robotic Systems},
  year    = {2021}
}

@article{ordaz2024improving,
  title   = {Improving performance in swarm robots using multi-objective optimization},
  author  = {Ordaz-Rivas, Erick and Torres-Treviño, Luis},
  journal = {Mathematics and Computers in Simulation},
  volume  = {223},
  pages   = {433--457},
  year    = {2024}
}
```

---

## Autores

**Dr. Erick Ordaz-Rivas**
Facultad de Ingeniería Mecánica y Eléctrica (FIME)
Universidad Autónoma de Nuevo León (UANL)
`erick.ordazrv@uanl.edu.mx`

---

## Licencia

Este proyecto es parte de una investigación académica activa en FIME-UANL.
Contactar al autor para uso externo.
