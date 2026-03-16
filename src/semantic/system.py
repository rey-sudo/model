"""
SemanticMatrix3D
================
Sistema semántico donde cada concepto es una coordenada en un espacio 3D.
Las definiciones son conjuntos de otros conceptos (también coordenadas).
El sistema soporta razonamiento semántico, evolución temporal y fronteras conceptuales.

Estructura central:
  - Concepto   → punto (x, y, z) en el espacio
  - Definición → lista de nombres de otros conceptos (también puntos)
  - Frontera   → región entre un concepto y sus vecinos
  - Historia   → versiones anteriores de la definición con timestamps
"""

from __future__ import annotations

import math
import time
import copy
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────
# TIPOS BASE
# ─────────────────────────────────────────────

Coord3D = tuple[float, float, float]


def euclidean(a: Coord3D, b: Coord3D) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def midpoint(a: Coord3D, b: Coord3D) -> Coord3D:
    return tuple((ai + bi) / 2 for ai, bi in zip(a, b))


def vector(a: Coord3D, b: Coord3D) -> Coord3D:
    """Vector de a hacia b."""
    return tuple(bi - ai for ai, bi in zip(a, b))


def add_vectors(a: Coord3D, b: Coord3D) -> Coord3D:
    return tuple(ai + bi for ai, bi in zip(a, b))


def scale_vector(v: Coord3D, s: float) -> Coord3D:
    return tuple(vi * s for vi in v)


def normalize(v: Coord3D) -> Coord3D:
    mag = math.sqrt(sum(vi ** 2 for vi in v))
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return tuple(vi / mag for vi in v)


def dot(a: Coord3D, b: Coord3D) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def cosine_similarity(a: Coord3D, b: Coord3D) -> float:
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot(a, b) / (mag_a * mag_b)


# ─────────────────────────────────────────────
# CONCEPTO
# ─────────────────────────────────────────────

class DriftMode:
    """
    Estrategia de coordenadas cuando la definición cambia.

    FIXED      — las coordenadas nunca se tocan (ancla nominal).
                 El concepto tiene un "lugar histórico" inmutable.
                 Útil para modelar que la palabra/etiqueta es la identidad.

    AUTO       — tras cada cambio de definición las coordenadas se
                 recalculan como el centroide de los conceptos que la
                 conforman.  El concepto *es* su significado actual.

    TRAJECTORY — igual que AUTO pero guarda cada posición anterior
                 en coord_history.  Permite reconstruir el recorrido
                 completo del concepto a través del espacio semántico.
    """
    FIXED      = "fixed"
    AUTO       = "auto"
    TRAJECTORY = "trajectory"


@dataclass
class CoordSnapshot:
    """Posición histórica de un concepto en un momento dado."""
    timestamp: float
    coords: Coord3D
    note: str = ""


@dataclass
class ConceptSnapshot:
    """Versión histórica de la definición de un concepto."""
    timestamp: float
    definition: list[str]
    note: str = ""


@dataclass
class Concept:
    """
    Un concepto es:
      - Un nombre único
      - Una posición (x, y, z) en el espacio semántico
      - Una definición: lista de nombres de otros conceptos
      - Una historia de cambios en su definición
      - Un modo de deriva: FIXED | AUTO | TRAJECTORY
    """
    name: str
    coords: Coord3D
    definition: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[ConceptSnapshot] = field(default_factory=list)
    coord_history: list[CoordSnapshot] = field(default_factory=list)
    drift_mode: str = DriftMode.FIXED
    created_at: float = field(default_factory=time.time)

    # ── Gestión de definición ──────────────────

    def _snapshot(self, note: str = "") -> None:
        self.history.append(ConceptSnapshot(
            timestamp=time.time(),
            definition=list(self.definition),
            note=note
        ))

    def _record_coord(self, note: str = "") -> None:
        """Guarda la posición actual en coord_history."""
        self.coord_history.append(CoordSnapshot(
            timestamp=time.time(),
            coords=self.coords,
            note=note
        ))

    def _maybe_drift(self, matrix: "SemanticMatrix3D", note: str = "") -> None:
        """
        Si el modo es AUTO o TRAJECTORY, recalcula las coordenadas
        como el centroide de los conceptos de la definición actual.
        En TRAJECTORY también archiva la posición anterior.
        """
        if self.drift_mode == DriftMode.FIXED:
            return
        new_coords = self.centroid_definition(matrix)
        if new_coords is None:
            return                          # sin referencias resolubles, no mover
        if self.drift_mode == DriftMode.TRAJECTORY:
            self._record_coord(note)        # archiva ANTES de mover
        self.coords = new_coords

    def set_definition(self, concepts: list[str], note: str = "",
                       matrix: "SemanticMatrix3D | None" = None) -> None:
        self._snapshot(note or "set")
        self.definition = list(concepts)
        if matrix:
            self._maybe_drift(matrix, note or "set")

    def expand(self, concepts: list[str], note: str = "",
               matrix: "SemanticMatrix3D | None" = None) -> None:
        """Añade conceptos a la definición (sin duplicados)."""
        self._snapshot(note or "expand")
        for c in concepts:
            if c not in self.definition:
                self.definition.append(c)
        if matrix:
            self._maybe_drift(matrix, note or "expand")

    def reduce(self, concepts: list[str], note: str = "",
               matrix: "SemanticMatrix3D | None" = None) -> None:
        """Elimina conceptos de la definición."""
        self._snapshot(note or "reduce")
        self.definition = [c for c in self.definition if c not in concepts]
        if matrix:
            self._maybe_drift(matrix, note or "reduce")

    def replace(self, old: str, new: str, note: str = "",
                matrix: "SemanticMatrix3D | None" = None) -> None:
        """Sustituye un concepto por otro en la definición."""
        self._snapshot(note or f"replace {old} → {new}")
        self.definition = [new if c == old else c for c in self.definition]
        if matrix:
            self._maybe_drift(matrix, note or f"replace {old}→{new}")

    def reposition(self, new_coords: Coord3D, note: str = "") -> None:
        """
        Reposicionamiento manual. Siempre archiva la posición anterior
        (independientemente del drift_mode).
        """
        self._record_coord(note or "manual reposition")
        self.coords = new_coords

    # ── Propiedades espaciales ──────────────────

    def centroid_definition(self, matrix: "SemanticMatrix3D") -> Coord3D | None:
        """Centroide (punto medio) de todos los conceptos que forman la definición."""
        coords = [matrix.get(c).coords for c in self.definition if matrix.get(c)]
        if not coords:
            return None
        n = len(coords)
        return tuple(sum(c[i] for c in coords) / n for i in range(3))

    def trajectory_length(self) -> float:
        """Distancia total recorrida a lo largo de toda la trayectoria espacial."""
        positions = [snap.coords for snap in self.coord_history] + [self.coords]
        if len(positions) < 2:
            return 0.0
        return sum(euclidean(positions[i], positions[i+1])
                   for i in range(len(positions) - 1))

    def displacement(self) -> float:
        """Distancia entre la posición original y la actual."""
        if not self.coord_history:
            return 0.0
        return euclidean(self.coord_history[0].coords, self.coords)

    def original_coords(self) -> Coord3D:
        """Coordenadas originales (antes de cualquier deriva)."""
        if self.coord_history:
            return self.coord_history[0].coords
        return self.coords

    def velocity(self) -> float:
        """Velocidad media de cambio espacial (distancia / número de pasos)."""
        steps = len(self.coord_history)
        if steps == 0:
            return 0.0
        return self.trajectory_length() / steps

    def direction_of_drift(self) -> Coord3D:
        """Vector normalizado desde la posición original hasta la actual."""
        orig = self.original_coords()
        return normalize(vector(orig, self.coords))
        """Vector desde el concepto hacia el centroide de su definición."""
        centroid = self.centroid_definition(matrix)
        if centroid is None:
            return (0.0, 0.0, 0.0)
        return vector(self.coords, centroid)

    def __repr__(self) -> str:
        x, y, z = self.coords
        return f"Concept('{self.name}' @ ({x:.2f},{y:.2f},{z:.2f}) def=[{', '.join(self.definition)}])"


# ─────────────────────────────────────────────
# MATRIZ SEMÁNTICA 3D
# ─────────────────────────────────────────────

class SemanticMatrix3D:
    """
    Espacio semántico tridimensional.

    Cada concepto ocupa una coordenada. Su definición es un conjunto de
    otros conceptos (también coordenadas). Las relaciones entre conceptos
    forman una red navegable en el espacio.
    """

    def __init__(self, name: str = "SemanticMatrix"):
        self.name = name
        self._concepts: dict[str, Concept] = {}

    # ── CRUD ────────────────────────────────────

    def add(
        self,
        name: str,
        coords: Coord3D,
        definition: list[str] | None = None,
        metadata: dict | None = None,
        note: str = "",
        drift_mode: str = DriftMode.FIXED,
    ) -> Concept:
        if name in self._concepts:
            raise ValueError(f"Concepto '{name}' ya existe. Usa update().")
        c = Concept(
            name=name,
            coords=coords,
            definition=list(definition or []),
            metadata=metadata or {},
            drift_mode=drift_mode,
        )
        if note:
            c._snapshot(note)
        self._concepts[name] = c
        return c

    def get(self, name: str) -> Concept | None:
        return self._concepts.get(name)

    def require(self, name: str) -> Concept:
        c = self.get(name)
        if c is None:
            raise KeyError(f"Concepto '{name}' no encontrado.")
        return c

    def remove(self, name: str) -> None:
        self._concepts.pop(name, None)
        # Limpiar referencias en definiciones
        for c in self._concepts.values():
            if name in c.definition:
                c.reduce([name], note=f"auto: '{name}' eliminado del sistema")

    def all_names(self) -> list[str]:
        return list(self._concepts.keys())

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, name: str) -> bool:
        return name in self._concepts

    # ── Distancia y Vecindad ────────────────────

    def distance(self, a: str, b: str) -> float:
        """Distancia euclidiana entre dos conceptos."""
        return euclidean(self.require(a).coords, self.require(b).coords)

    def nearest(self, name: str, n: int = 5, exclude_self: bool = True) -> list[tuple[str, float]]:
        """Los N conceptos más cercanos en el espacio 3D."""
        source = self.require(name).coords
        distances = [
            (cname, euclidean(source, c.coords))
            for cname, c in self._concepts.items()
            if not (exclude_self and cname == name)
        ]
        return sorted(distances, key=lambda x: x[1])[:n]

    def within_radius(self, name: str, radius: float) -> list[tuple[str, float]]:
        """Todos los conceptos dentro de un radio dado."""
        source = self.require(name).coords
        result = []
        for cname, c in self._concepts.items():
            if cname == name:
                continue
            d = euclidean(source, c.coords)
            if d <= radius:
                result.append((cname, d))
        return sorted(result, key=lambda x: x[1])

    def frontier(self, name: str, radius: float | None = None) -> dict[str, float]:
        """
        Frontera conceptual: conceptos que delimitan el espacio de este concepto.
        Si no se da radio, se calcula como la distancia promedio a los vecinos
        más cercanos de la definición.
        """
        concept = self.require(name)
        if radius is None:
            if not concept.definition:
                # Usar distancia mínima global como radio
                neighbors = self.nearest(name, n=3)
                radius = neighbors[0][1] * 1.5 if neighbors else 1.0
            else:
                dists = [self.distance(name, d) for d in concept.definition if d in self]
                radius = (sum(dists) / len(dists)) * 1.2 if dists else 1.0
        nearby = self.within_radius(name, radius)
        return {n: d for n, d in nearby}

    # ── Orientación y Similitud ─────────────────

    def cosine_sim(self, a: str, b: str) -> float:
        """Similitud coseno entre los vectores-coordenada de dos conceptos."""
        return cosine_similarity(self.require(a).coords, self.require(b).coords)

    def most_similar(self, name: str, n: int = 5) -> list[tuple[str, float]]:
        """Conceptos más similares por coseno (orientación similar desde el origen)."""
        source = self.require(name).coords
        sims = [
            (cname, cosine_similarity(source, c.coords))
            for cname, c in self._concepts.items()
            if cname != name
        ]
        return sorted(sims, key=lambda x: -x[1])[:n]

    # ── Grafo Semántico ─────────────────────────

    def definition_graph(self) -> dict[str, list[str]]:
        """Grafo de definiciones: concepto → lista de conceptos en su definición."""
        return {name: list(c.definition) for name, c in self._concepts.items()}

    def semantic_field(self, name: str, depth: int = 3) -> dict[str, int]:
        """
        Campo semántico: todos los conceptos alcanzables desde 'name'
        siguiendo la cadena de definiciones. Retorna {concepto: profundidad}.
        """
        visited: dict[str, int] = {}
        queue = deque([(name, 0)])
        while queue:
            current, d = queue.popleft()
            if current in visited or d > depth:
                continue
            visited[current] = d
            concept = self.get(current)
            if concept:
                for child in concept.definition:
                    if child not in visited:
                        queue.append((child, d + 1))
        return visited

    def reverse_index(self) -> dict[str, list[str]]:
        """¿Qué conceptos USAN a X en su definición? (índice inverso)"""
        index: dict[str, list[str]] = defaultdict(list)
        for name, c in self._concepts.items():
            for ref in c.definition:
                index[ref].append(name)
        return dict(index)

    def used_by(self, name: str) -> list[str]:
        """Conceptos que usan a 'name' en su definición."""
        return self.reverse_index().get(name, [])

    # ── Razonamiento ────────────────────────────

    def path(self, start: str, end: str, max_depth: int = 10) -> list[str] | None:
        """
        Camino conceptual de 'start' a 'end' siguiendo definiciones.
        BFS sobre el grafo semántico.
        """
        if start not in self or end not in self:
            return None
        queue = deque([[start]])
        visited = {start}
        while queue:
            current_path = queue.popleft()
            current = current_path[-1]
            if current == end:
                return current_path
            if len(current_path) > max_depth:
                continue
            concept = self.get(current)
            if concept:
                for neighbor in concept.definition:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(current_path + [neighbor])
        return None

    def analogy(self, a: str, b: str, c: str, n: int = 3) -> list[tuple[str, float]]:
        """
        Razonamiento analógico en el espacio vectorial:
        a : b :: c : ?
        El resultado es c + (b - a) → buscar el concepto más cercano.
        """
        ca, cb, cc = self.require(a).coords, self.require(b).coords, self.require(c).coords
        # Vector diferencia a→b aplicado desde c
        diff = vector(ca, cb)
        target = add_vectors(cc, diff)
        distances = [
            (name, euclidean(target, concept.coords))
            for name, concept in self._concepts.items()
            if name not in {a, b, c}
        ]
        return sorted(distances, key=lambda x: x[1])[:n]

    def intersection(self, *names: str) -> list[str]:
        """Conceptos que aparecen en la definición de TODOS los nombres dados."""
        if not names:
            return []
        sets = [set(self.require(n).definition) for n in names if n in self]
        if not sets:
            return []
        return list(set.intersection(*sets))

    def difference(self, a: str, b: str) -> tuple[list[str], list[str]]:
        """
        Diferencia conceptual: qué tiene 'a' que no tiene 'b' y viceversa.
        Retorna (solo_en_a, solo_en_b).
        """
        da = set(self.require(a).definition)
        db = set(self.require(b).definition)
        return list(da - db), list(db - da)

    def implication_chain(self, name: str) -> list[str]:
        """
        Cadena de implicación: expande la definición de 'name' recursivamente
        hasta llegar a conceptos sin definición (conceptos primitivos / atómicos).
        Retorna la lista ordenada de conceptos primitivos.
        """
        primitives = []
        visited = set()

        def _expand(n: str):
            if n in visited:
                return
            visited.add(n)
            c = self.get(n)
            if not c or not c.definition:
                primitives.append(n)
                return
            for child in c.definition:
                _expand(child)

        _expand(name)
        primitives = [p for p in primitives if p != name]
        return primitives

    def common_ancestors(self, a: str, b: str, depth: int = 6) -> list[str]:
        """Conceptos que son base (antecesores) comunes de 'a' y 'b'."""
        field_a = set(self.semantic_field(a, depth).keys())
        field_b = set(self.semantic_field(b, depth).keys())
        return list((field_a & field_b) - {a, b})

    def contradictions(self) -> list[tuple[str, str, str]]:
        """
        Detecta posibles contradicciones: dos conceptos A y B que están en la
        definición de C pero se definen mutuamente de forma excluyente
        (sus campos semánticos no se solapan en absoluto).
        Retorna lista de (C, A, B).
        """
        result = []
        for name, c in self._concepts.items():
            defn = c.definition
            for i, a in enumerate(defn):
                for b in defn[i + 1:]:
                    if a not in self or b not in self:
                        continue
                    fa = set(self.semantic_field(a, 2).keys())
                    fb = set(self.semantic_field(b, 2).keys())
                    # Si no hay nada en común y son vecinos del mismo concepto,
                    # puede indicar tensión semántica
                    if not fa.intersection(fb) and len(fa) > 1 and len(fb) > 1:
                        result.append((name, a, b))
        return result

    # ── Evolución Temporal ──────────────────────

    def coord_drift(self, name: str) -> list[dict]:
        """
        Historial de posiciones de un concepto en el espacio 3D.
        Muestra la trayectoria completa con timestamps.
        """
        c = self.require(name)
        all_positions = list(c.coord_history) + [
            CoordSnapshot(timestamp=time.time(), coords=c.coords, note="(posición actual)")
        ]
        result = []
        for i, snap in enumerate(all_positions):
            entry = {
                "step": i,
                "timestamp": snap.timestamp,
                "coords": snap.coords,
                "note": snap.note,
            }
            if i > 0:
                prev = all_positions[i - 1].coords
                entry["delta"] = round(euclidean(prev, snap.coords), 4)
            result.append(entry)
        return result

    def most_displaced(self, n: int = 5) -> list[tuple[str, float]]:
        """Conceptos que más se han desplazado desde su posición original."""
        return sorted(
            [(name, c.displacement()) for name, c in self._concepts.items()],
            key=lambda x: -x[1]
        )[:n]

    def snapshot_matrix(self, timestamp: float) -> dict[str, Coord3D]:
        """
        Reconstruye las coordenadas de todos los conceptos tal como eran
        en un momento dado (interpolando si hace falta).
        Útil para "rebobinar" el espacio semántico.
        """
        result = {}
        for name, c in self._concepts.items():
            # Buscar la última posición archivada antes del timestamp
            past = [s for s in c.coord_history if s.timestamp <= timestamp]
            if past:
                result[name] = past[-1].coords
            else:
                # O la posición actual si nunca se movió
                result[name] = c.coords
        return result
        """
        Evolución histórica de la definición de un concepto.
        Retorna lista de cambios con timestamp, añadidos y eliminados.
        """
        c = self.require(name)
        if not c.history:
            return []
        changes = []
        prev = []
        for snap in c.history:
            added = [x for x in snap.definition if x not in prev]
            removed = [x for x in prev if x not in snap.definition]
            changes.append({
                "timestamp": snap.timestamp,
                "note": snap.note,
                "added": added,
                "removed": removed,
                "definition": list(snap.definition),
            })
            prev = snap.definition
        return changes

    def age(self, name: str) -> float:
        """Tiempo desde que fue creado el concepto (segundos)."""
        return time.time() - self.require(name).created_at

    def most_evolved(self, n: int = 5) -> list[tuple[str, int]]:
        """Conceptos que más han cambiado su definición a lo largo del tiempo."""
        return sorted(
            [(name, len(c.history)) for name, c in self._concepts.items()],
            key=lambda x: -x[1]
        )[:n]

    # ── Proyección y Análisis Espacial ──────────

    def projection_2d(self, axis: str = "z") -> dict[str, tuple[float, float]]:
        """
        Proyecta el espacio 3D en un plano 2D eliminando un eje.
        axis: 'x', 'y' o 'z' (eje que se elimina)
        """
        axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
        i, j = axes.get(axis, (0, 1))
        return {
            name: (c.coords[i], c.coords[j])
            for name, c in self._concepts.items()
        }

    def bounding_box(self) -> tuple[Coord3D, Coord3D]:
        """Caja límite del espacio semántico (min y max por cada eje)."""
        xs = [c.coords[0] for c in self._concepts.values()]
        ys = [c.coords[1] for c in self._concepts.values()]
        zs = [c.coords[2] for c in self._concepts.values()]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def centroid(self) -> Coord3D:
        """Centroide de todo el espacio semántico."""
        n = len(self._concepts)
        if n == 0:
            return (0.0, 0.0, 0.0)
        coords = list(self._concepts.values())
        return tuple(sum(c.coords[i] for c in coords) / n for i in range(3))

    def density(self, center: Coord3D, radius: float) -> int:
        """Número de conceptos dentro de una esfera dada."""
        return sum(
            1 for c in self._concepts.values()
            if euclidean(c.coords, center) <= radius
        )

    def clusters_naive(self, k: int = 3) -> dict[str, int]:
        """
        Clustering simple por k-means sobre coordenadas 3D.
        Retorna {nombre_concepto: cluster_id}.
        """
        import random
        names = list(self._concepts.keys())
        coords = [self._concepts[n].coords for n in names]
        if len(names) <= k:
            return {n: i for i, n in enumerate(names)}

        # Inicializar centroides al azar
        centroids = random.sample(coords, k)

        for _ in range(50):
            # Asignar clusters
            assignments = []
            for c in coords:
                dists = [euclidean(c, cent) for cent in centroids]
                assignments.append(dists.index(min(dists)))

            # Recalcular centroides
            new_centroids = []
            for ki in range(k):
                cluster_coords = [coords[i] for i, a in enumerate(assignments) if a == ki]
                if not cluster_coords:
                    new_centroids.append(centroids[ki])
                else:
                    n = len(cluster_coords)
                    new_centroids.append(tuple(
                        sum(c[j] for c in cluster_coords) / n for j in range(3)
                    ))

            if new_centroids == centroids:
                break
            centroids = new_centroids

        return {names[i]: assignments[i] for i in range(len(names))}

    # ── Orden y Precedencia ─────────────────────

    def precedes(self, a: str, b: str) -> bool:
        """
        ¿'a' precede a 'b' en el espacio?
        Por defecto: 'a' precede a 'b' si está en la definición de 'b',
        O si está en el campo semántico de 'b' pero no viceversa.
        """
        b_field = set(self.semantic_field(b).keys())
        return a in b_field

    def topological_order(self) -> list[str] | None:
        """
        Orden topológico del grafo de definiciones.
        Los conceptos sin dependencias van primero (primitivos).
        Retorna None si hay ciclos.
        """
        in_degree: dict[str, int] = defaultdict(int)
        graph = self.definition_graph()

        for name in self._concepts:
            in_degree.setdefault(name, 0)

        for name, deps in graph.items():
            for dep in deps:
                if dep in self._concepts:
                    in_degree[name] += 1  # name depende de dep

        # Conceptos sin dependencias
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            # Quien depende de 'node'? → los que lo tienen en su definición
            for name, deps in graph.items():
                if node in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(order) != len(self._concepts):
            return None  # Ciclo detectado
        return order

    # ── Visualización Textual ───────────────────

    def describe(self, name: str) -> str:
        """Descripción completa de un concepto."""
        c = self.require(name)
        x, y, z = c.coords
        orig = c.original_coords()
        ox, oy, oz = orig
        lines = [
            f"┌─ Concepto: '{name}'",
            f"│  Coordenadas : ({x:.3f}, {y:.3f}, {z:.3f})",
            f"│  Origen      : ({ox:.3f}, {oy:.3f}, {oz:.3f})  "
            f"desplazamiento={c.displacement():.3f}  modo={c.drift_mode}",
            f"│  Definición  : {c.definition if c.definition else '(primitivo / sin definición)'}",
            f"│  Metadata    : {c.metadata if c.metadata else '{}'}",
            f"│  Versiones   : {len(c.history)} cambios de definición  |  "
            f"{len(c.coord_history)} pasos espaciales",
        ]
        nearest = self.nearest(name, n=3)
        lines.append(f"│  Vecinos     : {[(n, round(d, 2)) for n, d in nearest]}")
        frontier = self.frontier(name)
        lines.append(f"│  Frontera    : {list(frontier.keys())}")
        used = self.used_by(name)
        lines.append(f"│  Usado en    : {used}")
        primitivos = self.implication_chain(name)
        lines.append(f"└  Primitivos  : {primitivos}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Resumen del estado de toda la matriz."""
        bbox_min, bbox_max = self.bounding_box() if self._concepts else ((0,0,0),(0,0,0))
        vol = math.prod(max(bbox_max[i] - bbox_min[i], 0.001) for i in range(3))
        cx, cy, cz = self.centroid()
        lines = [
            f"═══ SemanticMatrix3D: '{self.name}' ═══",
            f"  Conceptos  : {len(self._concepts)}",
            f"  Centroide  : ({cx:.2f}, {cy:.2f}, {cz:.2f})",
            f"  BoundingBox: {tuple(round(v,2) for v in bbox_min)} → {tuple(round(v,2) for v in bbox_max)}",
            f"  Volumen est.: {vol:.2f}",
        ]
        topo = self.topological_order()
        lines.append(f"  Orden topo. : {'CON CICLOS' if topo is None else ' → '.join(topo[:8]) + ('...' if len(topo)>8 else '')}")
        return "\n".join(lines)

    def ascii_map(self, axis: str = "z", width: int = 50, height: int = 20) -> str:
        """Mapa ASCII 2D proyectado del espacio (proyectando el eje 'axis')."""
        proj = self.projection_2d(axis)
        if not proj:
            return "(vacío)"
        xs = [v[0] for v in proj.values()]
        ys = [v[1] for v in proj.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xrange = max(xmax - xmin, 0.001)
        yrange = max(ymax - ymin, 0.001)

        grid = [[" "] * width for _ in range(height)]

        for name, (px, py) in proj.items():
            col = int((px - xmin) / xrange * (width - 1))
            row = int((py - ymin) / yrange * (height - 1))
            row = height - 1 - row  # invertir eje Y
            grid[row][col] = name[0].upper()

        lines = ["┌" + "─" * width + "┐"]
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append("└" + "─" * width + "┘")
        lines.append(f"  Proyección sobre plano perpendicular al eje '{axis}'")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# DEMOSTRACIÓN COMPLETA
# ─────────────────────────────────────────────

    def definition_drift(self, name: str) -> list[dict]:
        """
        Evolución histórica de la definición de un concepto.
        Retorna lista de cambios con timestamp, añadidos y eliminados.
        """
        c = self.require(name)
        if not c.history:
            return []
        changes = []
        prev = []
        for snap in c.history:
            added = [x for x in snap.definition if x not in prev]
            removed = [x for x in prev if x not in snap.definition]
            changes.append({
                "timestamp": snap.timestamp,
                "note": snap.note,
                "added": added,
                "removed": removed,
                "definition": list(snap.definition),
            })
            prev = snap.definition
        return changes


def demo():
    print("=" * 65)
    print("  SISTEMA SEMÁNTICO 3D — TRES MODOS DE DERIVA")
    print("=" * 65)

    m = SemanticMatrix3D("Demo-Deriva")

    # ── Primitivos (anclas fijas del espacio) ──────────────────
    m.add("materia",    (0.0, 0.0, 0.0))
    m.add("vacío",      (1.0, 0.0, 0.0))
    m.add("indivisible",(0.0, 1.0, 0.0))
    m.add("divisible",  (1.0, 1.0, 0.0))
    m.add("nuclear",    (0.5, 0.5, 1.0))
    m.add("cuántico",   (0.5, 0.5, 2.0))
    m.add("onda",       (0.0, 0.5, 1.5))
    m.add("partícula",  (1.0, 0.5, 1.5))

    print("\n  Primitivos añadidos (posición fija, sin definición).")

    # ── MODO 1: FIXED ──────────────────────────────────────────
    # El concepto "átomo" según Demócrito (siglo V a.C.)
    # Definición original: materia indivisible que existe en el vacío
    atomo_fixed = m.add(
        "átomo[FIXED]",
        (0.2, 0.8, 0.1),                        # coordenada original
        definition=["materia", "indivisible", "vacío"],
        drift_mode=DriftMode.FIXED,
        note="Demócrito ~400 a.C.",
    )

    print("\n── MODO FIXED ─────────────────────────────────────────────")
    print(f"  Posición inicial: {atomo_fixed.coords}")
    atomo_fixed.expand(["divisible", "nuclear"], note="Dalton 1808 — divisible pero atómico")
    atomo_fixed.expand(["cuántico", "onda", "partícula"], note="Bohr/Heisenberg 1927")
    print(f"  Posición final  : {atomo_fixed.coords}  (sin cambio)")
    print(f"  Desplazamiento  : {atomo_fixed.displacement():.4f}  ← siempre 0")
    print(f"  Definición actual: {atomo_fixed.definition}")
    print(f"  Pasos espaciales: {len(atomo_fixed.coord_history)}")

    # ── MODO 2: AUTO ───────────────────────────────────────────
    # Misma historia, pero las coords se recalculan al centroide
    atomo_auto = m.add(
        "átomo[AUTO]",
        (0.2, 0.8, 0.1),
        definition=["materia", "indivisible", "vacío"],
        drift_mode=DriftMode.AUTO,
        note="Demócrito ~400 a.C.",
    )
    # Forzar la posición inicial como centroide de su definición actual
    atomo_auto._maybe_drift(m, "inicio")

    print("\n── MODO AUTO ──────────────────────────────────────────────")
    print(f"  Posición post-inicio: {tuple(round(v,3) for v in atomo_auto.coords)}")

    atomo_auto.expand(["divisible", "nuclear"], note="Dalton 1808", matrix=m)
    print(f"  Posición post-Dalton: {tuple(round(v,3) for v in atomo_auto.coords)}")

    atomo_auto.expand(["cuántico", "onda", "partícula"], note="Bohr/Heisenberg 1927", matrix=m)
    print(f"  Posición post-cuántica: {tuple(round(v,3) for v in atomo_auto.coords)}")

    print(f"  Desplazamiento  : {atomo_auto.displacement():.4f}  ← no hay historia, sólo coord actual")
    print(f"  Pasos espaciales: {len(atomo_auto.coord_history)}  ← AUTO no archiva")

    # ── MODO 3: TRAJECTORY ─────────────────────────────────────
    # La más completa: se mueve Y guarda cada posición
    atomo_traj = m.add(
        "átomo[TRAJ]",
        (0.2, 0.8, 0.1),
        definition=["materia", "indivisible", "vacío"],
        drift_mode=DriftMode.TRAJECTORY,
        note="Demócrito ~400 a.C.",
    )
    atomo_traj._maybe_drift(m, "inicio")

    print("\n── MODO TRAJECTORY ────────────────────────────────────────")
    print(f"  Posición inicial : {tuple(round(v,3) for v in atomo_traj.coords)}")

    atomo_traj.expand(["divisible", "nuclear"], note="Dalton 1808", matrix=m)
    atomo_traj.expand(["cuántico"], note="Bohr 1913", matrix=m)
    atomo_traj.expand(["onda", "partícula"], note="De Broglie / Heisenberg 1927", matrix=m)
    atomo_traj.reduce(["indivisible"], note="confirmación experimental — el átomo es divisible", matrix=m)

    print(f"  Posición final   : {tuple(round(v,3) for v in atomo_traj.coords)}")
    print(f"  Desplazamiento   : {atomo_traj.displacement():.4f}")
    print(f"  Long. trayectoria: {atomo_traj.trajectory_length():.4f}")
    print(f"  Velocidad media  : {atomo_traj.velocity():.4f}")
    print(f"  Dirección deriva : {tuple(round(v,3) for v in atomo_traj.direction_of_drift())}")
    print(f"  Pasos espaciales : {len(atomo_traj.coord_history)}")

    print("\n  Trayectoria completa:")
    for step in m.coord_drift("átomo[TRAJ]"):
        arrow = f"  Δ={step['delta']:.4f}" if "delta" in step else ""
        print(f"    paso {step['step']}: {tuple(round(v,3) for v in step['coords'])}  "
              f"[{step['note']}]{arrow}")

    # ── Snapshot histórico del espacio ─────────────────────────
    print("\n── Snapshot del espacio en un momento pasado ─────────────")
    # Tomar el timestamp del segundo paso de la trayectoria
    if len(atomo_traj.coord_history) >= 2:
        t_pasado = atomo_traj.coord_history[1].timestamp
        snap = m.snapshot_matrix(t_pasado)
        print(f"  'átomo[TRAJ]' en t={t_pasado:.3f}:")
        coords_past = snap.get("átomo[TRAJ]")
        if coords_past:
            print(f"    {tuple(round(v,3) for v in coords_past)}")

    # ── Comparación final ──────────────────────────────────────
    print("\n── Comparación de los tres modos ─────────────────────────")
    print(f"  {'Modo':<14} {'Coords actuales':<28} {'Desplaz.':<12} {'Pasos'}")
    print(f"  {'─'*14} {'─'*28} {'─'*12} {'─'*5}")
    for name in ["átomo[FIXED]", "átomo[AUTO]", "átomo[TRAJ]"]:
        c = m.require(name)
        coord_str = str(tuple(round(v,3) for v in c.coords))
        print(f"  {name:<14} {coord_str:<28} {c.displacement():<12.4f} {len(c.coord_history)}")

    print("\n  Conclusión:")
    print("  - FIXED     → identidad nominal: el concepto tiene una 'dirección postal' inmutable.")
    print("  - AUTO      → identidad semántica: el concepto ES su significado actual.")
    print("  - TRAJECTORY→ identidad continua: el concepto tiene historia y recorrido en el espacio.")
    print("=" * 65)


if __name__ == "__main__":
    demo()