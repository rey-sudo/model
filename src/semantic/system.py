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
    """
    name: str
    coords: Coord3D
    definition: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[ConceptSnapshot] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ── Gestión de definición ──────────────────

    def _snapshot(self, note: str = "") -> None:
        self.history.append(ConceptSnapshot(
            timestamp=time.time(),
            definition=list(self.definition),
            note=note
        ))

    def set_definition(self, concepts: list[str], note: str = "") -> None:
        self._snapshot(note or "set")
        self.definition = list(concepts)

    def expand(self, concepts: list[str], note: str = "") -> None:
        """Añade conceptos a la definición (sin duplicados)."""
        self._snapshot(note or "expand")
        for c in concepts:
            if c not in self.definition:
                self.definition.append(c)

    def reduce(self, concepts: list[str], note: str = "") -> None:
        """Elimina conceptos de la definición."""
        self._snapshot(note or "reduce")
        self.definition = [c for c in self.definition if c not in concepts]

    def replace(self, old: str, new: str, note: str = "") -> None:
        """Sustituye un concepto por otro en la definición."""
        self._snapshot(note or f"replace {old} → {new}")
        self.definition = [new if c == old else c for c in self.definition]

    # ── Propiedades espaciales ──────────────────

    def centroid_definition(self, matrix: "SemanticMatrix3D") -> Coord3D | None:
        """Centroide (punto medio) de todos los conceptos que forman la definición."""
        coords = [matrix.get(c).coords for c in self.definition if matrix.get(c)]
        if not coords:
            return None
        n = len(coords)
        return tuple(sum(c[i] for c in coords) / n for i in range(3))

    def semantic_vector(self, matrix: "SemanticMatrix3D") -> Coord3D:
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
        note: str = ""
    ) -> Concept:
        if name in self._concepts:
            raise ValueError(f"Concepto '{name}' ya existe. Usa update().")
        c = Concept(
            name=name,
            coords=coords,
            definition=list(definition or []),
            metadata=metadata or {},
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
        lines = [
            f"┌─ Concepto: '{name}'",
            f"│  Coordenadas : ({x:.3f}, {y:.3f}, {z:.3f})",
            f"│  Definición  : {c.definition if c.definition else '(primitivo / sin definición)'}",
            f"│  Metadata    : {c.metadata if c.metadata else '{}'}",
            f"│  Versiones   : {len(c.history)} cambios históricos",
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

def demo():
    print("=" * 60)
    print("  SISTEMA SEMÁNTICO 3D — DEMOSTRACIÓN")
    print("=" * 60)

    # ── Construir un universo semántico mínimo ──
    m = SemanticMatrix3D("Filosofía-Ciencia")

    # Conceptos primitivos (sin definición → átomos del sistema)
    m.add("ser",       (0.0, 0.0, 0.0))
    m.add("no-ser",    (0.0, 0.0, 1.0))
    m.add("tiempo",    (1.0, 0.0, 0.0))
    m.add("espacio",   (0.0, 1.0, 0.0))
    m.add("materia",   (1.0, 1.0, 0.0))
    m.add("energía",   (1.0, 1.0, 1.0))
    m.add("causa",     (0.5, 0.0, 0.5))
    m.add("efecto",    (0.5, 0.1, 0.6))
    m.add("mente",     (0.0, 0.5, 0.5))
    m.add("conciencia",(0.1, 0.6, 0.6))

    # Conceptos compuestos (definidos por primitivos)
    m.add("cambio",    (1.0, 0.0, 1.0), ["ser", "no-ser", "tiempo"])
    m.add("movimiento",(1.1, 0.1, 1.0), ["cambio", "espacio", "tiempo"])
    m.add("física",    (1.5, 1.5, 0.5), ["materia", "energía", "espacio", "tiempo"])
    m.add("causalidad",(0.5, 0.5, 1.0), ["causa", "efecto", "tiempo"])
    m.add("existencia",(0.2, 0.2, 0.2), ["ser", "espacio", "tiempo"])
    m.add("mente-cuerpo", (0.5, 1.5, 0.5), ["mente", "materia", "causa"])
    m.add("libre-albedrío", (0.3, 1.0, 0.8), ["conciencia", "causalidad", "mente"])

    print("\n" + m.summary())

    # ── Describe un concepto ────────────────────
    print("\n" + m.describe("libre-albedrío"))

    # ── Camino semántico ────────────────────────
    print("\n── Camino conceptual: 'libre-albedrío' → 'ser' ──")
    path = m.path("libre-albedrío", "ser")
    print(" → ".join(path) if path else "Sin camino directo")

    # ── Razonamiento analógico ──────────────────
    print("\n── Analogía: movimiento : espacio :: causalidad : ? ──")
    results = m.analogy("movimiento", "espacio", "causalidad", n=3)
    for r, d in results:
        print(f"   {r:20s}  (distancia vectorial: {d:.3f})")

    # ── Primitivos de un concepto ───────────────
    print("\n── Cadena de implicación (primitivos de 'libre-albedrío') ──")
    prims = m.implication_chain("libre-albedrío")
    print(f"  {prims}")

    # ── Ancestros comunes ───────────────────────
    print("\n── Ancestros comunes: 'movimiento' y 'libre-albedrío' ──")
    anc = m.common_ancestors("movimiento", "libre-albedrío")
    print(f"  {anc}")

    # ── Diferencia conceptual ───────────────────
    print("\n── Diferencia: 'libre-albedrío' vs 'mente-cuerpo' ──")
    solo_a, solo_b = m.difference("libre-albedrío", "mente-cuerpo")
    print(f"  Solo en libre-albedrío : {solo_a}")
    print(f"  Solo en mente-cuerpo   : {solo_b}")

    # ── Orden topológico ────────────────────────
    print("\n── Orden topológico (precedencia) ──")
    topo = m.topological_order()
    if topo:
        print("  " + " → ".join(topo))
    else:
        print("  Ciclos detectados — no hay orden lineal")

    # ── Clusters ────────────────────────────────
    print("\n── Clusters semánticos (k=3) ──")
    clusters = m.clusters_naive(k=3)
    by_cluster: dict[int, list] = defaultdict(list)
    for name, cid in clusters.items():
        by_cluster[cid].append(name)
    for cid, members in sorted(by_cluster.items()):
        print(f"  Cluster {cid}: {members}")

    # ── Evolución temporal de un concepto ───────
    print("\n── Evolución temporal de 'causalidad' ──")
    caus = m.require("causalidad")
    caus.expand(["espacio"], note="revisión post-Kant")
    time.sleep(0.01)
    caus.reduce(["espacio"], note="vuelta al modelo clásico")
    time.sleep(0.01)
    caus.replace("causa", "acción", note="enfoque neocausal")
    time.sleep(0.01)
    caus.replace("acción", "causa", note="revert")

    drift = m.definition_drift("causalidad")
    for change in drift:
        print(f"  [{change['note']}] +{change['added']} -{change['removed']}")

    # ── Mapa ASCII 2D ───────────────────────────
    print("\n── Mapa ASCII (proyección plano XY) ──")
    print(m.ascii_map(axis="z", width=50, height=15))

    # ── Campo semántico ─────────────────────────
    print("\n── Campo semántico de 'física' (profundidad 2) ──")
    field = m.semantic_field("física", depth=2)
    for concept, depth in sorted(field.items(), key=lambda x: x[1]):
        indent = "  " * depth
        print(f"  {indent}{'└─' if depth else '●'} {concept}")

    print("\n" + "=" * 60)
    print("  FIN DE LA DEMOSTRACIÓN")
    print("=" * 60)


if __name__ == "__main__":
    demo()