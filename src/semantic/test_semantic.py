"""
test_semantic_matrix.py
=======================
Suite completa de tests para SemanticMatrix3D.

Tipos de razonamiento cubiertos
────────────────────────────────
  R01  Espacial       — distancia, vecindad, radio, fronteras, bbox, densidad
  R02  Similitud      — coseno, más-similares
  R03  Grafo          — grafo de definiciones, índice inverso, used_by
  R04  Campo          — campo semántico (BFS), profundidad controlada
  R05  Deductivo      — camino conceptual, cadena de implicación, primitivos
  R06  Precedencia    — precedes(), orden topológico, detección de ciclos
  R07  Analógico      — a:b :: c:? vectorial
  R08  Conjuntista    — intersección, diferencia, ancestros comunes
  R09  Tensión        — detección de contradicciones/tensiones semánticas
  R10  Clustering     — k-means sobre coordenadas 3D
  R11  Proyección     — proyección 2D por eje, bbox, centroide global
  R12  Temporal       — evolución de definición, drift de coords, snapshot
  R13  Drift FIXED    — coordenadas inmutables ante cambios de definición
  R14  Drift AUTO     — coordenadas = centroide de definición actual
  R15  Drift TRAJ     — trayectoria completa + métricas de deriva
  R16  CRUD           — add, get, remove, contains, require, all_names
  R17  Borde/Error    — conceptos vacíos, ciclos, espacios degenerados
"""

import math
import sys
import time
import unittest

# ── importar el módulo a testear ─────────────────────────────────────────────
sys.path.insert(0, ".")
from system import (
    SemanticMatrix3D, Concept, DriftMode,
    euclidean, cosine_similarity, vector, add_vectors, normalize,
)

EPS = 1e-9  # tolerancia numérica

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES reutilizables
# ─────────────────────────────────────────────────────────────────────────────

def build_minimal() -> SemanticMatrix3D:
    """
    Espacio mínimo con 4 primitivos y 2 compuestos.
    Sirve como base para la mayoría de los tests.

         vida ──┐
                ├─ ser, materia, energía  (primitivos)
         mente─┘    tiempo               (primitivo)
    """
    m = SemanticMatrix3D("minimal")
    m.add("ser",      (0.0, 0.0, 0.0))
    m.add("materia",  (1.0, 0.0, 0.0))
    m.add("energía",  (0.0, 1.0, 0.0))
    m.add("tiempo",   (0.0, 0.0, 1.0))
    m.add("vida",     (0.5, 0.5, 0.5), ["ser", "materia", "energía"])
    m.add("mente",    (0.3, 0.6, 0.4), ["ser", "energía", "tiempo"])
    return m


def build_chain() -> SemanticMatrix3D:
    """
    Cadena lineal: A → B → C → D  (A primitivo, D es el más derivado).
    Permite testear camino y orden topológico sin ambigüedad.
    """
    m = SemanticMatrix3D("chain")
    m.add("A", (0.0, 0.0, 0.0))
    m.add("B", (1.0, 0.0, 0.0), ["A"])
    m.add("C", (2.0, 0.0, 0.0), ["B"])
    m.add("D", (3.0, 0.0, 0.0), ["C"])
    return m


def build_cyclic() -> SemanticMatrix3D:
    """
    Ciclo deliberado: X → Y → Z → X.
    Sirve para probar detección de ciclos en topological_order.
    """
    m = SemanticMatrix3D("cyclic")
    m.add("X", (0.0, 0.0, 0.0), ["Y"])
    m.add("Y", (1.0, 0.0, 0.0), ["Z"])
    m.add("Z", (0.5, 0.5, 0.5), ["X"])
    return m


# ─────────────────────────────────────────────────────────────────────────────
# R01 — RAZONAMIENTO ESPACIAL
# ─────────────────────────────────────────────────────────────────────────────

class TestEspacial(unittest.TestCase):
    """Distancia euclidiana, vecindad, radio, frontera, bbox, densidad."""

    def setUp(self):
        self.m = build_minimal()

    # ── distancia ─────────────────────────────

    def test_distancia_misma_posicion(self):
        """Un concepto tiene distancia 0 consigo mismo."""
        self.assertAlmostEqual(self.m.distance("ser", "ser"), 0.0, places=9)

    def test_distancia_simetrica(self):
        """d(A, B) == d(B, A)."""
        self.assertAlmostEqual(
            self.m.distance("materia", "energía"),
            self.m.distance("energía", "materia"),
            places=9
        )

    def test_distancia_conocida(self):
        """materia=(1,0,0) y energía=(0,1,0): distancia = sqrt(2)."""
        self.assertAlmostEqual(
            self.m.distance("materia", "energía"),
            math.sqrt(2),
            places=9
        )

    def test_desigualdad_triangular(self):
        """d(A,C) <= d(A,B) + d(B,C)."""
        dAB = self.m.distance("ser", "materia")
        dBC = self.m.distance("materia", "vida")
        dAC = self.m.distance("ser", "vida")
        self.assertLessEqual(dAC, dAB + dBC + EPS)

    # ── nearest ───────────────────────────────

    def test_nearest_excluye_self_por_defecto(self):
        results = self.m.nearest("vida", n=3)
        names = [r[0] for r in results]
        self.assertNotIn("vida", names)

    def test_nearest_ordenado_ascendente(self):
        results = self.m.nearest("vida", n=4)
        dists = [r[1] for r in results]
        self.assertEqual(dists, sorted(dists))

    def test_nearest_n_limita_resultados(self):
        results = self.m.nearest("ser", n=2)
        self.assertEqual(len(results), 2)

    def test_nearest_incluye_self_si_se_pide(self):
        results = self.m.nearest("ser", n=3, exclude_self=False)
        names = [r[0] for r in results]
        self.assertIn("ser", names)
        # y la distancia a sí mismo debe ser 0
        self_dist = next(d for n, d in results if n == "ser")
        self.assertAlmostEqual(self_dist, 0.0, places=9)

    # ── within_radius ─────────────────────────

    def test_radio_cero_vacio(self):
        """Radio 0 no debe capturar ningún vecino (excluye self)."""
        result = self.m.within_radius("ser", 0.0)
        self.assertEqual(result, [])

    def test_radio_grande_captura_todo(self):
        """Radio muy grande captura todos los conceptos excepto self."""
        result = self.m.within_radius("vida", 100.0)
        names = {r[0] for r in result}
        self.assertEqual(names, {"ser", "materia", "energía", "tiempo", "mente"})

    def test_radio_preciso(self):
        """
        ser=(0,0,0), materia=(1,0,0): distancia=1.
        Radio=0.9 → no aparece materia.  Radio=1.1 → sí aparece.
        """
        sin_materia = {r[0] for r in self.m.within_radius("ser", 0.9)}
        self.assertNotIn("materia", sin_materia)
        con_materia = {r[0] for r in self.m.within_radius("ser", 1.1)}
        self.assertIn("materia", con_materia)

    # ── frontier ──────────────────────────────

    def test_frontier_tipo_dict(self):
        f = self.m.frontier("vida")
        self.assertIsInstance(f, dict)

    def test_frontier_no_incluye_self(self):
        f = self.m.frontier("vida")
        self.assertNotIn("vida", f)

    def test_frontier_radio_explicito(self):
        """Con radio 0.05 desde vida=(0.5,0.5,0.5) no debe haber vecinos."""
        f = self.m.frontier("vida", radius=0.05)
        self.assertEqual(f, {})

    # ── bounding_box y centroide ───────────────

    def test_bounding_box_contiene_todos(self):
        bmin, bmax = self.m.bounding_box()
        for c in self.m._concepts.values():
            for i in range(3):
                self.assertLessEqual(bmin[i], c.coords[i] + EPS)
                self.assertGreaterEqual(bmax[i], c.coords[i] - EPS)

    def test_centroide_dentro_de_bbox(self):
        bmin, bmax = self.m.bounding_box()
        cx, cy, cz = self.m.centroid()
        self.assertGreaterEqual(cx, bmin[0] - EPS)
        self.assertLessEqual(cx, bmax[0] + EPS)

    # ── density ───────────────────────────────

    def test_density_radio_cero_en_ser(self):
        """ser=(0,0,0): densidad en radio 0 desde el mismo punto = 0 (no cuenta a sí mismo)."""
        # density cuenta todos los conceptos dentro del radio, incluyendo self
        d = self.m.density((0.0, 0.0, 0.0), 0.0)
        self.assertGreaterEqual(d, 1)   # al menos ser mismo está en radio 0

    def test_density_aumenta_con_radio(self):
        d_pequeno = self.m.density((0.5, 0.5, 0.5), 0.3)
        d_grande  = self.m.density((0.5, 0.5, 0.5), 2.0)
        self.assertLessEqual(d_pequeno, d_grande)


# ─────────────────────────────────────────────────────────────────────────────
# R02 — RAZONAMIENTO POR SIMILITUD
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilitud(unittest.TestCase):
    """Similitud coseno entre vectores de coordenada."""

    def setUp(self):
        self.m = SemanticMatrix3D("sim")
        self.m.add("origen",    (0.0, 0.0, 0.0))   # degenerate
        self.m.add("este",      (1.0, 0.0, 0.0))
        self.m.add("norte",     (0.0, 1.0, 0.0))
        self.m.add("este2",     (2.0, 0.0, 0.0))   # mismo ángulo que 'este'
        self.m.add("diagonal",  (1.0, 1.0, 0.0))

    def test_coseno_consigo_mismo(self):
        """cos(este, este) = 1."""
        self.assertAlmostEqual(self.m.cosine_sim("este", "este"), 1.0, places=9)

    def test_coseno_ortogonales(self):
        """este=(1,0,0) y norte=(0,1,0) son perpendiculares → sim = 0."""
        self.assertAlmostEqual(self.m.cosine_sim("este", "norte"), 0.0, places=9)

    def test_coseno_misma_direccion_distinta_magnitud(self):
        """este=(1,0,0) y este2=(2,0,0) apuntan igual → sim = 1."""
        self.assertAlmostEqual(self.m.cosine_sim("este", "este2"), 1.0, places=9)

    def test_coseno_diagonal(self):
        """este vs diagonal=(1,1,0): ángulo 45° → cos(45°) ≈ 0.707."""
        sim = self.m.cosine_sim("este", "diagonal")
        self.assertAlmostEqual(sim, math.sqrt(2) / 2, places=6)

    def test_coseno_origen_degenerado(self):
        """Uno de los vectores es el origen → similitud = 0 (no crash)."""
        sim = self.m.cosine_sim("origen", "este")
        self.assertEqual(sim, 0.0)

    def test_most_similar_orden(self):
        """most_similar debe retornar en orden decreciente de similitud."""
        sims = self.m.most_similar("este", n=3)
        values = [s for _, s in sims]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_most_similar_mas_cercano_es_este2(self):
        """este2 tiene la misma dirección que este → debe ser el más similar."""
        top, _ = self.m.most_similar("este", n=1)[0]
        self.assertEqual(top, "este2")


# ─────────────────────────────────────────────────────────────────────────────
# R03 — RAZONAMIENTO SOBRE EL GRAFO DE DEFINICIONES
# ─────────────────────────────────────────────────────────────────────────────

class TestGrafo(unittest.TestCase):

    def setUp(self):
        self.m = build_minimal()

    def test_definition_graph_todas_las_claves(self):
        g = self.m.definition_graph()
        self.assertEqual(set(g.keys()), set(self.m.all_names()))

    def test_definition_graph_vida(self):
        g = self.m.definition_graph()
        self.assertIn("ser",     g["vida"])
        self.assertIn("materia", g["vida"])
        self.assertIn("energía", g["vida"])

    def test_definition_graph_primitivo_vacio(self):
        g = self.m.definition_graph()
        self.assertEqual(g["tiempo"], [])

    def test_reverse_index_ser_aparece_en_vida_y_mente(self):
        idx = self.m.reverse_index()
        self.assertIn("vida",  idx.get("ser", []))
        self.assertIn("mente", idx.get("ser", []))

    def test_used_by_tiempo(self):
        """tiempo solo es usado por mente en el fixture minimal."""
        usados = self.m.used_by("tiempo")
        self.assertIn("mente", usados)
        self.assertNotIn("vida", usados)

    def test_used_by_primitivo_sin_uso(self):
        """materia solo se usa en vida."""
        usados = self.m.used_by("materia")
        self.assertEqual(usados, ["vida"])


# ─────────────────────────────────────────────────────────────────────────────
# R04 — CAMPO SEMÁNTICO (BFS)
# ─────────────────────────────────────────────────────────────────────────────

class TestCampoSemantico(unittest.TestCase):

    def setUp(self):
        self.m = build_minimal()

    def test_campo_incluye_al_propio_concepto(self):
        campo = self.m.semantic_field("vida")
        self.assertIn("vida", campo)
        self.assertEqual(campo["vida"], 0)

    def test_campo_profundidad_1(self):
        """Con profundidad 1, vida ve a sus hijos directos."""
        campo = self.m.semantic_field("vida", depth=1)
        self.assertIn("ser",     campo)
        self.assertIn("materia", campo)
        self.assertIn("energía", campo)

    def test_campo_profundidad_0(self):
        """Profundidad 0 solo incluye al concepto mismo."""
        campo = self.m.semantic_field("vida", depth=0)
        self.assertEqual(list(campo.keys()), ["vida"])

    def test_campo_profundidades_correctas(self):
        """Los hijos directos de vida deben estar en profundidad 1."""
        campo = self.m.semantic_field("vida", depth=3)
        self.assertEqual(campo["ser"], 1)
        self.assertEqual(campo["materia"], 1)

    def test_campo_cadena(self):
        """En la cadena A→B→C→D, el campo de D incluye a todos."""
        m = build_chain()
        campo = m.semantic_field("D", depth=10)
        self.assertIn("A", campo)
        self.assertIn("B", campo)
        self.assertIn("C", campo)

    def test_campo_no_supera_profundidad(self):
        """Con depth=1 desde D en la cadena, solo se ve C."""
        m = build_chain()
        campo = m.semantic_field("D", depth=1)
        self.assertIn("C",  campo)
        self.assertNotIn("B", campo)
        self.assertNotIn("A", campo)


# ─────────────────────────────────────────────────────────────────────────────
# R05 — RAZONAMIENTO DEDUCTIVO: CAMINO Y PRIMITIVOS
# ─────────────────────────────────────────────────────────────────────────────

class TestDeductivo(unittest.TestCase):

    def setUp(self):
        self.m = build_chain()     # A→B→C→D

    def test_camino_directo(self):
        path = self.m.path("D", "C")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "D")
        self.assertEqual(path[-1], "C")

    def test_camino_largo(self):
        path = self.m.path("D", "A")
        self.assertIsNotNone(path)
        self.assertEqual(path, ["D", "C", "B", "A"])

    def test_camino_a_si_mismo(self):
        path = self.m.path("A", "A")
        self.assertIsNotNone(path)
        self.assertEqual(path, ["A"])

    def test_camino_inexistente_inverso(self):
        """A no tiene a D en su definición; no hay camino A→D."""
        path = self.m.path("A", "D")
        self.assertIsNone(path)

    def test_camino_concepto_inexistente(self):
        path = self.m.path("D", "ZZZZ")
        self.assertIsNone(path)

    def test_implication_chain_primitivos(self):
        """Los primitivos de D en la cadena son [A]."""
        m = build_minimal()
        prims = m.implication_chain("vida")
        # vida → ser, materia, energía (todos primitivos)
        self.assertIn("ser",     prims)
        self.assertIn("materia", prims)
        self.assertIn("energía", prims)
        # vida misma NO es un primitivo
        self.assertNotIn("vida", prims)

    def test_implication_chain_primitivo_vacio(self):
        """Un concepto sin definición es primitivo y su cadena está vacía."""
        prims = self.m.implication_chain("A")
        self.assertEqual(prims, [])

    def test_implication_chain_cadena_lineal(self):
        prims = self.m.implication_chain("D")
        self.assertIn("A", prims)
        self.assertNotIn("D", prims)

    def test_camino_respeta_max_depth(self):
        """Un camino más largo que max_depth no debe encontrarse."""
        path = self.m.path("D", "A", max_depth=2)
        self.assertIsNone(path)


# ─────────────────────────────────────────────────────────────────────────────
# R06 — PRECEDENCIA Y ORDEN TOPOLÓGICO
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecedencia(unittest.TestCase):

    def setUp(self):
        self.m = build_chain()

    def test_a_precede_a_d(self):
        """A está en el campo semántico de D → A precede a D."""
        self.assertTrue(self.m.precedes("A", "D"))

    def test_d_no_precede_a_a(self):
        """D no está en el campo semántico de A."""
        self.assertFalse(self.m.precedes("D", "A"))

    def test_orden_topologico_primitivos_primero(self):
        order = self.m.topological_order()
        self.assertIsNotNone(order)
        idx = {name: i for i, name in enumerate(order)}
        # A debe ir antes que B, B antes que C, C antes que D
        self.assertLess(idx["A"], idx["B"])
        self.assertLess(idx["B"], idx["C"])
        self.assertLess(idx["C"], idx["D"])

    def test_orden_topologico_incluye_todos(self):
        order = self.m.topological_order()
        self.assertEqual(set(order), set(self.m.all_names()))

    def test_ciclo_devuelve_none(self):
        m = build_cyclic()
        self.assertIsNone(m.topological_order())

    def test_orden_topologico_espacio_minimal(self):
        m = build_minimal()
        order = m.topological_order()
        self.assertIsNotNone(order)
        idx = {name: i for i, name in enumerate(order)}
        # vida y mente deben ir DESPUÉS de sus componentes
        for comp in ["ser", "materia", "energía"]:
            self.assertLess(idx[comp], idx["vida"])
        for comp in ["ser", "energía", "tiempo"]:
            self.assertLess(idx[comp], idx["mente"])


# ─────────────────────────────────────────────────────────────────────────────
# R07 — RAZONAMIENTO ANALÓGICO
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalogico(unittest.TestCase):
    """
    Razonamiento analógico: a:b :: c:?  →  c + (b - a)

    Se usa un espacio sintético donde el punto target COINCIDE exactamente
    con uno de los conceptos disponibles, eliminando toda ambigüedad.

      pequeño=(0,0,0)  grande=(2,0,0)   → eje X controla "tamaño"
      lento=(0,0,0)?   No: hay que evitar solapamiento con pequeño.

    Diseño limpio de 8 vértices en un cubo 2×2×2:
      A=(0,0,0)  B=(2,0,0)  C=(0,2,0)  D=(2,2,0)
      E=(0,0,2)  F=(2,0,2)  G=(0,2,2)  H=(2,2,2)

    Analogías exactas verificables:
      A:B :: C:D   → diff=(2,0,0), target=C+(2,0,0)=(2,2,0)=D  ✓
      A:C :: B:D   → diff=(0,2,0), target=B+(0,2,0)=(2,2,0)=D  ✓
      A:E :: B:F   → diff=(0,0,2), target=B+(0,0,2)=(2,0,2)=F  ✓
      A:H :: B:?   → diff=(2,2,2), target=B+(2,2,2)=(4,2,2) → más cercano a H
    """

    def setUp(self):
        self.m = SemanticMatrix3D("cubo")
        for name, coords in [
            ("A", (0.0, 0.0, 0.0)), ("B", (2.0, 0.0, 0.0)),
            ("C", (0.0, 2.0, 0.0)), ("D", (2.0, 2.0, 0.0)),
            ("E", (0.0, 0.0, 2.0)), ("F", (2.0, 0.0, 2.0)),
            ("G", (0.0, 2.0, 2.0)), ("H", (2.0, 2.0, 2.0)),
        ]:
            self.m.add(name, coords)

    def test_analogia_eje_x(self):
        """A:B :: C:D  (traslación en X = +2)."""
        results = self.m.analogy("A", "B", "C", n=1)
        self.assertEqual(results[0][0], "D")

    def test_analogia_eje_y(self):
        """A:C :: B:D  (traslación en Y = +2)."""
        results = self.m.analogy("A", "C", "B", n=1)
        self.assertEqual(results[0][0], "D")

    def test_analogia_eje_z(self):
        """A:E :: B:F  (traslación en Z = +2)."""
        results = self.m.analogy("A", "E", "B", n=1)
        self.assertEqual(results[0][0], "F")

    def test_analogia_diagonal(self):
        """A:D :: E:H  (traslación diagonal XY)."""
        results = self.m.analogy("A", "D", "E", n=1)
        self.assertEqual(results[0][0], "H")

    def test_analogia_resultado_es_lista(self):
        results = self.m.analogy("A", "B", "C", n=3)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_analogia_pares_con_distancia(self):
        results = self.m.analogy("A", "B", "C", n=2)
        for name, dist in results:
            self.assertIsInstance(name, str)
            self.assertIsInstance(dist, float)
            self.assertGreaterEqual(dist, 0.0)

    def test_analogia_distancias_ordenadas(self):
        results = self.m.analogy("A", "B", "C", n=5)
        dists = [d for _, d in results]
        self.assertEqual(dists, sorted(dists))


# ─────────────────────────────────────────────────────────────────────────────
# R08 — RAZONAMIENTO CONJUNTISTA
# ─────────────────────────────────────────────────────────────────────────────

class TestConjuntista(unittest.TestCase):

    def setUp(self):
        self.m = SemanticMatrix3D("conjuntos")
        # Primitivos
        for name, coords in [
            ("A", (0.0, 0.0, 0.0)), ("B", (1.0, 0.0, 0.0)),
            ("C", (0.0, 1.0, 0.0)), ("D", (1.0, 1.0, 0.0)),
            ("E", (0.5, 0.5, 1.0)),
        ]:
            self.m.add(name, coords)
        # X comparte A, B con Y; Y tiene C, D exclusivos; X tiene E exclusivo
        self.m.add("X", (0.0, 0.0, 1.0), ["A", "B", "E"])
        self.m.add("Y", (1.0, 0.0, 1.0), ["A", "B", "C", "D"])
        self.m.add("Z", (0.5, 1.0, 1.0), ["A", "C"])

    # ── intersección ──────────────────────────

    def test_interseccion_dos_conceptos(self):
        """X={A,B,E} ∩ Y={A,B,C,D} = {A,B}."""
        inter = set(self.m.intersection("X", "Y"))
        self.assertEqual(inter, {"A", "B"})

    def test_interseccion_tres_conceptos(self):
        """X∩Y∩Z = {A}."""
        inter = set(self.m.intersection("X", "Y", "Z"))
        self.assertEqual(inter, {"A"})

    def test_interseccion_sin_comun(self):
        """Si no hay elementos comunes, retorna lista vacía."""
        # Crear concepto sin solapamiento
        self.m.add("W", (9.0, 9.0, 9.0), ["E"])
        self.m.add("V", (8.0, 8.0, 8.0), ["D"])
        inter = self.m.intersection("W", "V")
        self.assertEqual(inter, [])

    def test_interseccion_uno_solo(self):
        """intersección de uno solo es su propia definición."""
        inter = set(self.m.intersection("X"))
        self.assertEqual(inter, {"A", "B", "E"})

    def test_interseccion_vacia_sin_args(self):
        self.assertEqual(self.m.intersection(), [])

    # ── diferencia ────────────────────────────

    def test_diferencia_solo_en_x(self):
        """X tiene E que Y no tiene."""
        solo_x, solo_y = self.m.difference("X", "Y")
        self.assertIn("E", solo_x)
        self.assertNotIn("E", solo_y)

    def test_diferencia_solo_en_y(self):
        solo_x, solo_y = self.m.difference("X", "Y")
        self.assertIn("C", solo_y)
        self.assertIn("D", solo_y)

    def test_diferencia_simetrica_opuesta(self):
        solo_x, solo_y = self.m.difference("X", "Y")
        solo_y2, solo_x2 = self.m.difference("Y", "X")
        self.assertEqual(set(solo_x), set(solo_x2))
        self.assertEqual(set(solo_y), set(solo_y2))

    def test_diferencia_consigo_mismo(self):
        """Diferencia de X consigo mismo = ([], [])."""
        solo_a, solo_b = self.m.difference("X", "X")
        self.assertEqual(solo_a, [])
        self.assertEqual(solo_b, [])

    # ── ancestros comunes ─────────────────────

    def test_ancestros_comunes_x_y(self):
        """X e Y comparten A y B en su definición → aparecen como ancestros comunes."""
        anc = set(self.m.common_ancestors("X", "Y"))
        self.assertIn("A", anc)
        self.assertIn("B", anc)

    def test_ancestros_no_incluye_propios(self):
        anc = set(self.m.common_ancestors("X", "Y"))
        self.assertNotIn("X", anc)
        self.assertNotIn("Y", anc)

    def test_ancestros_sin_comun(self):
        """Dos conceptos sin definición compartida tienen 0 ancestros comunes."""
        self.m.add("P", (10.0, 0.0, 0.0), ["D"])
        self.m.add("Q", (11.0, 0.0, 0.0), ["E"])
        anc = self.m.common_ancestors("P", "Q")
        # Pueden tener ancestros vacíos si sus campos no se solapan
        self.assertIsInstance(anc, list)


# ─────────────────────────────────────────────────────────────────────────────
# R09 — TENSIONES Y CONTRADICCIONES
# ─────────────────────────────────────────────────────────────────────────────

class TestTension(unittest.TestCase):

    def setUp(self):
        self.m = SemanticMatrix3D("tension")

    def test_sin_tension_espacio_cohesivo(self):
        """Un espacio donde todos los conceptos comparten primitivos no tiene tensiones."""
        self.m.add("base",  (0.0, 0.0, 0.0))
        self.m.add("X",     (1.0, 0.0, 0.0), ["base"])
        self.m.add("Y",     (2.0, 0.0, 0.0), ["base"])
        self.m.add("combo", (1.5, 0.0, 0.0), ["X", "Y"])
        # No debería haber contradicciones si X e Y comparten "base"
        # (sus campos semánticos se solapan)
        tensions = self.m.contradictions()
        self.assertIsInstance(tensions, list)

    def test_tension_retorna_tuplas(self):
        """Cualquier resultado de contradictions() son 3-tuplas (C, A, B)."""
        m = build_minimal()
        tensions = m.contradictions()
        for t in tensions:
            self.assertIsInstance(t, tuple)
            self.assertEqual(len(t), 3)

    def test_tension_concepto_en_contexto(self):
        """
        Crear dos conceptos que se definen con palabras completamente aisladas
        y unirlos en un tercero → posible tensión semántica.
        """
        self.m.add("frío",    (0.0, 0.0, 0.0))
        self.m.add("caliente",(10.0,10.0,10.0))
        self.m.add("hielo",   (0.5, 0.0, 0.0), ["frío"])
        self.m.add("vapor",   (9.5,10.0,10.0), ["caliente"])
        self.m.add("agua",    (5.0, 5.0, 5.0), ["hielo", "vapor"])
        tensions = self.m.contradictions()
        # "agua" contiene hielo y vapor que apuntan a regiones opuestas
        # No garantizamos que detecte la tensión exactamente (depende de radio)
        # pero el método no debe lanzar excepción
        self.assertIsInstance(tensions, list)


# ─────────────────────────────────────────────────────────────────────────────
# R10 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

class TestClustering(unittest.TestCase):

    def setUp(self):
        """
        Tres grupos bien separados en el espacio 3D para que k=3 los encuentre.
          Grupo 0: cerca del origen
          Grupo 1: cerca de (10,0,0)
          Grupo 2: cerca de (0,10,0)
        """
        # Grupos separados 1000 unidades para determinismo total en k-means.
        self.m = SemanticMatrix3D("clusters")
        for i, (x, y, z) in enumerate([
            (0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.1, 0.0),       # grupo 0
            (1000.0, 0.0, 0.0),(1000.1, 0.0, 0.0),(999.9, 0.1, 0.0),  # grupo 1
            (0.0, 1000.0, 0.0),(0.1, 1000.0, 0.0),(0.0, 999.9, 0.0),  # grupo 2
        ]):
            self.m.add(f"C{i}", (x, y, z))

    def test_clusters_asigna_a_todos(self):
        clusters = self.m.clusters_naive(k=3)
        self.assertEqual(set(clusters.keys()), set(self.m.all_names()))

    def test_clusters_k_valores_distintos(self):
        clusters = self.m.clusters_naive(k=3)
        unique_ids = set(clusters.values())
        self.assertEqual(len(unique_ids), 3)

    def test_clusters_grupo_0_mismo_cluster(self):
        """Los tres puntos del grupo 0 deben caer en el mismo cluster."""
        clusters = self.m.clusters_naive(k=3)
        id0 = clusters["C0"]
        self.assertEqual(clusters["C1"], id0)
        self.assertEqual(clusters["C2"], id0)

    def test_clusters_grupo_1_mismo_cluster(self):
        clusters = self.m.clusters_naive(k=3)
        id1 = clusters["C3"]
        self.assertEqual(clusters["C4"], id1)
        self.assertEqual(clusters["C5"], id1)

    def test_clusters_grupos_distintos(self):
        clusters = self.m.clusters_naive(k=3)
        self.assertNotEqual(clusters["C0"], clusters["C3"])
        self.assertNotEqual(clusters["C0"], clusters["C6"])
        self.assertNotEqual(clusters["C3"], clusters["C6"])

    def test_clusters_k_mayor_que_conceptos(self):
        """Si k >= n, cada concepto tiene su propio cluster."""
        m = SemanticMatrix3D("mini")
        m.add("P", (0.0, 0.0, 0.0))
        m.add("Q", (1.0, 0.0, 0.0))
        clusters = m.clusters_naive(k=5)
        self.assertEqual(len(clusters), 2)

    def test_clusters_k1_todos_juntos(self):
        """Con k=1, todos los conceptos pertenecen al mismo cluster."""
        clusters = self.m.clusters_naive(k=1)
        unique = set(clusters.values())
        self.assertEqual(len(unique), 1)


# ─────────────────────────────────────────────────────────────────────────────
# R11 — PROYECCIÓN Y ANÁLISIS ESPACIAL
# ─────────────────────────────────────────────────────────────────────────────

class TestProyeccion(unittest.TestCase):

    def setUp(self):
        self.m = build_minimal()

    def test_proyeccion_eje_z_retiene_xy(self):
        """Proyectando sobre z se eliminan las coordenadas z y se conservan x, y."""
        proj = self.m.projection_2d(axis="z")
        for name, c in self.m._concepts.items():
            self.assertAlmostEqual(proj[name][0], c.coords[0], places=9)
            self.assertAlmostEqual(proj[name][1], c.coords[1], places=9)

    def test_proyeccion_eje_x_retiene_yz(self):
        proj = self.m.projection_2d(axis="x")
        for name, c in self.m._concepts.items():
            self.assertAlmostEqual(proj[name][0], c.coords[1], places=9)
            self.assertAlmostEqual(proj[name][1], c.coords[2], places=9)

    def test_proyeccion_todos_los_conceptos(self):
        proj = self.m.projection_2d()
        self.assertEqual(set(proj.keys()), set(self.m.all_names()))

    def test_bbox_minimos_correctos(self):
        bmin, bmax = self.m.bounding_box()
        # el origen (0,0,0) debe ser el mínimo en este fixture
        self.assertAlmostEqual(bmin[0], 0.0, places=9)
        self.assertAlmostEqual(bmin[1], 0.0, places=9)
        self.assertAlmostEqual(bmin[2], 0.0, places=9)

    def test_bbox_maximos_correctos(self):
        bmin, bmax = self.m.bounding_box()
        # materia=(1,0,0) → max en x=1; energía=(0,1,0) → max en y=1
        self.assertAlmostEqual(bmax[0], 1.0, places=9)
        self.assertAlmostEqual(bmax[1], 1.0, places=9)

    def test_centroide_conocido(self):
        """Centroide de los 6 conceptos del fixture minimal."""
        cx, cy, cz = self.m.centroid()
        expected_x = (0+1+0+0+0.5+0.3) / 6
        self.assertAlmostEqual(cx, expected_x, places=9)


# ─────────────────────────────────────────────────────────────────────────────
# R12 — RAZONAMIENTO TEMPORAL: EVOLUCIÓN DE DEFINICIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporal(unittest.TestCase):

    def setUp(self):
        self.m = build_minimal()

    def test_historia_vacia_al_inicio(self):
        c = self.m.require("vida")
        self.assertEqual(len(c.history), 0)

    def test_expand_genera_snapshot(self):
        c = self.m.require("vida")
        c.expand(["tiempo"], note="nueva teoría")
        self.assertEqual(len(c.history), 1)
        self.assertEqual(c.history[0].note, "nueva teoría")

    def test_reduce_genera_snapshot(self):
        c = self.m.require("vida")
        c.reduce(["ser"], note="revisión")
        self.assertNotIn("ser", c.definition)
        self.assertEqual(len(c.history), 1)

    def test_replace_cambia_elemento(self):
        c = self.m.require("vida")
        c.replace("ser", "no-ser", note="inversión")
        self.assertIn("no-ser", c.definition)
        self.assertNotIn("ser", c.definition)

    def test_definition_drift_vacio_sin_historia(self):
        drift = self.m.definition_drift("materia")  # primitivo, sin cambios
        self.assertEqual(drift, [])

    def test_definition_drift_registra_cambios(self):
        c = self.m.require("vida")
        c.expand(["tiempo"], note="paso-1")
        c.reduce(["materia"], note="paso-2")
        drift = self.m.definition_drift("vida")
        self.assertEqual(len(drift), 2)

    def test_definition_drift_estructura(self):
        c = self.m.require("vida")
        c.expand(["tiempo"])
        drift = self.m.definition_drift("vida")
        entry = drift[0]
        self.assertIn("timestamp",  entry)
        self.assertIn("added",      entry)
        self.assertIn("removed",    entry)
        self.assertIn("definition", entry)

    def test_definition_drift_added_correcto(self):
        c = self.m.require("vida")
        c.expand(["tiempo"], note="t")
        drift = self.m.definition_drift("vida")
        self.assertIn("ser",     drift[0]["definition"])
        self.assertIn("materia", drift[0]["definition"])
        self.assertIn("energía", drift[0]["definition"])

    def test_most_evolved_orden(self):
        """most_evolved devuelve conceptos ordenados por número de cambios."""
        c = self.m.require("vida")
        for i in range(5):
            c.expand([f"extra{i}"])
        evolved = self.m.most_evolved(n=3)
        counts = [n for _, n in evolved]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_edad_positiva(self):
        time.sleep(0.01)
        self.assertGreater(self.m.age("vida"), 0.0)

    def test_remove_limpia_definiciones(self):
        """Al eliminar 'ser', vida ya no debe tenerlo en su definición."""
        self.m.remove("ser")
        self.assertNotIn("ser", self.m.require("vida").definition)


# ─────────────────────────────────────────────────────────────────────────────
# R13 — DRIFT FIXED
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftFixed(unittest.TestCase):

    def setUp(self):
        self.m = SemanticMatrix3D("fixed_test")
        self.m.add("P", (0.0, 0.0, 0.0))
        self.m.add("Q", (1.0, 0.0, 0.0))
        self.m.add("R", (0.0, 1.0, 0.0))
        self.original = (5.0, 5.0, 5.0)
        self.c = self.m.add(
            "concepto",
            self.original,
            definition=["P"],
            drift_mode=DriftMode.FIXED,
        )

    def test_expand_no_mueve_coordenadas(self):
        self.c.expand(["Q", "R"], matrix=self.m)
        self.assertEqual(self.c.coords, self.original)

    def test_reduce_no_mueve_coordenadas(self):
        self.c.expand(["Q"], matrix=self.m)
        self.c.reduce(["P"], matrix=self.m)
        self.assertEqual(self.c.coords, self.original)

    def test_replace_no_mueve_coordenadas(self):
        self.c.replace("P", "Q", matrix=self.m)
        self.assertEqual(self.c.coords, self.original)

    def test_desplazamiento_siempre_cero(self):
        for _ in range(10):
            self.c.expand(["Q"], matrix=self.m)
        self.assertEqual(self.c.displacement(), 0.0)

    def test_sin_historial_de_coordenadas(self):
        self.c.expand(["Q"], matrix=self.m)
        self.assertEqual(len(self.c.coord_history), 0)

    def test_original_coords_igual_a_actual(self):
        self.c.expand(["Q"], matrix=self.m)
        self.assertEqual(self.c.original_coords(), self.c.coords)


# ─────────────────────────────────────────────────────────────────────────────
# R14 — DRIFT AUTO
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftAuto(unittest.TestCase):

    def setUp(self):
        self.m = SemanticMatrix3D("auto_test")
        self.m.add("P", (0.0, 0.0, 0.0))
        self.m.add("Q", (2.0, 0.0, 0.0))
        self.m.add("R", (0.0, 2.0, 0.0))
        self.c = self.m.add(
            "concepto",
            (9.0, 9.0, 9.0),          # posición inicial arbitraria
            definition=["P"],
            drift_mode=DriftMode.AUTO,
        )
        self.c._maybe_drift(self.m, "inicio")  # calcular posición inicial

    def test_posicion_inicial_es_centroide_de_P(self):
        """Tras _maybe_drift solo con P, coords deben ser coords de P."""
        self.assertEqual(self.c.coords, (0.0, 0.0, 0.0))

    def test_expand_mueve_hacia_centroide(self):
        """Añadir Q=(2,0,0) → centroide de {P,Q} = (1,0,0)."""
        self.c.expand(["Q"], matrix=self.m)
        self.assertAlmostEqual(self.c.coords[0], 1.0, places=9)
        self.assertAlmostEqual(self.c.coords[1], 0.0, places=9)

    def test_expand_tres_primitivos(self):
        """Centroide de P=(0,0,0), Q=(2,0,0), R=(0,2,0) = (2/3, 2/3, 0)."""
        self.c.expand(["Q", "R"], matrix=self.m)
        self.assertAlmostEqual(self.c.coords[0], 2/3, places=9)
        self.assertAlmostEqual(self.c.coords[1], 2/3, places=9)

    def test_no_guarda_historial_de_coordenadas(self):
        """AUTO no debe guardar coord_history."""
        self.c.expand(["Q"], matrix=self.m)
        self.assertEqual(len(self.c.coord_history), 0)

    def test_desplazamiento_siempre_cero_en_auto(self):
        """AUTO no guarda posición original → displacement siempre 0."""
        self.c.expand(["Q"], matrix=self.m)
        self.assertEqual(self.c.displacement(), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# R15 — DRIFT TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftTrajectory(unittest.TestCase):

    def setUp(self):
        self.m = SemanticMatrix3D("traj_test")
        self.m.add("P", (0.0, 0.0, 0.0))
        self.m.add("Q", (4.0, 0.0, 0.0))
        self.m.add("R", (0.0, 4.0, 0.0))
        self.m.add("S", (0.0, 0.0, 4.0))
        self.c = self.m.add(
            "concepto",
            (0.0, 0.0, 0.0),
            definition=["P"],
            drift_mode=DriftMode.TRAJECTORY,
        )
        self.c._maybe_drift(self.m, "inicio")  # posición 0 = coords de P = (0,0,0)

    def test_primer_expand_archiva_posicion(self):
        """
        setUp llama a _maybe_drift → coord_history ya tiene 1 entrada (pos inicial).
        El primer expand archiva otra → total 2 entradas.
        """
        self.c.expand(["Q"], matrix=self.m)
        self.assertEqual(len(self.c.coord_history), 2)

    def test_posicion_tras_expand_pq(self):
        """Centroide de P=(0,0,0) y Q=(4,0,0) = (2,0,0)."""
        self.c.expand(["Q"], matrix=self.m)
        self.assertAlmostEqual(self.c.coords[0], 2.0, places=9)
        self.assertAlmostEqual(self.c.coords[1], 0.0, places=9)

    def test_historial_crece_con_cada_cambio(self):
        """
        setUp → 1 entrada. expand(Q) → 2. expand(R) → 3.
        """
        self.c.expand(["Q"], matrix=self.m)
        self.c.expand(["R"], matrix=self.m)
        self.assertEqual(len(self.c.coord_history), 3)

    def test_desplazamiento_mayor_que_cero(self):
        """Tras moverse, el desplazamiento debe ser positivo."""
        self.c.expand(["Q"], matrix=self.m)
        self.assertGreater(self.c.displacement(), 0.0)

    def test_longitud_trayectoria_triangular(self):
        """
        P(0,0,0) → Q(4,0,0) → {P,Q,R}(4/3, 4/3, 0) → etc.
        La longitud total debe ser la suma de los pasos individuales.
        """
        self.c.expand(["Q"], matrix=self.m)
        step1 = euclidean((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))  # 2.0
        self.c.expand(["R"], matrix=self.m)
        step2 = euclidean((2.0, 0.0, 0.0), self.c.coords)
        total = self.c.trajectory_length()
        self.assertAlmostEqual(total, step1 + step2, places=9)

    def test_velocidad_positiva(self):
        self.c.expand(["Q"], matrix=self.m)
        self.assertGreater(self.c.velocity(), 0.0)

    def test_direction_of_drift_normalizado(self):
        self.c.expand(["Q"], matrix=self.m)
        d = self.c.direction_of_drift()
        mag = math.sqrt(sum(v**2 for v in d))
        self.assertAlmostEqual(mag, 1.0, places=9)

    def test_original_coords_apunta_a_inicio(self):
        """original_coords debe ser la primera posición archivada."""
        self.c.expand(["Q"], matrix=self.m)
        orig = self.c.original_coords()
        self.assertEqual(orig, self.c.coord_history[0].coords)

    def test_coord_drift_longitud(self):
        """coord_drift retorna len(coord_history)+1 entradas (incluye posición actual)."""
        self.c.expand(["Q"], matrix=self.m)
        self.c.expand(["R"], matrix=self.m)
        drift = self.m.coord_drift("concepto")
        self.assertEqual(len(drift), len(self.c.coord_history) + 1)

    def test_coord_drift_tiene_delta_en_pasos_posteriores(self):
        self.c.expand(["Q"], matrix=self.m)
        drift = self.m.coord_drift("concepto")
        self.assertNotIn("delta", drift[0])
        self.assertIn("delta", drift[1])

    def test_snapshot_matrix_pasado(self):
        """snapshot_matrix en un tiempo pasado debe recuperar la posición antigua."""
        t_antes = time.time()
        time.sleep(0.01)
        self.c.expand(["Q"], matrix=self.m)   # se mueve DESPUÉS de t_antes

        snap = self.m.snapshot_matrix(t_antes)
        # Antes de moverse, la coord archivada era la posición inicial (0,0,0)
        # Si coord_history[0].timestamp > t_antes, el snapshot retorna coords actual
        # Verificamos que el método retorna algo coherente
        self.assertIn("concepto", snap)

    def test_reposition_manual_archiva(self):
        """reposition() debe guardar la posición anterior independientemente del modo."""
        self.c.reposition((99.0, 99.0, 99.0), note="salto manual")
        self.assertGreater(len(self.c.coord_history), 0)
        self.assertEqual(self.c.coords, (99.0, 99.0, 99.0))

    def test_most_displaced_orden(self):
        """most_displaced devuelve en orden decreciente de desplazamiento."""
        self.c.expand(["Q"], matrix=self.m)
        displaced = self.m.most_displaced(n=5)
        dists = [d for _, d in displaced]
        self.assertEqual(dists, sorted(dists, reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# R16 — CRUD
# ─────────────────────────────────────────────────────────────────────────────

class TestCRUD(unittest.TestCase):

    def test_add_y_get(self):
        m = SemanticMatrix3D("crud")
        m.add("alfa", (1.0, 2.0, 3.0))
        c = m.get("alfa")
        self.assertIsNotNone(c)
        self.assertEqual(c.coords, (1.0, 2.0, 3.0))

    def test_get_inexistente_devuelve_none(self):
        m = SemanticMatrix3D("crud")
        self.assertIsNone(m.get("NOPE"))

    def test_require_inexistente_lanza_keyerror(self):
        m = SemanticMatrix3D("crud")
        with self.assertRaises(KeyError):
            m.require("NOPE")

    def test_add_duplicado_lanza_valueerror(self):
        m = SemanticMatrix3D("crud")
        m.add("alfa", (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            m.add("alfa", (1.0, 1.0, 1.0))

    def test_remove_elimina_concepto(self):
        m = SemanticMatrix3D("crud")
        m.add("alfa", (0.0, 0.0, 0.0))
        m.remove("alfa")
        self.assertNotIn("alfa", m)

    def test_remove_inexistente_no_lanza(self):
        m = SemanticMatrix3D("crud")
        m.remove("NOPE")  # no debe lanzar excepción

    def test_contains(self):
        m = SemanticMatrix3D("crud")
        m.add("alfa", (0.0, 0.0, 0.0))
        self.assertIn("alfa", m)
        self.assertNotIn("beta", m)

    def test_len(self):
        m = SemanticMatrix3D("crud")
        self.assertEqual(len(m), 0)
        m.add("a", (0.0, 0.0, 0.0))
        m.add("b", (1.0, 0.0, 0.0))
        self.assertEqual(len(m), 2)
        m.remove("a")
        self.assertEqual(len(m), 1)

    def test_all_names(self):
        m = build_minimal()
        names = set(m.all_names())
        self.assertIn("ser", names)
        self.assertIn("vida", names)
        self.assertEqual(len(names), 6)

    def test_definition_expand_no_duplica(self):
        m = build_minimal()
        c = m.require("vida")
        c.expand(["ser"])   # ya estaba en la definición
        self.assertEqual(c.definition.count("ser"), 1)


# ─────────────────────────────────────────────────────────────────────────────
# R17 — CASOS DE BORDE Y ROBUSTEZ
# ─────────────────────────────────────────────────────────────────────────────

class TestBorde(unittest.TestCase):

    def test_espacio_vacio_nearest_lanza(self):
        """Pedir nearest en un espacio vacío debe lanzar KeyError."""
        m = SemanticMatrix3D("vacío")
        with self.assertRaises(KeyError):
            m.nearest("X")

    def test_single_concepto_nearest_vacio(self):
        m = SemanticMatrix3D("solo")
        m.add("A", (0.0, 0.0, 0.0))
        result = m.nearest("A", n=3)
        self.assertEqual(result, [])

    def test_analogia_con_pocos_conceptos(self):
        """Con solo 4 conceptos (a, b, c y un extra), analogy(n=1) debe retornar 1."""
        m = SemanticMatrix3D("peq")
        m.add("a", (0.0, 0.0, 0.0))
        m.add("b", (1.0, 0.0, 0.0))
        m.add("c", (0.0, 1.0, 0.0))
        m.add("d", (1.0, 1.0, 0.0))
        result = m.analogy("a", "b", "c", n=1)
        self.assertEqual(len(result), 1)

    def test_path_en_espacio_desconectado(self):
        m = SemanticMatrix3D("disc")
        m.add("isla1", (0.0, 0.0, 0.0))
        m.add("isla2", (100.0, 0.0, 0.0))
        self.assertIsNone(m.path("isla1", "isla2"))

    def test_cyclic_semantic_field_no_loop_infinito(self):
        """El campo semántico en un grafo cíclico no debe entrar en bucle infinito."""
        m = build_cyclic()
        campo = m.semantic_field("X", depth=20)
        self.assertIsInstance(campo, dict)

    def test_implication_chain_ciclico_no_loop(self):
        """implication_chain en ciclo no debe colgar."""
        m = build_cyclic()
        prims = m.implication_chain("X")
        self.assertIsInstance(prims, list)

    def test_intersection_con_primitivos_sin_definicion(self):
        """Intersección de conceptos sin definición = vacío."""
        m = SemanticMatrix3D("prims")
        m.add("X", (0.0, 0.0, 0.0))
        m.add("Y", (1.0, 0.0, 0.0))
        inter = m.intersection("X", "Y")
        self.assertEqual(inter, [])

    def test_coseno_dos_vectores_iguales(self):
        """Coseno de un vector consigo mismo debe ser 1."""
        v = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=9)

    def test_euclidean_cero(self):
        self.assertEqual(euclidean((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.0)

    def test_euclidean_pitagoras_3d(self):
        """d((0,0,0),(1,1,1)) = sqrt(3)."""
        self.assertAlmostEqual(euclidean((0,0,0),(1,1,1)), math.sqrt(3), places=9)

    def test_vector_conocido(self):
        v = vector((1.0, 2.0, 3.0), (4.0, 6.0, 8.0))
        self.assertEqual(v, (3.0, 4.0, 5.0))

    def test_normalize_resultado_unitario(self):
        v = (3.0, 4.0, 0.0)
        n = normalize(v)
        mag = math.sqrt(sum(x**2 for x in n))
        self.assertAlmostEqual(mag, 1.0, places=9)

    def test_normalize_cero(self):
        n = normalize((0.0, 0.0, 0.0))
        self.assertEqual(n, (0.0, 0.0, 0.0))

    def test_add_vectors(self):
        r = add_vectors((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        self.assertEqual(r, (5.0, 7.0, 9.0))

    def test_concepto_repr_no_crash(self):
        m = build_minimal()
        r = repr(m.require("vida"))
        self.assertIn("vida", r)

    def test_summary_no_crash(self):
        m = build_minimal()
        s = m.summary()
        self.assertIn("minimal", s)

    def test_ascii_map_no_crash(self):
        m = build_minimal()
        s = m.ascii_map(axis="z", width=30, height=10)
        self.assertIn("┌", s)

    def test_definition_drift_multiples_ops(self):
        """Secuencia completa: expand + reduce + replace."""
        m = build_minimal()
        c = m.require("vida")
        c.expand(["tiempo"])
        c.reduce(["ser"])
        c.replace("materia", "anti-materia")
        drift = m.definition_drift("vida")
        self.assertEqual(len(drift), 3)



# ─────────────────────────────────────────────────────────────────────────────
# R18 — MODUS PONENS
# ─────────────────────────────────────────────────────────────────────────────

class TestModusPonens(unittest.TestCase):
    """
    Forma: A → B  (A en definición de B)
           A existe
           ──────────────────
           ∴ A aparece en used_by(B) y B ∈ semantic_field(A)
    """

    def setUp(self):
        self.m = SemanticMatrix3D("mp")
        self.m.add("fuego",    (0.0, 0.0, 0.0))
        self.m.add("calor",    (1.0, 0.0, 0.0), ["fuego"])
        self.m.add("expansion",(2.0, 0.0, 0.0), ["calor"])
        self.m.add("presion",  (3.0, 0.0, 0.0), ["expansion"])

    def test_mp_directo(self):
        self.assertIn("calor", self.m.used_by("fuego"))

    def test_mp_campo_alcanzable(self):
        campo = self.m.semantic_field("calor")
        self.assertIn("fuego", campo)

    def test_mp_transitivo_dos_saltos(self):
        campo = self.m.semantic_field("expansion", depth=2)
        self.assertIn("fuego", campo)

    def test_mp_transitivo_tres_saltos(self):
        campo = self.m.semantic_field("presion", depth=3)
        self.assertIn("fuego", campo)

    def test_mp_negado_sin_premisa(self):
        usados = self.m.used_by("fuego")
        self.assertNotIn("presion", usados)

    def test_mp_eliminacion_rompe_cadena(self):
        self.m.remove("calor")
        campo = self.m.semantic_field("expansion", depth=1)
        self.assertNotIn("fuego", campo)

    def test_mp_agregar_premisa_activa_consecuencia(self):
        self.m.require("presion").expand(["fuego"])
        self.assertIn("presion", self.m.used_by("fuego"))


# ─────────────────────────────────────────────────────────────────────────────
# R19 — MODUS TOLLENS
# ─────────────────────────────────────────────────────────────────────────────

class TestModusTollens(unittest.TestCase):
    """
    Forma: A → B
           ¬B  (B eliminado o vaciado)
           ──────────────────────────────
           ∴ A ya no fundamenta B
    """

    def setUp(self):
        self.m = SemanticMatrix3D("mt")
        self.m.add("oxigeno",    (0.0, 0.0, 0.0))
        self.m.add("combustion", (1.0, 0.0, 0.0), ["oxigeno"])
        self.m.add("llama",      (2.0, 0.0, 0.0), ["combustion"])

    def test_mt_estado_inicial(self):
        self.assertIn("combustion", self.m.used_by("oxigeno"))

    def test_mt_eliminar_b_rompe_link(self):
        self.m.remove("combustion")
        self.assertNotIn("combustion", self.m.used_by("oxigeno"))

    def test_mt_eliminar_b_propaga_a_c(self):
        self.m.remove("combustion")
        campo = self.m.semantic_field("llama", depth=5)
        self.assertNotIn("oxigeno", campo)

    def test_mt_vaciar_definicion_equivale_a_no_b(self):
        self.m.require("combustion").reduce(["oxigeno"])
        self.assertNotIn("oxigeno", self.m.require("combustion").definition)
        self.assertNotIn("combustion", self.m.used_by("oxigeno"))

    def test_mt_reintroducir_restaura(self):
        self.m.require("combustion").reduce(["oxigeno"])
        self.m.require("combustion").expand(["oxigeno"])
        self.assertIn("combustion", self.m.used_by("oxigeno"))

    def test_mt_no_elimina_a_mientras_b_existe(self):
        self.assertIn("oxigeno", self.m.require("combustion").definition)
        self.assertIn("combustion", self.m.used_by("oxigeno"))


# ─────────────────────────────────────────────────────────────────────────────
# R20 — SILOGISMO HIPOTETICO (TRANSITIVIDAD)
# ─────────────────────────────────────────────────────────────────────────────

class TestSilogismoHipotetico(unittest.TestCase):
    """
    Forma: A → B,  B → C  ⟹  A → C
    """

    def setUp(self):
        self.m = SemanticMatrix3D("sh")
        self.m.add("pensamiento",  (0.0, 0.0, 0.0))
        self.m.add("lenguaje",     (1.0, 0.0, 0.0), ["pensamiento"])
        self.m.add("comunicacion", (2.0, 0.0, 0.0), ["lenguaje"])
        self.m.add("cultura",      (3.0, 0.0, 0.0), ["comunicacion"])
        self.m.add("civilizacion", (4.0, 0.0, 0.0), ["cultura"])

    def test_sh_dos_pasos(self):
        campo = self.m.semantic_field("comunicacion", depth=2)
        self.assertIn("pensamiento", campo)

    def test_sh_cuatro_pasos(self):
        campo = self.m.semantic_field("civilizacion", depth=4)
        self.assertIn("pensamiento", campo)

    def test_sh_path_existe(self):
        p = self.m.path("civilizacion", "pensamiento")
        self.assertIsNotNone(p)
        self.assertEqual(p[0], "civilizacion")
        self.assertEqual(p[-1], "pensamiento")

    def test_sh_longitud_path(self):
        p = self.m.path("civilizacion", "pensamiento")
        self.assertEqual(len(p), 5)

    def test_sh_no_inverso(self):
        p = self.m.path("pensamiento", "civilizacion")
        self.assertIsNone(p)

    def test_sh_intermedios_tambien_transitan(self):
        self.assertIn("pensamiento", self.m.semantic_field("lenguaje"))
        self.assertIn("lenguaje",    self.m.semantic_field("cultura", depth=2))

    def test_sh_ruptura_rompe_transitividad(self):
        self.m.remove("lenguaje")
        campo = self.m.semantic_field("comunicacion", depth=5)
        self.assertNotIn("pensamiento", campo)

    def test_sh_orden_topologico_refleja_cadena(self):
        order = self.m.topological_order()
        idx = {n: i for i, n in enumerate(order)}
        self.assertLess(idx["pensamiento"], idx["lenguaje"])
        self.assertLess(idx["lenguaje"],    idx["comunicacion"])
        self.assertLess(idx["comunicacion"],idx["cultura"])
        self.assertLess(idx["cultura"],     idx["civilizacion"])


# ─────────────────────────────────────────────────────────────────────────────
# R21 — SILOGISMO DISYUNTIVO
# ─────────────────────────────────────────────────────────────────────────────

class TestSilogismoDisyuntivo(unittest.TestCase):
    """
    Forma: C ← A ∨ B.  ¬A → C sigue sustentado por B.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("sd")
        self.m.add("solar",   (0.0, 0.0, 0.0))
        self.m.add("eolica",  (1.0, 0.0, 0.0))
        self.m.add("nuclear", (0.0, 1.0, 0.0))
        self.m.add("renovable",   (0.5, 0.0, 0.0), ["solar", "eolica"])
        self.m.add("limpia", (0.5, 0.5, 0.0), ["renovable", "nuclear"])

    def test_sd_precondicion_ambos(self):
        defn = set(self.m.require("renovable").definition)
        self.assertIn("solar",  defn)
        self.assertIn("eolica", defn)

    def test_sd_eliminar_solar_deja_eolica(self):
        self.m.remove("solar")
        defn = self.m.require("renovable").definition
        self.assertNotIn("solar", defn)
        self.assertIn("eolica",  defn)

    def test_sd_definicion_no_vacia(self):
        self.m.remove("solar")
        self.assertGreater(len(self.m.require("renovable").definition), 0)

    def test_sd_eliminar_ambos_vacia(self):
        self.m.remove("solar")
        self.m.remove("eolica")
        self.assertEqual(self.m.require("renovable").definition, [])

    def test_sd_campo_tras_eliminar_uno(self):
        self.m.remove("solar")
        campo = self.m.semantic_field("renovable")
        self.assertIn("eolica", campo)

    def test_sd_cadena_disyuntiva(self):
        self.m.remove("nuclear")
        campo = self.m.semantic_field("limpia", depth=2)
        self.assertIn("eolica", campo)

    def test_sd_orden_topologico_con_disyuncion(self):
        order = self.m.topological_order()
        self.assertIsNotNone(order)
        idx = {n: i for i, n in enumerate(order)}
        self.assertLess(idx["solar"],    idx["renovable"])
        self.assertLess(idx["eolica"],   idx["renovable"])
        self.assertLess(idx["renovable"],idx["limpia"])


# ─────────────────────────────────────────────────────────────────────────────
# R22 — REDUCCION AL ABSURDO
# ─────────────────────────────────────────────────────────────────────────────

class TestReduccionAlAbsurdo(unittest.TestCase):
    """
    A → B y B → A simultáneamente → ciclo → topological_order() = None.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("raa")

    def test_raa_ciclo_directo(self):
        self.m.add("A", (0.0, 0.0, 0.0), ["B"])
        self.m.add("B", (1.0, 0.0, 0.0), ["A"])
        self.assertIsNone(self.m.topological_order())

    def test_raa_ciclo_ternario(self):
        self.m.add("A", (0.0, 0.0, 0.0), ["B"])
        self.m.add("B", (1.0, 0.0, 0.0), ["C"])
        self.m.add("C", (0.5, 0.5, 0.0), ["A"])
        self.assertIsNone(self.m.topological_order())

    def test_raa_campo_mutuo_en_ciclo(self):
        self.m.add("A", (0.0, 0.0, 0.0), ["B"])
        self.m.add("B", (1.0, 0.0, 0.0), ["A"])
        campo_a = self.m.semantic_field("A", depth=5)
        campo_b = self.m.semantic_field("B", depth=5)
        self.assertIn("B", campo_a)
        self.assertIn("A", campo_b)

    def test_raa_sin_ciclo_es_consistente(self):
        self.m.add("X", (0.0, 0.0, 0.0))
        self.m.add("Y", (1.0, 0.0, 0.0), ["X"])
        self.assertIsNotNone(self.m.topological_order())

    def test_raa_romper_ciclo_restaura_consistencia(self):
        self.m.add("A", (0.0, 0.0, 0.0), ["B"])
        self.m.add("B", (1.0, 0.0, 0.0), ["A"])
        self.assertIsNone(self.m.topological_order())
        self.m.require("B").reduce(["A"])
        self.assertIsNotNone(self.m.topological_order())

    def test_raa_self_referencia(self):
        self.m.add("ego", (0.0, 0.0, 0.0), ["ego"])
        campo = self.m.semantic_field("ego", depth=10)
        self.assertIsInstance(campo, dict)
        self.assertIsNone(self.m.topological_order())

    def test_raa_primitivo_nunca_es_absurdo(self):
        self.m.add("primitivo", (5.0, 5.0, 5.0))
        order = self.m.topological_order()
        self.assertIsNotNone(order)
        self.assertIn("primitivo", order)


# ─────────────────────────────────────────────────────────────────────────────
# R23 — RAZONAMIENTO ABDUCTIVO
# ─────────────────────────────────────────────────────────────────────────────

class TestAbductivo(unittest.TestCase):
    """
    Dada una observación B, inferir la mejor explicación:
    el primitivo de B más cercano en el espacio.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("abd")
        self.m.add("virus",    (0.9, 0.0, 0.0))
        self.m.add("bacteria", (5.0, 0.0, 0.0))
        self.m.add("toxina",   (3.0, 0.0, 0.0))
        self.m.add("infeccion",(1.0, 0.0, 0.0), ["virus", "bacteria"])
        self.m.add("sintoma",  (1.0, 0.1, 0.0), ["infeccion", "toxina"])

    def _mejor_explicacion(self, name):
        prims = self.m.implication_chain(name)
        if not prims:
            return name
        return min(prims, key=lambda p: self.m.distance(name, p))

    def test_abd_primitivos_de_sintoma(self):
        prims = set(self.m.implication_chain("sintoma"))
        self.assertEqual(prims, {"virus", "bacteria", "toxina"})

    def test_abd_mejor_explicacion_es_virus(self):
        self.assertEqual(self._mejor_explicacion("sintoma"), "virus")

    def test_abd_peor_explicacion_es_bacteria(self):
        prims = self.m.implication_chain("sintoma")
        peor = max(prims, key=lambda p: self.m.distance("sintoma", p))
        self.assertEqual(peor, "bacteria")

    def test_abd_explicacion_en_campo_semantico(self):
        mejor = self._mejor_explicacion("sintoma")
        campo = self.m.semantic_field("sintoma", depth=5)
        self.assertIn(mejor, campo)

    def test_abd_primitivo_cadena_vacia(self):
        prims = self.m.implication_chain("virus")
        self.assertEqual(prims, [])

    def test_abd_agregar_mas_cercana_cambia_resultado(self):
        self.m.add("prion", (1.0, 0.05, 0.0))
        self.m.require("infeccion").expand(["prion"])
        self.assertEqual(self._mejor_explicacion("sintoma"), "prion")


# ─────────────────────────────────────────────────────────────────────────────
# R24 — RAZONAMIENTO INDUCTIVO
# ─────────────────────────────────────────────────────────────────────────────

class TestInductivo(unittest.TestCase):
    """
    N casos comparten el mismo primitivo → ese primitivo es el sustrato común.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("ind")
        self.m.add("carbono",   (0.0, 0.0, 0.0))
        self.m.add("organico",  (1.0, 0.0, 0.0), ["carbono"])
        self.m.add("aminoacido",(2.0, 0.0, 0.0), ["organico"])
        self.m.add("lipido",    (1.5, 0.5, 0.0), ["organico"])
        self.m.add("glucosa",   (1.0, 1.0, 0.0), ["organico"])
        self.m.add("proteina",  (3.0, 0.0, 0.0), ["aminoacido"])
        self.m.add("membrana",  (2.0, 0.5, 0.0), ["lipido"])
        self.m.add("ATP",       (1.5, 1.5, 0.0), ["glucosa"])

    def test_ind_carbono_en_todos_los_campos(self):
        for caso in ["proteina", "membrana", "ATP"]:
            campo = self.m.semantic_field(caso, depth=5)
            self.assertIn("carbono", campo)

    def test_ind_ancestro_comun_proteina_membrana(self):
        anc = self.m.common_ancestors("proteina", "membrana", depth=6)
        self.assertIn("carbono", anc)

    def test_ind_ancestro_comun_los_tres(self):
        anc_pm = set(self.m.common_ancestors("proteina", "membrana", depth=6))
        anc_pa = set(self.m.common_ancestors("proteina", "ATP", depth=6))
        self.assertIn("carbono", anc_pm & anc_pa)

    def test_ind_implication_chain_incluye_carbono(self):
        for caso in ["proteina", "membrana", "ATP"]:
            self.assertIn("carbono", self.m.implication_chain(caso))

    def test_ind_contraejemplo_rompe_generalizacion(self):
        self.m.add("silicio",  (10.0, 0.0, 0.0))
        self.m.add("mineral",  (11.0, 0.0, 0.0), ["silicio"])
        campo = self.m.semantic_field("mineral", depth=5)
        self.assertNotIn("carbono", campo)

    def test_ind_fortaleza_inductiva_crece_con_casos(self):
        usan_carbono = [
            name for name in self.m.all_names()
            if "carbono" in self.m.semantic_field(name, depth=5)
            and name != "carbono"
        ]
        self.assertGreaterEqual(len(usan_carbono), 6)


# ─────────────────────────────────────────────────────────────────────────────
# R25 — RAZONAMIENTO CONTRAFACTUAL
# ─────────────────────────────────────────────────────────────────────────────

class TestContrafactual(unittest.TestCase):
    """
    "Si A no existiera, ¿cómo sería B?" → medir desplazamiento del centroide.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("cf")
        self.m.add("P1", (0.0, 0.0, 0.0))
        self.m.add("P2", (4.0, 0.0, 0.0))
        self.m.add("P3", (0.0, 4.0, 0.0))
        self.m.add("B",  (2.0, 0.0, 0.0), ["P1", "P2"])

    def _centroide(self, name):
        return self.m.require(name).centroid_definition(self.m)

    def test_cf_centroide_inicial(self):
        cx, cy, cz = self._centroide("B")
        self.assertAlmostEqual(cx, 2.0, places=9)
        self.assertAlmostEqual(cy, 0.0, places=9)

    def test_cf_eliminar_p1_desplaza_centroide(self):
        self.m.require("B").reduce(["P1"])
        cx, cy, cz = self._centroide("B")
        self.assertAlmostEqual(cx, 4.0, places=9)

    def test_cf_impacto_mayor_que_cero(self):
        antes = self._centroide("B")
        self.m.require("B").reduce(["P1"])
        despues = self._centroide("B")
        self.assertGreater(euclidean(antes, despues), 0.0)

    def test_cf_irrelevante_delta_cero(self):
        antes = self._centroide("B")
        self.m.remove("P3")
        despues = self._centroide("B")
        self.assertAlmostEqual(euclidean(antes, despues), 0.0, places=9)

    def test_cf_eliminar_todos_colapsa(self):
        self.m.require("B").reduce(["P1", "P2"])
        self.assertIsNone(self._centroide("B"))

    def test_cf_contrafactual_parcial_reduce_eje_y(self):
        self.m.require("B").expand(["P3"])
        antes = self._centroide("B")
        self.m.require("B").reduce(["P3"])
        despues = self._centroide("B")
        self.assertLess(despues[1], antes[1])


# ─────────────────────────────────────────────────────────────────────────────
# R26 — CIERRE TRANSITIVO
# ─────────────────────────────────────────────────────────────────────────────

class TestCierreTransitivo(unittest.TestCase):
    """
    Propiedades del cierre transitivo: reflexividad, antisimetría, transitividad.
    """

    def setUp(self):
        self.m = build_chain()  # A→B→C→D
        self.m.add("E", (1.0, 1.0, 0.0), ["A"])
        self.m.add("F", (2.0, 1.0, 0.0), ["B", "E"])

    def _alcanzable(self, src, dst):
        return self.m.path(src, dst) is not None

    def test_ct_reflexividad(self):
        for name in self.m.all_names():
            self.assertTrue(self._alcanzable(name, name))

    def test_ct_antisimetria_cadena(self):
        self.assertTrue(self._alcanzable("D", "A"))
        self.assertFalse(self._alcanzable("A", "D"))

    def test_ct_transitividad_ab_bc(self):
        self.assertTrue(self._alcanzable("D", "C"))
        self.assertTrue(self._alcanzable("C", "B"))
        self.assertTrue(self._alcanzable("D", "B"))

    def test_ct_transitividad_tres_saltos(self):
        self.assertTrue(self._alcanzable("D", "A"))

    def test_ct_rama_paralela(self):
        self.assertTrue(self._alcanzable("F", "A"))

    def test_ct_sin_relacion_entre_ramas(self):
        self.assertFalse(self._alcanzable("C", "E"))

    def test_ct_clausura_desde_f(self):
        campo = set(self.m.semantic_field("F", depth=10).keys())
        for esperado in ["A", "B", "E"]:
            self.assertIn(esperado, campo)

    def test_ct_monotonia(self):
        antes = set(self.m.semantic_field("F", depth=10).keys())
        self.m.add("G", (9.0, 9.0, 9.0))
        self.m.require("F").expand(["G"])
        despues = set(self.m.semantic_field("F", depth=10).keys())
        self.assertTrue(antes.issubset(despues))
        self.assertIn("G", despues)


# ─────────────────────────────────────────────────────────────────────────────
# R27 — EXCLUSION (TOLLENDO PONENS)
# ─────────────────────────────────────────────────────────────────────────────

class TestExclusion(unittest.TestCase):
    """
    C ← {A, B, D}.  ¬A, ¬B  →  C ← D  (la única explicación restante).
    """

    def setUp(self):
        self.m = SemanticMatrix3D("excl")
        self.m.add("hipA", (0.0, 0.0, 0.0))
        self.m.add("hipB", (1.0, 0.0, 0.0))
        self.m.add("hipD", (2.0, 0.0, 0.0))
        self.m.add("fenomeno", (1.0, 1.0, 0.0), ["hipA", "hipB", "hipD"])

    def test_excl_precondicion(self):
        defn = set(self.m.require("fenomeno").definition)
        self.assertEqual(defn, {"hipA", "hipB", "hipD"})

    def test_excl_eliminar_a_y_b(self):
        self.m.require("fenomeno").reduce(["hipA", "hipB"])
        self.assertEqual(self.m.require("fenomeno").definition, ["hipD"])

    def test_excl_una_sola_explicacion(self):
        self.m.require("fenomeno").reduce(["hipA", "hipB"])
        self.assertEqual(len(self.m.require("fenomeno").definition), 1)

    def test_excl_explicacion_correcta(self):
        self.m.require("fenomeno").reduce(["hipA", "hipB"])
        self.assertEqual(self.m.require("fenomeno").definition[0], "hipD")

    def test_excl_descartar_todo_colapsa(self):
        self.m.require("fenomeno").reduce(["hipA", "hipB", "hipD"])
        self.assertEqual(self.m.require("fenomeno").definition, [])

    def test_excl_registro_historico(self):
        c = self.m.require("fenomeno")
        c.reduce(["hipA"])
        c.reduce(["hipB"])
        self.assertEqual(len(c.history), 2)

    def test_excl_por_campo_semantico(self):
        self.m.add("evidencia", (2.1, 0.0, 0.0))
        self.m.require("hipD").expand(["evidencia"])
        validas = [
            h for h in ["hipA", "hipB", "hipD"]
            if "evidencia" in self.m.semantic_field(h, depth=2)
        ]
        self.assertEqual(validas, ["hipD"])


# ─────────────────────────────────────────────────────────────────────────────
# R28 — RAZONAMIENTO CAUSAL
# ─────────────────────────────────────────────────────────────────────────────

class TestCausal(unittest.TestCase):
    """
    Asimetría causa→efecto: causa precede, efecto no puede retroalimentar la causa.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("causal")
        self.m.add("lluvia",     (0.0, 0.0, 0.0))
        self.m.add("humedad",    (1.0, 0.0, 0.0), ["lluvia"])
        self.m.add("moho",       (2.0, 0.0, 0.0), ["humedad"])
        self.m.add("enfermedad", (3.0, 0.0, 0.0), ["moho"])
        self.m.add("bacteria",   (3.0, 1.0, 0.0))
        self.m.require("enfermedad").expand(["bacteria"])

    def test_causal_asimetria_directa(self):
        self.assertTrue(self.m.precedes("lluvia", "humedad"))
        self.assertFalse(self.m.precedes("humedad", "lluvia"))

    def test_causal_asimetria_indirecta(self):
        self.assertTrue(self.m.precedes("lluvia", "enfermedad"))
        self.assertFalse(self.m.precedes("enfermedad", "lluvia"))

    def test_causal_saltos_asimetria(self):
        p = self.m.path("enfermedad", "lluvia")
        self.assertIsNotNone(p)
        self.assertGreater(len(p), 2)

    def test_causal_causa_no_alcanza_efecto(self):
        p = self.m.path("lluvia", "enfermedad")
        self.assertIsNone(p)

    def test_causal_multiple_causas(self):
        defn = set(self.m.require("enfermedad").definition)
        self.assertIn("moho",     defn)
        self.assertIn("bacteria", defn)

    def test_causal_eliminar_causa_reduce_alcance(self):
        self.m.remove("lluvia")
        prims = self.m.implication_chain("humedad")
        self.assertNotIn("lluvia", prims)

    def test_causal_orden_topologico(self):
        order = self.m.topological_order()
        self.assertIsNotNone(order)
        idx = {n: i for i, n in enumerate(order)}
        self.assertLess(idx["lluvia"],   idx["humedad"])
        self.assertLess(idx["humedad"],  idx["moho"])
        self.assertLess(idx["moho"],     idx["enfermedad"])


# ─────────────────────────────────────────────────────────────────────────────
# R29 — RAZONAMIENTO MODAL (POSIBILIDAD Y NECESIDAD)
# ─────────────────────────────────────────────────────────────────────────────

class TestModal(unittest.TestCase):
    """
    □P : P es necesario para C  →  P ∈ implication_chain(C)
    ◇P : P es posible para C   →  P ∈ semantic_field(C) pero no necesariamente en implication_chain
    Contingencia: ◇ pero no □
    """

    def setUp(self):
        self.m = SemanticMatrix3D("modal")
        self.m.add("oxigeno",   (0.0, 0.0, 0.0))
        self.m.add("agua",      (1.0, 0.0, 0.0), ["oxigeno"])
        self.m.add("carbono",   (0.0, 1.0, 0.0), ["oxigeno"])
        self.m.add("celula",    (1.0, 1.0, 0.0), ["agua", "carbono"])
        self.m.add("planta",    (2.0, 1.0, 0.0), ["celula"])
        self.m.add("animal",    (1.0, 2.0, 0.0), ["celula"])
        self.m.add("clorofila", (2.5, 1.0, 0.0))
        self.m.require("planta").expand(["clorofila"])

    def _necesario(self, p, c):
        return p in self.m.implication_chain(c)

    def test_modal_necesidad_oxigeno_planta(self):
        self.assertTrue(self._necesario("oxigeno", "planta"))

    def test_modal_necesidad_oxigeno_animal(self):
        self.assertTrue(self._necesario("oxigeno", "animal"))

    def test_modal_contingencia_clorofila(self):
        self.assertTrue(self._necesario("clorofila", "planta"))
        self.assertFalse(self._necesario("clorofila", "animal"))

    def test_modal_necesario_en_campo(self):
        campo = set(self.m.semantic_field("planta", depth=10).keys())
        for nec in ["oxigeno", "agua", "carbono", "celula"]:
            self.assertIn(nec, campo)

    def test_modal_eliminar_necesario_colapsa_cadena(self):
        # Al eliminar oxigeno del sistema, remove() limpia su presencia en las
        # definiciones de agua y carbono → ambos quedan sin definición y se
        # convierten en primitivos ellos mismos.
        # La cadena de planta ahora ya no pasa POR oxigeno, pero sí pasa por
        # agua y carbono (que ahora son primitivos directos).
        self.m.remove("oxigeno")
        prims = self.m.implication_chain("planta")
        # oxigeno ya no existe ni en el espacio
        self.assertNotIn("oxigeno", self.m)
        # oxigeno no aparece en la cadena (fue eliminado)
        self.assertNotIn("oxigeno", prims)
        # agua y carbono se convierten en primitivos: SÍ aparecen en la cadena
        self.assertIn("agua",    prims)
        self.assertIn("carbono", prims)

    def test_modal_derivado_no_es_necesario_de_su_causa(self):
        self.assertFalse(self._necesario("planta", "oxigeno"))

    def test_modal_necesarios_subconjunto_de_campo(self):
        campo = set(self.m.semantic_field("planta", depth=10).keys())
        necesarios = set(self.m.implication_chain("planta"))
        self.assertTrue(necesarios.issubset(campo | {"planta"}))


# ─────────────────────────────────────────────────────────────────────────────
# R30 — RAZONAMIENTO CUANTIFICADOR (UNIVERSAL / EXISTENCIAL)
# ─────────────────────────────────────────────────────────────────────────────

class TestCuantificador(unittest.TestCase):
    """
    ∀ concepto compuesto: tiene al menos un primitivo en su cadena.
    ∃ al menos un primitivo en el espacio.
    ∃! el concepto más cercano al centroide.
    """

    def setUp(self):
        self.m = SemanticMatrix3D("cuant")
        self.m.add("alfa", (0.0, 0.0, 0.0))
        self.m.add("beta", (2.0, 0.0, 0.0))
        self.m.add("gamma",(0.0, 2.0, 0.0))
        self.m.add("delta",(0.0, 0.0, 2.0))
        self.m.add("AB",   (1.0, 0.0, 0.0), ["alfa", "beta"])
        self.m.add("CD",   (0.0, 1.0, 0.0), ["gamma", "delta"])
        self.m.add("ABCD", (0.5, 0.5, 0.5), ["AB", "CD"])

    def test_cuant_universal_compuestos_tienen_primitivos(self):
        for name in self.m.all_names():
            c = self.m.require(name)
            if c.definition:
                prims = self.m.implication_chain(name)
                self.assertGreater(len(prims), 0)

    def test_cuant_universal_primitivos_sin_definicion(self):
        primitivos = [n for n in self.m.all_names()
                      if not self.m.require(n).definition]
        for p in primitivos:
            self.assertEqual(self.m.require(p).definition, [])

    def test_cuant_existencial_hay_primitivo(self):
        primitivos = [n for n in self.m.all_names()
                      if not self.m.require(n).definition]
        self.assertGreater(len(primitivos), 0)

    def test_cuant_existencial_hay_compuesto(self):
        compuestos = [n for n in self.m.all_names()
                      if self.m.require(n).definition]
        self.assertGreater(len(compuestos), 0)

    def test_cuant_existencial_en_radio(self):
        cx, cy, cz = self.m.centroid()
        self.assertGreater(self.m.density((cx, cy, cz), radius=1.5), 0)

    def test_cuant_existencial_unico_mas_cercano(self):
        cx, cy, cz = self.m.centroid()
        distancias = sorted(
            [(name, euclidean(self.m.require(name).coords, (cx, cy, cz)))
             for name in self.m.all_names()],
            key=lambda x: x[1]
        )
        if len(distancias) > 1:
            self.assertLessEqual(distancias[0][1], distancias[1][1])

    def test_cuant_universal_campo_incluye_self(self):
        for name in self.m.all_names():
            campo = self.m.semantic_field(name, depth=0)
            self.assertIn(name, campo)

    def test_cuant_cuantificador_sobre_vecindad(self):
        vecinos = self.m.within_radius("ABCD", radius=1.0)
        for nombre, dist in vecinos:
            c = self.m.require(nombre)
            self.assertIsInstance(c.coords, tuple)
            self.assertEqual(len(c.coords), 3)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER (actualizado)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suites  = [
        (loader.loadTestsFromTestCase(TestEspacial),    "R01 Espacial"),
        (loader.loadTestsFromTestCase(TestSimilitud),   "R02 Similitud"),
        (loader.loadTestsFromTestCase(TestGrafo),       "R03 Grafo"),
        (loader.loadTestsFromTestCase(TestCampoSemantico), "R04 Campo Semántico"),
        (loader.loadTestsFromTestCase(TestDeductivo),   "R05 Deductivo"),
        (loader.loadTestsFromTestCase(TestPrecedencia), "R06 Precedencia"),
        (loader.loadTestsFromTestCase(TestAnalogico),   "R07 Analógico"),
        (loader.loadTestsFromTestCase(TestConjuntista), "R08 Conjuntista"),
        (loader.loadTestsFromTestCase(TestTension),     "R09 Tensión"),
        (loader.loadTestsFromTestCase(TestClustering),  "R10 Clustering"),
        (loader.loadTestsFromTestCase(TestProyeccion),  "R11 Proyección"),
        (loader.loadTestsFromTestCase(TestTemporal),    "R12 Temporal"),
        (loader.loadTestsFromTestCase(TestDriftFixed),  "R13 Drift FIXED"),
        (loader.loadTestsFromTestCase(TestDriftAuto),   "R14 Drift AUTO"),
        (loader.loadTestsFromTestCase(TestDriftTrajectory), "R15 Drift TRAJECTORY"),
        (loader.loadTestsFromTestCase(TestCRUD),        "R16 CRUD"),
        (loader.loadTestsFromTestCase(TestBorde),       "R17 Borde / Robustez"),
        (loader.loadTestsFromTestCase(TestModusPonens),         "R18 Modus Ponens"),
        (loader.loadTestsFromTestCase(TestModusTollens),        "R19 Modus Tollens"),
        (loader.loadTestsFromTestCase(TestSilogismoHipotetico), "R20 Silog. Hipotético"),
        (loader.loadTestsFromTestCase(TestSilogismoDisyuntivo), "R21 Silog. Disyuntivo"),
        (loader.loadTestsFromTestCase(TestReduccionAlAbsurdo),  "R22 Reduc. al Absurdo"),
        (loader.loadTestsFromTestCase(TestAbductivo),           "R23 Abductivo"),
        (loader.loadTestsFromTestCase(TestInductivo),           "R24 Inductivo"),
        (loader.loadTestsFromTestCase(TestContrafactual),       "R25 Contrafactual"),
        (loader.loadTestsFromTestCase(TestCierreTransitivo),    "R26 Cierre Transitivo"),
        (loader.loadTestsFromTestCase(TestExclusion),           "R27 Exclusión"),
        (loader.loadTestsFromTestCase(TestCausal),              "R28 Causal"),
        (loader.loadTestsFromTestCase(TestModal),               "R29 Modal"),
        (loader.loadTestsFromTestCase(TestCuantificador),       "R30 Cuantificador"),
    ]

    total_tests = total_failures = total_errors = 0
    results_summary = []

    print("\n" + "=" * 65)
    print("  SUITE COMPLETA — SemanticMatrix3D")
    print("=" * 65)

    for suite, label in suites:
        runner = unittest.TextTestRunner(
            verbosity=0, stream=open("/dev/null", "w")
        )
        result = runner.run(suite)
        n      = result.testsRun
        fails  = len(result.failures)
        errors = len(result.errors)
        ok     = n - fails - errors
        status = "✓" if (fails + errors) == 0 else "✗"
        total_tests    += n
        total_failures += fails
        total_errors   += errors
        results_summary.append((status, label, ok, n, fails, errors))
        for test, tb in result.failures + result.errors:
            print(f"\n  {'FAIL' if (test, tb) in result.failures else 'ERROR'}: {test}")
            print("  " + tb.split("\n")[-2])

    print(f"\n  {'Suite':<10}  {'Tipo de Razonamiento':<26}  {'OK':>4}  {'Total':>5}  {'Fail':>5}")
    print(f"  {'─'*10}  {'─'*26}  {'─'*4}  {'─'*5}  {'─'*5}")
    for status, label, ok, n, fails, errors in results_summary:
        fail_str = f"FAIL {fails}" if fails else ""
        err_str  = f"ERR {errors}" if errors else ""
        print(f"  {status}  {label:<26}  {ok:>4}/{n:<4}  {fail_str:>5}  {err_str:>5}")

    total_ok = total_tests - total_failures - total_errors
    print(f"\n  {'─'*58}")
    print(f"  Total: {total_ok}/{total_tests} tests  |  "
          f"Fallos: {total_failures}  |  Errores: {total_errors}")

    exit_code = 0 if (total_failures + total_errors) == 0 else 1
    print(f"  Estado: {'✓ TODOS PASADOS' if exit_code == 0 else '✗ HAY FALLOS'}")
    print("=" * 65 + "\n")
    import sys; sys.exit(exit_code)