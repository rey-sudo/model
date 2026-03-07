"""
Test: Entrenar ConceptMatrix y hacer consultas semánticas por propagación de señal.
"""

import numpy as np
from prompt_toolkit import PromptSession
from src.matrix import STOPWORDS, ConceptMatrix
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

# ──────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ──────────────────────────────────────────────

MATRIX_SHAPE = 1_000_000_000
cm = ConceptMatrix(shape=(MATRIX_SHAPE, MATRIX_SHAPE, MATRIX_SHAPE))


# ──────────────────────────────────────────────
# 2. CORPUS DE ENTRENAMIENTO
# ──────────────────────────────────────────────


ecosistema_lexico = [
  {
    "concept": "forest",
    "definition": [
      "noun", "a", "large", "area", "of", "land", "covered", "with", "trees", "and", "underbrush"
    ]
  },
  {
    "concept": "tree",
    "definition": [
      "noun", "a", "woody", "perennial", "plant", "having", "a", "single", "usually", "elongate", "main", "stem", "generally", "with", "few", "or", "no", "branches", "on", "its", "lower", "part"
    ]
  },
  {
    "concept": "soil",
    "definition": [
      "noun", "the", "upper", "layer", "of", "earth", "in", "which", "plants", "grow", "a", "black", "or", "dark", "brown", "material", "typically", "consisting", "of", "a", "mixture", "of", "organic", "remains", "clay", "and", "rock", "particles"
    ]
  },
  {
    "concept": "fauna",
    "definition": [
      "noun", "the", "animals", "of", "a", "particular", "region", "habitat", "or", "geological", "period"
    ]
  },
  {
    "concept": "climate",
    "definition": [
      "noun", "the", "weather", "conditions", "prevailing", "in", "an", "area", "in", "general", "or", "over", "a", "long", "period"
    ]
  },
  {
    "concept": "ecosystem",
    "definition": [
      "noun", "a", "biological", "community", "of", "interacting", "organisms", "and", "their", "physical", "environment"
    ]
  },
  {
    "concept": "plant",
    "definition": [
      "noun", "a", "living", "organism", "typically", "growing", "in", "a", "permanent", "site", "absorbing", "water", "and", "inorganic", "substances", "through", "its", "roots", "and", "synthesizing", "nutrients", "in", "its", "leaves", "by", "photosynthesis"
    ]
  },
  {
    "concept": "vegetation",
    "definition": [
      "noun", "plants", "considered", "collectively", "especially", "those", "found", "in", "a", "particular", "area", "or", "habitat"
    ]
  },
  {
    "concept": "community",
    "definition": [
      "noun", "a", "group", "of", "interdependent", "organisms", "of", "different", "species", "growing", "or", "living", "together", "in", "a", "specified", "habitat"
    ]
  },
  {
    "concept": "conditions",
    "definition": [
      "noun", "plural", "the", "circumstances", "or", "factors", "that", "affect", "the", "way", "in", "which", "people", "live", "or", "work", "especially", "with", "regard", "to", "their", "well-being"
    ]
  },
  {
    "concept": "flora",
    "definition": [
      "noun", "the", "plants", "of", "a", "particular", "region", "habitat", "or", "geological", "period"
    ]
  },
  {
    "concept": "perennial",
    "definition": [
      "adjective", "of", "a", "plant", "living", "for", "several", "years", "typically", "with", "new", "growth", "of", "herbaceous", "parts", "from", "a", "part", "that", "survives", "from", "season", "to", "season"
    ]
  },
  {
    "concept": "atmosphere",
    "definition": [
      "noun", "the", "envelope", "of", "gases", "surrounding", "the", "earth", "or", "another", "planet"
    ]
  },
  {
    "concept": "terrain",
    "definition": [
      "noun", "a", "stretch", "of", "land", "especially", "with", "regard", "to", "its", "physical", "features"
    ]
  },
  {
    "concept": "root",
    "definition": [
      "noun", "the", "part", "of", "a", "plant", "which", "attaches", "it", "to", "the", "ground", "or", "to", "a", "support", "conveying", "water", "and", "nourishment", "to", "the", "rest", "of", "the", "plant"
    ]
  },
  {
    "concept": "water",
    "definition": [
      "noun", "a", "colorless", "transparent", "odorless", "liquid", "that", "forms", "the", "seas", "lakes", "rivers", "and", "rain", "and", "is", "the", "basis", "of", "the", "fluids", "of", "living", "organisms"
    ]
  },
  {
    "concept": "light",
    "definition": [
      "noun", "the", "natural", "agent", "that", "stimulates", "sight", "and", "makes", "things", "visible"
    ]
  },
  {
    "concept": "photosynthesis",
    "definition": [
      "noun", "the", "process", "by", "which", "green", "plants", "and", "some", "other", "organisms", "use", "sunlight", "to", "synthesize", "foods", "from", "carbon", "dioxide", "and", "water"
    ]
  },
  {
    "concept": "nutrients",
    "definition": [
      "noun", "plural", "substances", "that", "provide", "nourishment", "essential", "for", "growth", "and", "the", "maintenance", "of", "life"
    ]
  },
  {
    "concept": "biodiversity",
    "definition": [
      "noun", "the", "variety", "of", "life", "in", "the", "world", "or", "in", "a", "particular", "habitat", "or", "ecosystem"
    ]
  },
  {
    "concept": "underground",
    "definition": [
      "adjective", "situated", "done", "or", "used", "beneath", "the", "surface", "of", "the", "ground"
    ]
  },
  {
    "concept": "coexists",
    "definition": [
      "verb", "exist", "at", "the", "same", "time", "or", "in", "the", "same", "place"
    ]
  },
  {
    "concept": "heat",
    "definition": [
      "noun", "the", "quality", "of", "being", "hot", "high", "temperature"
    ]
  },
  {
    "concept": "oxygen",
    "definition": [
      "noun", "a", "colorless", "odorless", "reactive", "gas", "the", "chemical", "element", "of", "atomic", "number", "8", "and", "the", "life-supporting", "component", "of", "the", "air"
    ]
  },
  {
    "concept": "region",
    "definition": [
      "noun", "an", "area", "or", "division", "especially", "part", "of", "a", "country", "or", "the", "world", "having", "definable", "characteristics", "but", "not", "always", "fixed", "boundaries"
    ]
  },
  {
    "concept": "growth",
    "definition": [
      "noun", "the", "process", "of", "increasing", "in", "physical", "size"
    ]
  },
  {
    "concept": "permanent",
    "definition": [
      "adjective", "lasting", "or", "intended", "to", "last", "or", "remain", "unchanged", "indefinitely"
    ]
  },
  {
    "concept": "part",
    "definition": [
      "noun", "an", "amount", "or", "section", "which", "with", "others", "makes", "up", "the", "whole", "of", "something"
    ]
  },
  {
    "concept": "leaves",
    "definition": [
      "noun", "plural", "flattened", "structures", "of", "a", "higher", "plant", "typically", "green", "and", "bladelike", "that", "are", "attached", "to", "a", "stem", "and", "are", "the", "main", "organs", "of", "photosynthesis"
    ]
  },
  {
    "concept": "being",
    "definition": [
      "noun", "the", "nature", "or", "essence", "of", "a", "person", "or", "thing"
    ]
  },
  {
    "concept": "living",
    "definition": [
      "adjective", "alive", "not", "dead"
    ]
  },
  {
    "concept": "underground",
    "definition": [
      "adjective", "situated", "done", "or", "used", "beneath", "the", "surface", "of", "the", "ground"
    ]
  },
  {
    "concept": "coexists",
    "definition": [
      "verb", "exist", "at", "the", "same", "time", "or", "in", "the", "same", "place"
    ]
  },
  {
    "concept": "heat",
    "definition": [
      "noun", "the", "quality", "of", "being", "hot", "high", "temperature"
    ]
  },
  {
    "concept": "oxygen",
    "definition": [
      "noun", "a", "colorless", "odorless", "reactive", "gas", "the", "life-supporting", "component", "of", "the", "air"
    ]
  }
]






# ──────────────────────────────────────────────
# 3. FASE 1 — Registrar definiciones en la matriz
# ──────────────────────────────────────────────

print("=" * 60)
print("FASE 1: Registrando conceptos en la matriz...")
print("=" * 60)

for item in ecosistema_lexico:
    coords = cm.add_concept(item["concept"], item["definition"])
    print(f"  ✓ '{item['concept']}' → {coords}")

print(f"\nConceptos registrados: {cm.nnz} celdas ocupadas\n")


# ──────────────────────────────────────────────
# 4. FASE 2 — Entrenamiento Hebbiano
# ──────────────────────────────────────────────

print("=" * 60)
print("FASE 2: Entrenamiento Hebbiano con corpus...")
print("=" * 60)

corpus = [
    "The forest has trees and fertile soil",

    "The tree grows from the ground to the sky",

    "The ecosystem includes the forest's fauna and flora",

    "The climate determines the forest's fauna and vegetation",

    "The forest soil supports the plants and trees",

    "Plants need soil, climate, and light to grow",

    "Fauna lives in the forest ecosystem",

    "The ecosystem depends on climate, soil, and fauna",

    "Trees and plants make up the forest's vegetation",

    "forest ecosystem tree soil fauna climate plant"
]

for i, sentence in enumerate(corpus, 1):
    print(f"  [{i:02d}] \"{sentence}\"")
    cm.train(sentence, learning_rate=0.08)

print()


# ──────────────────────────────────────────────
# 5. FASE 3 — Función de consulta (prompt)
# ──────────────────────────────────────────────

def query(concept_word: str, max_hops: int = 6, top_k: int = 8) -> list:
    """
    Dado un concepto de entrada, propaga su señal de identidad
    por la red y retorna los conceptos más resonantes.
    """
    coords = cm.get_coo_from_symbol(concept_word)
    
    # Si el nodo no existe aún, lo creamos on-the-fly
    if coords not in cm._node_storage:
        cm.add_node(coords, concept_word)

    node = cm._node_storage[coords]
    initial_signal = node.get_identity_vector()  # señal = identidad del concepto

    results = cm.propagate(
        start_coords=coords,
        initial_signal=initial_signal,
        max_hops=max_hops
    )

    # Eliminar el propio concepto de la respuesta y limitar a top_k
    filtered = [(name, energy) for name, energy in results if name != concept_word]
    return filtered[:top_k]


def query_frase(frase: str, max_hops: int = 6, top_k: int = 8) -> list:
    words = frase.lower().split()
    
    if not words:
        return []

    # Acumular energías de todas las palabras
    energias_totales = {}

    for word in words:
        coords = cm.get_coo_from_symbol(word)
        if coords not in cm._node_storage:
            cm.add_node(coords, word)

        node = cm._node_storage[coords]
        signal = node.get_identity_vector()
        resultados = cm.propagate(coords, signal, max_hops=max_hops)

        for name, energy in resultados:
            if name not in words:  # no incluir las palabras del propio prompt
                energias_totales[name] = energias_totales.get(name, 0) + energy

    # Normalizar por cantidad de palabras
    n = len(words)
    energias_totales = {k: v / n for k, v in energias_totales.items()}

    return sorted(energias_totales.items(), key=lambda x: x[1], reverse=True)[:top_k]






# ──────────────────────────────────────────────
# 7. REPL INTERACTIVO (prompt_toolkit)
# ──────────────────────────────────────────────

HELP_TEXT = """
  Comandos disponibles:
  ─────────────────────────────────────────────
  <palabra>          → propagar señal desde ese concepto
  :train <oración>   → entrenar con una oración nueva
  :hops <n>          → cambiar profundidad máxima (default: 6)
  :top <n>           → cambiar cantidad de resultados (default: 8)
  :conceptos         → listar todos los nodos registrados
  :stats             → estadísticas de la matriz
  :help              → mostrar este menú
  :salir             → cerrar el REPL
  ─────────────────────────────────────────────
"""

repl_style = Style.from_dict({
    "prompt": "#00aa00 bold",
})


def repl():
    max_hops = 6
    top_k    = 8

    session = PromptSession(
        history=InMemoryHistory(),
        style=repl_style,
    )

    print(f"\n{'=' * 60}")
    print("  ConceptMatrix REPL  —  escribe :help para ver comandos")
    print(f"{'=' * 60}\n")

    while True:
        try:
            raw = session.prompt(HTML("<prompt>  › </prompt>")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cerrando REPL.\n")
            break

        if not raw:
            continue

        # ── comandos especiales ──────────────────────
        if raw in (":salir", ":exit", ":q"):
            print("\n  Hasta luego.\n")
            break

        elif raw == ":help":
            print(HELP_TEXT)

        elif raw == ":stats":
            print(f"\n  Celdas ocupadas : {cm.nnz}")
            print(f"  Nodos activos   : {len(cm._node_storage)}")
            print(f"  Densidad        : {cm.density:.2e}")
            print(f"  max_hops        : {max_hops}")
            print(f"  top_k           : {top_k}\n")

        elif raw == ":conceptos":
            nombres = sorted(n.name for n in cm._node_storage.values())
            print(f"\n  {len(nombres)} conceptos registrados:")
            cols = 4
            for i in range(0, len(nombres), cols):
                fila = nombres[i:i + cols]
                print("  " + "  ".join(f"{n:<18}" for n in fila))
            print()

        elif raw.startswith(":hops "):
            try:
                max_hops = int(raw.split()[1])
                print(f"  max_hops → {max_hops}\n")
            except (ValueError, IndexError):
                print("  Uso: :hops <número>\n")

        elif raw.startswith(":top "):
            try:
                top_k = int(raw.split()[1])
                print(f"  top_k → {top_k}\n")
            except (ValueError, IndexError):
                print("  Uso: :top <número>\n")

        elif raw.startswith(":train "):
            sentence = raw[7:].strip()
            if sentence:
                cm.train(sentence, learning_rate=0.08)
                print(f"  ✓ Entrenado: \"{sentence}\"\n")
            else:
                print("  Uso: :train <oración>\n")

        elif raw.startswith(":generar "):
            partes = raw.split()
            seed   = partes[1] if len(partes) > 1 else "el"
            largo  = int(partes[2]) if len(partes) > 2 else 6
            frase  = cm.generar(seed, longitud=largo)
            print(f"\n  Generado › {frase}\n")

        # ── query conversacional ─────────────────────
        else:
            results = query_frase(raw, max_hops=max_hops, top_k=top_k)

            if not results:
                print(f"\n  Modelo › No tengo asociaciones para «{raw}».\n")
            else:
                print(f"\n{'─' * 60}")
                print(f"  PROMPT  →  \"{raw}\"")
                print(f"{'─' * 60}")
                print(f"  {'Concepto':<22} {'Energía':>10}  Resonancia")
                print(f"  {'─'*22} {'─'*10}  {'─'*20}")
                max_e = results[0][1]
                for name, energy in results:
                    bar = "█" * int((energy / max_e) * 20)
                    print(f"  {name:<22} {energy:>10.6f}  {bar}")
                print()

repl()