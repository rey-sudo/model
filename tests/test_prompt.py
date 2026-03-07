"""
Test: Entrenar ConceptMatrix y hacer consultas semánticas por propagación de señal.
"""

import numpy as np
from prompt_toolkit import PromptSession
from src.matrix import ConceptMatrix
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
        "concept": "bosque",
        "definition": [
            "sustantivo", "masculino", "ecosistema", "donde", "la", "vegetación",
            "predominante", "son", "los", "árboles", "y", "matas", "que", "cubre",
            "una", "extensión", "grande", "de", "terreno"
        ]
    },
    {
        "concept": "árbol",
        "definition": [
            "sustantivo", "masculino", "planta", "perenne", "de", "tronco",
            "leñoso", "y", "elevado", "que", "se", "ramifica", "a", "cierta",
            "altura", "del", "suelo", "formando", "una", "copa"
        ]
    },
    {
        "concept": "suelo",
        "definition": [
            "sustantivo", "masculino", "superficie", "de", "la", "corteza",
            "terrestre", "biológicamente", "activa", "que", "proviene", "de",
            "la", "desintegración", "de", "las", "rocas", "y", "residuos", "orgánicos"
        ]
    },
    {
        "concept": "fauna",
        "definition": [
            "sustantivo", "femenino", "conjunto", "de", "los", "animales", "de",
            "un", "país", "región", "o", "medio", "determinado", "que", "viven",
            "en", "un", "estado", "temporal", "o", "permanente"
        ]
    },
    {
        "concept": "clima",
        "definition": [
            "sustantivo", "masculino", "condiciones", "atmosféricas",
            "propias", "de", "un", "lugar", "constituido", "por", "la", "cantidad",
            "y", "frecuencia", "de", "lluvias", "humedad", "y", "temperatura"
        ]
    },
    {
        "concept": "ecosistema",
        "definition": [
            "sustantivo", "masculino", "comunidad", "de", "seres", "vivos",
            "que", "interactúan", "entre", "sí", "y", "con", "su", "entorno",
            "físico", "formando", "un", "sistema", "ecológico"
        ]
    },
    {
        "concept": "planta",
        "definition": [
            "sustantivo", "femenino", "ser", "vivo", "autótrofo", "que",
            "realiza", "la", "fotosíntesis", "para", "producir", "su",
            "propio", "alimento", "a", "partir", "de", "luz", "y", "agua"
        ]
    },
    {
        "concept": "vegetación",
        "definition": [
            "conjunto", "plantas", "árboles", "flora",
            "que", "cubre", "superficie", "terreno",
            "ecosistema", "región"
        ]
    },
    {
        "concept": "comunidad",
        "definition": [
            "conjunto", "seres", "vivos",
            "conviven", "interactúan", "mismo",
            "entorno", "comparten", "recursos", "espacio"
        ]
    },
    {
        "concept": "condiciones",
        "definition": [
            "conjunto", "factores", "físicos",
            "temperatura", "humedad", "luz",
            "determinan", "entorno", "ecosistema"
        ]
    },
    {
        "concept": "flora",
        "definition": [
            "conjunto", "plantas", "vegetación",
            "propias", "región", "ecosistema",
            "convive", "fauna"
        ]
    },
    {
        "concept": "perenne",
        "definition": [
            "que", "dura", "todo",
            "año", "no", "pierde",
            "hojas", "planta", "árbol", "permanente"
        ]
    },
    {
        "concept": "atmósfera",
        "definition": [
            "capa", "gases", "rodea",
            "tierra", "contiene", "aire",
            "oxígeno", "clima", "temperatura"
        ]
    },
    {
        "concept": "terreno",
        "definition": [
            "extensión", "suelo", "tierra",
            "superficie", "donde", "crecen",
            "plantas", "bosque", "ecosistema"
        ]
    },
    {
        "concept": "raíz",
        "definition": [
            "parte", "planta", "árbol",
            "subterránea", "absorbe", "agua",
            "nutrientes", "suelo", "fija", "terreno"
        ]
    },
    {
        "concept": "agua",
        "definition": [
            "recurso", "líquido", "esencial",
            "vida", "seres", "vivos",
            "planta", "fauna", "ecosistema", "lluvia"
        ]
    },
    {
        "concept": "luz",
        "definition": [
            "energía", "solar", "radiante",
            "permite", "fotosíntesis", "planta",
            "crecimiento", "vegetación", "calor", "clima"
        ]
    },
    {
        "concept": "fotosíntesis",
        "definition": [
            "proceso", "planta", "convierte",
            "luz", "agua", "dióxido",
            "carbono", "alimento", "oxígeno"
        ]
    },
    {
        "concept": "nutrientes",
        "definition": [
            "sustancias", "suelo", "agua",
            "necesarias", "crecimiento", "planta",
            "árbol", "flora", "vida"
        ]
    },
    {
        "concept": "biodiversidad",
        "definition": [
            "variedad", "seres", "vivos",
            "fauna", "flora", "ecosistema",
            "bosque", "región", "especie"
        ]
    },
    {
        "concept": "subterránea",
        "definition": [
            "debajo", "superficie", "suelo",
            "tierra", "raíz", "agua",
            "nutrientes", "profundidad", "oscuridad"
        ]
    },
    {
        "concept": "convive",
        "definition": [
            "coexiste", "comparte", "espacio",
            "entorno", "ecosistema", "comunidad",
            "fauna", "flora", "relación", "equilibrio"
        ]
    },
    {
        "concept": "calor",
        "definition": [
            "energía", "térmica", "temperatura",
            "sol", "luz", "clima",
            "atmósfera", "permite", "vida", "crecimiento"
        ]
    },
    {
        "concept": "oxígeno",
        "definition": [
            "gas", "atmósfera", "esencial",
            "respiración", "seres", "vivos",
            "producido", "fotosíntesis", "planta", "vida"
        ]
    },
    {
        "concept": "región",
        "definition": [
            "área", "territorio", "geográfico",
            "ecosistema", "bosque", "fauna",
            "flora", "clima", "biodiversidad", "comunidad"
        ]
    },
    {
        "concept": "crecimiento",
        "definition": [
            "proceso", "aumento", "desarrollo",
            "planta", "árbol", "raíz",
            "nutrientes", "agua", "luz", "suelo"
        ]
    },
    {
        "concept": "permanente",
        "definition": [
            "constante", "duradero", "estable",
            "perenne", "ecosistema", "fauna",
            "flora", "comunidad", "entorno"
        ]
    },
    {
        "concept": "parte",
        "definition": [
            "porción", "componente", "elemento",
            "raíz", "tronco", "hojas",
            "planta", "árbol", "copa", "suelo"
        ]
    },
    {
        "concept": "hojas",
        "definition": [
            "órgano", "planta", "árbol",
            "fotosíntesis", "luz", "agua",
            "vegetación", "perenne", "verde", "copa"
        ]
    },
    {
        "concept": "ser",
        "definition": [
            "entidad", "vivo", "existente",
            "planta", "animal", "fauna",
            "ecosistema", "comunidad", "vida"
        ]
    },
    {
        "concept": "vivos",
        "definition": [
            "organismos", "activos", "existentes",
            "fauna", "flora", "planta",
            "árbol", "ecosistema", "comunidad", "vida"
        ]
    },
    {
        "concept": "subterránea",
        "definition": [
            "debajo", "superficie", "suelo",
            "tierra", "raíz", "agua",
            "nutrientes", "profundidad", "oscuridad"
        ]
    },
    {
        "concept": "convive",
        "definition": [
            "coexiste", "comparte", "espacio",
            "entorno", "ecosistema", "comunidad",
            "fauna", "flora", "relación", "equilibrio"
        ]
    },
    {
        "concept": "calor",
        "definition": [
            "energía", "térmica", "temperatura",
            "sol", "luz", "clima",
            "atmósfera", "permite", "vida", "crecimiento"
        ]
    },
    {
        "concept": "oxígeno",
        "definition": [
            "gas", "atmósfera", "esencial",
            "respiración", "seres", "vivos",
            "producido", "fotosíntesis", "planta", "vida"
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
    "el bosque tiene árboles y suelo fértil",
    "el árbol crece desde el suelo hacia el cielo",
    "el ecosistema incluye fauna y flora del bosque",
    "el clima determina la fauna y vegetación del bosque",
    "el suelo del bosque sostiene la planta y el árbol",
    "la planta necesita suelo clima y luz para crecer",
    "la fauna vive en el ecosistema del bosque",
    "el ecosistema depende del clima suelo y fauna",
    "árbol y planta forman la vegetación del bosque",
    "bosque ecosistema árbol suelo fauna clima planta",
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


def print_response(prompt_word: str):
    print(f"\n{'─' * 60}")
    print(f"  PROMPT  →  \"{prompt_word}\"")
    print(f"{'─' * 60}")
    results = query(prompt_word)

    if not results:
        print("  (sin resonancia — concepto aislado)")
        return

    print(f"  {'Concepto':<20} {'Energía':>10}  {'Barra'}")
    print(f"  {'─'*20} {'─'*10}  {'─'*20}")
    max_e = results[0][1] if results else 1
    for name, energy in results:
        bar_len = int((energy / max_e) * 20)
        bar = "█" * bar_len
        print(f"  {name:<20} {energy:>10.6f}  {bar}")


# ──────────────────────────────────────────────
# 6. FASE 4 — Prompts de consulta
# ──────────────────────────────────────────────

print("=" * 60)
print("FASE 3: Consultas semánticas por propagación de señal")
print("=" * 60)

prompts = [
    "bosque",       # Concepto central — debería activar árbol, suelo, fauna
    "árbol",        # Debería resonar con bosque, planta, suelo
    "ecosistema",   # Debería activar fauna, clima, bosque
    "clima",        # Debería activar bosque, fauna, ecosistema
    "planta",       # Debería resonar con árbol, suelo, bosque
]

for prompt in prompts:
    print_response(prompt)

print(f"\n{'=' * 60}")
print("Test completado.")
print("=" * 60)


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

        # ── query conversacional ─────────────────────
        else:
            word = raw.lower().split()[0]
            results = query(word, max_hops=max_hops, top_k=top_k)

            if not results:
                print(f"\n  Modelo › No tengo asociaciones para «{word}».\n")
            else:
                conceptos  = [name for name, _ in results]
                primario   = conceptos[0]
                secundarios = conceptos[1:]

                if secundarios:
                    lista = ", ".join(secundarios[:-1])
                    ultimo = secundarios[-1]
                    cola = f"{lista} y {ultimo}" if lista else ultimo
                    frase = f"{word} evoca {primario}, que a su vez resuena con {cola}."
                else:
                    frase = f"{word} evoca directamente {primario}."

                print(f"\n  Modelo › {frase}\n")


repl()