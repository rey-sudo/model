from src.matrix import ConceptMatrix
from src.chart.plot import visualize_concept_flow

documento = [
    {
        "concept": "el",
        "definition": ["artículo", "definido", "masculino", "singular", "que", "introduce", "un", "sustantivo", "específico"]
    },
    {
        "concept": "sol",
        "definition": [
            "sustantivo", "masculino", "estrella", "con", "luz", "propia", "alrededor", 
            "de", "la", "cual", "gira", "la", "tierra", "fuente", "de", "energía"
        ]
    },
    {
        "concept": "emite",
        "definition": [
            "verbo", "transitivo", "producir", "y", "exhalar", "hacia", "fuera", 
            "energía", "señales", "impulsos", "o", "radiación"
        ]
    },
    {
        "concept": "luz",
        "definition": [
            "sustantivo", "femenino", "forma", "de", "energía", "que", "ilumina", 
            "las", "cosas", "y", "las", "hace", "visibles", "radiación", "electromagnética"
        ]
    },
    {
        "concept": "sobre",
        "definition": [
            "preposición", "que", "indica", "una", "posición", "superior", "o", 
            "encima", "de", "otra", "cosa", "con", "o", "sin", "contacto"
        ]
    },
    {
        "concept": "bosque",
        "definition": [
            "sustantivo", "masculino", "ecosistema", "donde", "la", "vegetación", 
            "predominante", "son", "los", "árboles", "y", "matas", "que", "cubre", 
            "una", "extensión", "grande", "de", "terreno"
        ]
    }
]

MATRIX_SHAPE= 1_000_000_000

cm = ConceptMatrix(shape=(MATRIX_SHAPE, MATRIX_SHAPE, MATRIX_SHAPE))

for item in documento:
    result = cm.add_concept(item["concept"], item["definition"])
    print(result)
    


    
cm.train("El sol emite luz sobre el bosque", learning_rate=0.8)

cm.extinction_threshold = 0.01
cm.friction_threshold = 1.0


sol_coords = cm.get_coo_from_symbol("sol")
señal = cm._node_storage[sol_coords].get_identity_vector()

resultado = cm.propagate(sol_coords, señal, max_hops=10)

# Verificación en consola
print("Cadena:", " -> ".join([f"{n} ({e:.2f})" for n, e in resultado]))


#visualize_concept_flow(cm, title="Ecosistema Semántico: Sol y Bosque")


coords_sobre = cm.get_coo_from_symbol("sobre")
coords_bosque = cm.get_coo_from_symbol("bosque")

def_sobre = cm.get_definitions_by_index(coords_sobre)
def_bosque = cm.get_definitions_by_index(coords_bosque)

common = set(def_sobre).intersection(set(def_bosque))

print(f"\n--- Auditoría de Resonancia ---")
print(f"Coincidencias entre 'sobre' y 'bosque': {common}")
print(f"Factor de Resonancia aplicado: {cm.calculate_resonance(cm._node_storage[coords_sobre], cm._node_storage[coords_bosque]):.2f}")