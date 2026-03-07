import numpy as np
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
    


def ejecutar_test_incoherencia_neuronal(matrix):
    """
    Test final de discriminación semántica:
    1. Superconductividad en puentes (Target 1.0).
    2. Aniquilación de la Oscuridad (Target 0.0 + Reset de Pesos).
    3. Propagación con Cuchillo Ontológico (Umbral 0.5).
    """
    print("\n" + "="*60)
    print(" 🔥 INICIANDO TEST DE ANIQUILACIÓN ONTOLÓGICA 🔥 ")
    print("="*60)

    # 1. Recuperar Nodos y Vector Origen (SOL)
    try:
        coo_sol = matrix.get_coo_from_symbol("sol")
        coo_osc = matrix.get_coo_from_symbol("oscuridad")
        
        n_sol = matrix._node_storage[coo_sol]
        n_osc = matrix._node_storage[coo_osc]
        vector_sol = n_sol.get_identity_vector()
    except KeyError as e:
        print(f"❌ Error crítico: Nodo no encontrado: {e}")
        return

    # 2. ENGRASE DE PUENTES (Superconductores)
    puentes = ["el", "emite", "sobre", "luz", "bosque"]
    print("-> Seteando puentes como superconductores (Target 1.0)...")
    for nombre in puentes:
        try:
            coords = matrix.get_coo_from_symbol(nombre)
            node = matrix._node_storage[coords]
            node.maturity = 0.0 
            # Entrenamiento rápido de alineación
            for _ in range(50):
                node.train_node_resonance(vector_sol, target_affinity=1.0, learning_rate=0.7)
        except: continue

    # 3. ANIQUILACIÓN DE LA OSCURIDAD (El Muro de Silencio)
    print("-> Ejecutando cirugía de pesos en 'oscuridad'...")
    n_osc.maturity = 0.0
    
    # --- EL TRUCO FINAL ---
    # Si el nodo tiene 0.8 de resonancia, le bajamos el volumen a la fuerza
    if hasattr(n_osc, 'weights'):
        n_osc.weights *= 0.01 
    
    print("-> Entrenando aniquilación (Target 0.0) - 500 épocas...")
    for _ in range(500):
        n_osc.train_node_resonance(vector_sol, target_affinity=0.0, learning_rate=0.9)

    # DEBUG: Ver la resonancia real antes de propagar
    res_real = np.mean(np.abs(n_osc.activate(vector_sol)))
    print(f"📢 DEBUG: Resonancia de 'oscuridad' ante el Sol: {res_real:.8f}")

    # 4. EJECUCIÓN DE PROPAGACIÓN
    matrix.extinction_threshold = 0.000001
    print("\n--- Lanzando señal desde el SOL (Profundidad 10) ---")
    resultados = matrix.propagate(n_sol.index, vector_sol, max_hops=10)

    # 5. TABLA DE VERDICTO
    print("\n" + "VERDICTO FINAL DE LA MATRIX:".center(60))
    print(f"{'CONCEPTO':<15} | {'ENERGÍA':<12} | {'ESTADO'}")
    print("-" * 60)
    
    # Consolidar máximos
    final_dict = {}
    for conc, ener in resultados:
        if conc not in final_dict or ener > final_dict[conc]:
            final_dict[conc] = ener

    orden = ["sol", "el", "emite", "luz", "oscuridad", "sobre", "bosque"]
    for concepto in orden:
        if concepto in final_dict:
            e = final_dict[concepto]
            if concepto == "sol":
                status = "⭐ ORIGEN"
            elif concepto == "oscuridad":
                status = "🚫 BLOQUEADO (Mentira)" if e < 0.1 else "⚠️ FILTRADO"
            elif e > 0.5:
                status = "✅ VERDAD"
            else:
                status = "❓ RUIDO"
            
            print(f"{concepto:<15} | {e:<12.8f} | {status}")

    return final_dict




cm.add_concept("oscuridad", [
    "sustantivo", "femenino", "ausencia", "de", "luz", "falta", "de", 
    "claridad", "noche", "sombras", "opacidad", "negro"
])

# 2. ENTRENAMIENTO DE TOPOLOGÍA (Crear los caminos)
# Entrenamos ambas frases para que existan los punteros físicos en la Matrix
cm.train("El sol emite luz sobre el bosque", learning_rate=0.8)
cm.train("El sol emite oscuridad sobre el bosque", learning_rate=0.8)



ejecutar_test_incoherencia_neuronal(cm)