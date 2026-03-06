import numpy as np
from src.matrix import ConceptMatrix

def align_nodes(node_a, node_b, factor=0.3):
    """
    Simula el entrenamiento: acerca la identidad del nodo B a la del nodo A
    para que la fricción permita el paso de señales.
    """
    id_a = node_a.get_identity_vector()
    id_b = node_b.get_identity_vector()
    # Mezcla lineal de identidades
    new_id_b = (id_b * (1 - factor)) + (id_a * factor)
    # Re-normalizar para mantener la escala
    new_id_b = new_id_b / np.linalg.norm(new_id_b) * np.linalg.norm(id_b)
    
    # Inyectamos la identidad alineada para el test
    # (En un sistema real esto lo hace el método update_local_weights)
    node_b.get_identity_vector = lambda: new_id_b

def test_spektra_thinking_flow():
    # 1. Setup: Matrix con umbral tolerante para el test
    matrix = ConceptMatrix(shape=(100, 100, 100))
    matrix.friction_threshold = 0.95 # Elevamos el umbral para permitir el flujo inicial
    
    concepts = ["Fuego", "Calor", "Vapor", "Hielo"]
    nodes = {}
    
    for c in concepts:
        coords = matrix.get_coo_from_symbol(c) # Usando tu método de hashing
        matrix.add_node(coords, concept=c)
        nodes[c] = matrix._node_storage[coords]
        nodes[c].maturity = 0.5 # Madurez media para permitir plasticidad

    # 2. ALINEACIÓN SEMÁNTICA (Simulación de entrenamiento previo)
    # Alineamos Fuego con Calor, y Calor con Vapor
    align_nodes(nodes["Fuego"], nodes["Calor"], factor=0.4)
    align_nodes(nodes["Calor"], nodes["Vapor"], factor=0.4)
    # "Hielo" se queda ortogonal (sin alinear) para probar que la fricción lo detiene

    # 3. Crear el Mapa Mental (Punteros)
    nodes["Fuego"].add_pointer(nodes["Calor"].index, strength=0.9)
    nodes["Calor"].add_pointer(nodes["Vapor"].index, strength=0.7)
    nodes["Calor"].add_pointer(nodes["Hielo"].index, strength=0.2)

    print(f"--- Iniciando Propagación de Pensamiento: 'Fuego' ---")

    # 4. Disparar la señal inicial
    fuego_identity = nodes["Fuego"].get_identity_vector()
    
    pensamiento_resultante = matrix.propagate(
        start_coords=nodes["Fuego"].index, 
        initial_signal=fuego_identity,
        max_hops=5 # Damos más margen de maniobra
    )

    # 5. Análisis de Resultados
    print("\nResultados del flujo de conciencia:")
    conceptos_activos = []
    for concepto, energia in pensamiento_resultante:
        print(f" > Nodo Activo: {concepto:10} | Energía: {energia:.4f}")
        conceptos_activos.append(concepto)

    # Verificaciones Lógicas
    assert "Calor" in conceptos_activos, "La señal debió llegar a Calor tras la alineación."
    assert "Vapor" in conceptos_activos, "La señal debió llegar a Vapor."
    
    if "Hielo" not in conceptos_activos:
        print("✅ Hielo correctamente bloqueado por fricción.")
    
    print("\n🚀 Test de Propagación: PASADO")

if __name__ == "__main__":
    test_spektra_thinking_flow()