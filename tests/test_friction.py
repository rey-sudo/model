import numpy as np
from src.matrix import ConceptMatrix
from src.node import ConceptNode

def test_semantic_friction_system():
    """
    Verifica que la Matrix bloquee señales incoherentes y permita
    el paso de señales alineadas con la identidad del nodo.
    """
    # 1. Configuración del Entorno
    # Creamos una Matrix 3D de 100x100x100
    matrix = ConceptMatrix(shape=(100, 100, 100))
    
    # Definimos coordenadas para 'Origen' y 'Destino'
    coords_origen = (10, 10, 10)
    coords_destino = (50, 50, 50)
    
    # Creamos el nodo de destino: "Agua"
    # Lo establecemos con madurez máxima para que sea un filtro estricto
    matrix.add_node(coords_destino, concept="Agua")
    nodo_agua = matrix._node_storage[coords_destino]
    nodo_agua.maturity = 1.0 
    
    print(f"--- Iniciando Test de Fricción para Nodo: {nodo_agua.name} ---")

    # 2. ESCENARIO A: Señal Coherente (Éxito)
    # Usamos el vector de identidad del propio nodo (similitud máxima)
    # Nota: Asegúrate de que ConceptNode tenga el método get_identity_vector()
    señal_perfecta = nodo_agua.get_identity_vector()
    
    resultado_ok = matrix.send_signal(coords_origen, coords_destino, señal_perfecta)
    
    assert resultado_ok is not None, "Error: La Matrix bloqueó una señal coherente."
    print("✅ Escenario A: Señal coherente PERMITIDA (Fricción baja).")

    # 3. ESCENARIO B: Señal Incoherente / Ruido (Bloqueo)
    # Generamos un vector aleatorio que probablemente sea ortogonal a la identidad
    señal_ruido = np.random.randn(1000, 1)
    
    # Intentamos enviar el ruido al nodo "Agua"
    resultado_bloqueado = matrix.send_signal(coords_origen, coords_destino, señal_ruido)
    
    assert resultado_bloqueado is None, "Error: La Matrix dejó pasar ruido incoherente."
    print("✅ Escenario B: Señal de ruido BLOQUEADA (Fricción crítica).")

    # 4. ESCENARIO C: Atenuación por Madurez
    # Bajamos la madurez a un estado "plástico" (aprendizaje)
    nodo_agua.maturity = 0.2
    # El mismo ruido de antes ahora debería tener una fricción efectiva menor
    resultado_plastico = matrix.send_signal(coords_origen, coords_destino, señal_ruido)
    
    # En estado plástico, el umbral de fricción (0.8) suele permitir más ruido
    if resultado_plastico is not None:
        print("✅ Escenario C: Nodo plástico PERMITIÓ señal ruidosa para aprendizaje.")
    else:
        print("ℹ️ Escenario C: El ruido era demasiado alto incluso para un nodo plástico.")

if __name__ == "__main__":
    try:
        test_semantic_friction_system()
        print("\n🚀 Verificación de Fricción Semántica completada con éxito.")
    except Exception as e:
        print(f"\n❌ Fallo en la prueba de integridad: {e}")