import numpy as np
import hashlib
from src.node import ConceptNode # Asumiendo que tu clase está en src/node.py

# Mock de la Matrix para las pruebas
class MockMatrix:
    def __init__(self):
        self._node_storage = {}
    def get(self, index):
        return self._node_storage.get(index)

def test_node_deterministic_identity():
    """
    Verifica que dos nodos creados con el mismo nombre tengan 
    exactamente los mismos pesos iniciales (Identidad Inmutable).
    """
    matrix = MockMatrix()
    node1 = ConceptNode(matrix, (1, 1, 1), "Justicia")
    node2 = ConceptNode(matrix, (1, 1, 1), "Justicia")
    
    # Los pesos deben ser idénticos bit a bit
    np.testing.assert_array_equal(node1.weights, node2.weights)
    assert node1.seed == node2.seed
    print("✅ Test de Identidad Determinista: PASADO")

def test_slfn_activation_shape():
    """
    Verifica que la activación de la SLFN produzca un vector 
    de salida con la dimensionalidad correcta (1000, 1).
    """
    matrix = MockMatrix()
    node = ConceptNode(matrix, (0, 0, 0), "Energía")
    
    input_vector = np.random.rand(1000, 1)
    output = node.activate(input_vector)
    
    assert output.shape == (1000, 1)
    assert np.all(output >= -1) and np.all(output <= 1) # Por la función tanh
    print("✅ Test de Activación SLFN (Shape y Rango): PASADO")

def test_maturity_and_learning_saturation():
    """
    Verifica que a medida que aumenta la madurez, el aprendizaje 
    se vuelve más difícil (resistencia al cambio).
    """
    matrix = MockMatrix()
    node = ConceptNode(matrix, (10, 10, 10), "Axioma")
    
    initial_weights = node.weights.copy()
    gradient = np.ones((1000, 1000)) * 0.1
    
    # 1. Aprender cuando es plástico (maturity 0)
    node.update_local_weights(gradient, learning_rate=0.5)
    assert not np.array_equal(node.weights, initial_weights)
    assert node.maturity > 0
    
    # 2. Forzar madurez máxima
    node.maturity = 1.0
    weights_at_maturity = node.weights.copy()
    
    # Intentar aprender cuando está consolidado
    node.update_local_weights(gradient, learning_rate=0.5)
    
    # Los pesos NO deberían haber cambiado
    np.testing.assert_array_equal(node.weights, weights_at_maturity)
    print("✅ Test de Madurez y Saturación de Aprendizaje: PASADO")

def test_pointer_reinforcement():
    """
    Verifica que los punteros se refuerzan pero no superan el límite de 1.0.
    """
    matrix = MockMatrix()
    node = ConceptNode(matrix, (0,0,0), "Origen")
    target = (1, 1, 1)
    
    node.add_pointer(target, strength=0.6)
    node.add_pointer(target, strength=0.6) # Refuerzo
    
    assert node.pointers[target] == 1.0
    print("✅ Test de Refuerzo de Punteros: PASADO")

if __name__ == "__main__":
    print("--- Iniciando Batería de Pruebas SPEKTRA: ConceptNode ---")
    try:
        test_node_deterministic_identity()
        test_slfn_activation_shape()
        test_maturity_and_learning_saturation()
        test_pointer_reinforcement()
        print("\n🚀 Todas las pruebas lógicas han pasado exitosamente.")
    except AssertionError as e:
        print(f"\n❌ Error en la verificación lógica: {e}")