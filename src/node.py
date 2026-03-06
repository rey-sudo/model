from typing import Any, Tuple
import numpy as np
import hashlib


class ConceptNode:
    def __init__(self, matrix_ref: Any, index: Tuple[int,...], concept_name):
        """
        Inicialización del nodo basada en la identidad del concepto.
        """
        from src.matrix import ConceptMatrix
        # 1. Identidad e Inmutabilidad
        self.index = index
        self.matrix = matrix_ref
        self.name = concept_name
        self.seed = self._generate_deterministic_seed(concept_name)
        
        # 2. Red Neuronal Local (Micro-red de 1,000 neuronas)
        # Usamos la semilla para que los pesos iniciales sean siempre los mismos
        self.rs = np.random.RandomState(self.seed)
        
        scale = np.sqrt(2.0 / 1000)
        self.weights = self.rs.randn(1000, 1000) * scale
        self.bias = np.zeros((1000, 1))
        
        # 3. Estructura de Relaciones (Punteros a Coordenadas Inmutables)
        # { (x, y, z): weight_strength }
        self.pointers = {}
        
        # 4. Estado de Evolución
        self.maturity = 0.0  # 0.0 (plástico) a 1.0 (consolidado)
        self.activation_count = 0

    def get_concept_definition(self):
        """
        El nodo consulta a la matriz su propia definición 
        en tiempo real para ajustar su comportamiento.
        """
        # 2. Consultar la definición guardada en la MatrixStorage
        definition_coords = self.matrix.get(self.index)
        
        if definition_coords and isinstance(definition_coords, list):
            # Retornar los objetos ConceptNode de mi definición
            return [self.matrix._node_storage.get(c) for c in definition_coords]
        return []

    def _generate_deterministic_seed(self, concept_name):
        # Usamos el prefijo 'WEIGHTS' para que la semilla de la red neuronal
        # sea distinta a la de la coordenada, pero igual de predecible.
        hash_input = f"WEIGHTS_{concept_name}"
        return int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % (2**32)

    def get_identity_vector(self):
        """
        Retorna el vector de identidad inmutable del concepto.
        Se genera de forma determinista a partir de la semilla original.
        """
        # Creamos un generador local para no alterar el estado de self.rs
        rs_identity = np.random.RandomState(self.seed)
        
        # Generamos un vector de 1000x1 que representa el "Axioma" del nodo
        identity = rs_identity.randn(1000, 1) * 0.1
        return identity
    
    def activate(self, input_vector):
        """
        Procesa una señal entrante y produce una salida.
        y = tanh(Wx + b)
        """
        # Asegurar que el vector sea una columna de 1000x1
        input_vector = input_vector.reshape(1000, 1)
        
        # Transformación no lineal (Función de activación)
        output = np.tanh(np.dot(self.weights, input_vector) + self.bias)
        
        self.activation_count += 1
        return output

    def add_pointer(self, target_coords, strength=0.1):
        """
        Crea o refuerza un enlace hacia otra coordenada absoluta.
        """
        if target_coords not in self.pointers:
            self.pointers[target_coords] = strength
        else:
            # El refuerzo es inversamente proporcional a la madurez
            refuerzo = strength * (1.0 - self.maturity)
            self.pointers[target_coords] += refuerzo
            
        # Limitar la fuerza del enlace a 1.0
        self.pointers[target_coords] = min(1.0, self.pointers[target_coords])

    def update_local_weights(self, gradient, learning_rate=0.01):
        """
        Ajusta los pesos internos (Backpropagation local).
        No afecta a otros nodos, solo a la 'inteligencia' de este concepto.
        """
        if self.maturity < 1.0:
            # Modulador de aprendizaje basado en madurez
            effective_lr = learning_rate * (1.0 - self.maturity)
            self.weights -= effective_lr * gradient
            
            # El aprendizaje aumenta la madurez del nodo
            self.maturity = min(1.0, self.maturity + 0.001)

    def get_top_pointers(self, limit=5):
        """Retorna las coordenadas de los conceptos más relacionados."""
        sorted_ptrs = sorted(self.pointers.items(), key=lambda x: x[1], reverse=True)
        return sorted_ptrs[:limit]