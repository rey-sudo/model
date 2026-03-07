import numpy as np
import sparse
import plotly.graph_objects as go
from typing import Any, List, Tuple, Union
from src.node import ConceptNode
from src.utils.hashing import coordinates_from_index, posiciones_en_abecedario

STOPWORDS = {"el", "la", "los", "las", "de", "del", "y", "o", 
             "a", "en", "que", "se", "un", "una", "su", "por", 
             "con", "para", "es", "son", "al",
             # categorías gramaticales
             "sustantivo", "masculino", "femenino", "verbo", "adjetivo",
             # si son muchos
             "tiene", "crece", "incluye"
            }

# Sentinel for empty cells
_EMPTY = object()

CoordList = List[Tuple[int, ...]]
Value = Union[float, int, str, CoordList, Any]


class ConceptMatrix:
    """
    Sparse N-dimensional matrix backed by a dict store.
    Each cell can hold any value: scalar, string, or an ordered
    list of coordinate tuples [(x1,y1,z1), (x2,y2,z2), ...].
    Empty cells return the _EMPTY sentinel — check with is_empty().
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self._matrix_storage: dict[Tuple[int, ...], Value] = {}
        self._node_storage = {}
        self.friction_threshold = 0.8
        self.extinction_threshold = 0.000001

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, index: Tuple[int, ...]):
        if len(index) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)}-D index, got {len(index)}-D.")
        for axis, (i, dim) in enumerate(zip(index, self.shape)):
            if not (0 <= i < dim):
                raise IndexError(f"Index {i} out of bounds for axis {axis} (size {dim}).")

    @staticmethod
    def _is_coord_list(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(c, (list, tuple)) for c in value)
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def set(self, index: Tuple[int, ...], concept: str, value: Value):
        """
        Store any value at index.
          cm.set((x,y,z), 3.14)                    # scalar
          cm.set((x,y,z), [(4,3,3), (8,9,0)])      # ordered coord list
          cm.set((x,y,z), "label")                  # string
          cm.set((x,y,z), None)                     # removes the cell
        """
        self._validate(index)
        
        #impl validate concept 
        
        if value is None:
            self._matrix_storage.pop(index, None)
        elif self._is_coord_list(value):
            self._matrix_storage[index] = [tuple(c) for c in value]  # preserve order
            self.add_node(index, concept)
        else:
            self._matrix_storage[index] = value
            self.add_node(index, concept)
        
            
    def add_node(self, index: Tuple[int,...], concept: str):
        self._node_storage[index] = ConceptNode(self, index, concept)
            
    def get(self, index: Tuple[int, ...]) -> Value:
        """Return stored value, or _EMPTY sentinel if the cell is empty."""
        self._validate(index)
        return self._matrix_storage.get(index, _EMPTY)

    def is_empty(self, index: Tuple[int, ...]) -> bool:
        """True if the cell has no stored value."""
        return self.get(index) is _EMPTY

    def delete(self, index: Tuple[int, ...]):
        """Remove a cell (reset to empty)."""
        self._validate(index)
        self._matrix_storage.pop(index, None)

    # ------------------------------------------------------------------
    # Sparse export (scalars only)
    # ------------------------------------------------------------------

    def to_coo(self) -> sparse.COO:
        """
        Export numeric cells to sparse.COO.
        Cells containing coord lists or strings are skipped.
        """
        scalar_items = {
            k: v for k, v in self._matrix_storage.items()
            if isinstance(v, (int, float))
        }
        if not scalar_items:
            return sparse.COO(
                coords=np.zeros((len(self.shape), 0), dtype=int),
                data=np.array([], dtype=float),
                shape=self.shape,
            )
        coords = np.array(list(scalar_items.keys()), dtype=int).T
        data = np.array(list(scalar_items.values()), dtype=float)
        return sparse.COO(coords=coords, data=data, shape=self.shape)

    def get_coo_from_symbol(self, concept: str) -> tuple[int, int, int]:
        concepto_raw_index = posiciones_en_abecedario(concept)
        concept_index = coordinates_from_index(concepto_raw_index)
        
        return concept_index

    def add_concept(self, concept: str, definition: list[str]) -> tuple[int, int, int]:
        concepto_raw_index = posiciones_en_abecedario(concept)
        concept_index = coordinates_from_index(concepto_raw_index)

        concept_definition = []

        for c in definition:
            c_array = posiciones_en_abecedario(c)
            coo = coordinates_from_index(c_array)
            concept_definition.append(coo)        
        
        self.set(concept_index, concept, concept_definition)
        
        definition_sentence = concept + " " + " ".join(definition)
        self.train(definition_sentence, learning_rate=0.1)
        
        return concept_index


    def train(self, text: str, learning_rate: float = 0.05):
        """
        Entrenamiento por asociación temporal (Aprendizaje Hebbiano)
        y Sincronización de Resonancia Neuronal.
        """
        
        words = [w for w in text.lower().split() if w not in STOPWORDS]
        
        # 1. Recuperar o crear los objetos ConceptNode
        sequence = []
        for word in words:
            coords = self.get_coo_from_symbol(word)
            if coords not in self._node_storage:
                self.add_node(coords, concept=word)
            sequence.append(self._node_storage[coords])

        # 2. Refuerzo de Ventana (Contexto y Redes Internas)
        for i, node in enumerate(sequence):
            # Ventana de contexto: 2 palabras adelante y 2 atrás
            start = max(0, i - 2)
            end = min(len(sequence), i + 3)
            
            for j in range(start, end):
                if i == j: continue
                
                neighbor = sequence[j]
                
                # --- CAPA 1: Asociación de Punteros (El "Dónde") ---
                # Calculamos la distancia (a más cerca, más fuerza)
                dist_factor = 1.0 / abs(i - j)
                strength = learning_rate * dist_factor
                
                # El nodo crea o fortalece el puntero hacia su vecino
                node.add_pointer(neighbor.index, strength=strength)

                # --- CAPA 2: Sincronización de Resonancia (El "Cómo") ---
                # El nodo actual extrae la identidad del vecino
                neighbor_identity = neighbor.get_identity_vector()
                
                # Entrenamos la red interna del nodo para que RECONOZCA al vecino.
                # Usamos la fuerza de la asociación (strength) como el target de afinidad.
                # Si están muy cerca (dist_factor 1.0), la afinidad buscada es alta.
                node.train_node_resonance(
                    sample_vector=neighbor_identity, 
                    target_affinity=dist_factor, 
                    learning_rate=learning_rate
                )
                
        print(f"Entrenamiento completado: {len(words)} tokens procesados con sincronización neuronal.")


    def send_signal(self, source_coords, target_coords, signal_vector):
        target_node = self._node_storage.get(target_coords)
        if not target_node:
            return None

        # 1. Asegurar dimensiones 1000x1
        signal_vector = np.array(signal_vector).reshape(1000, 1)
        identity_vector = target_node.get_identity_vector().reshape(1000, 1)

        # 2. Cálculo de Fricción (Coseno de Similitud invertido)
        norm_s = np.linalg.norm(signal_vector)
        norm_i = np.linalg.norm(identity_vector)
        
        if norm_s == 0 or norm_i == 0:
            friction = 1.0
        else:
            # Similitud de 1.0 (alineados) a 0.0 (ortogonales)
            dot_product = np.dot(signal_vector.flatten(), identity_vector.flatten())
            cos_sim = dot_product / (norm_s * norm_i)
            friction = 1.0 - max(0, cos_sim)

        # 3. Aplicar Madurez y Umbral
        effective_friction = friction * target_node.maturity

        if effective_friction > self.friction_threshold:
            # La señal es bloqueada por el Kernel
            return None 

        # 4. Atenuación y Activación
        attenuated_signal = signal_vector * (1.0 - effective_friction)
        return target_node.activate(attenuated_signal)


    def propagate(self, start_coords: Tuple[int, ...], initial_signal: np.ndarray, max_hops: int = 5):
        """
        Propaga una señal aplicando decaimiento, fricción y resonancia ontológica.
        """
        # (coords, vector, energía_actual, nivel_actual)
        queue = [(start_coords, initial_signal, 1.0, 0)]  
        results = {}
        visited = {} # Usamos dict para guardar la mejor energía alcanzada

        while queue:
            current_coords, current_signal, energy, hop = queue.pop(0)
            
            # 1. Condición de parada: Umbral de extinción o profundidad máxima
            if energy < self.extinction_threshold or hop >= max_hops: 
                continue
                
            node = self._node_storage.get(current_coords)
            if not node:
                continue

            # Evitar ciclos, pero permitir caminos más eficientes
            if current_coords in visited and visited[current_coords] >= energy:
                continue
            
            visited[current_coords] = energy
            results[node.name] = max(results.get(node.name, 0), energy)
            
            # 2. Obtener señales de salida del nodo actual
            output_signal = node.activate(current_signal)
            
            # 3. Explorar punteros (relaciones)
            top_ptrs = node.get_top_pointers(limit=10) # Aumentamos para no perder ramas
            
            for target_coords, strength in top_ptrs:
                target_node = self._node_storage.get(target_coords)
                if not target_node: continue
                
                # --- RESONANCIA ONTOLÓGICA ---
                # El nodo destino juzga la señal entrante
                validacion_vector = target_node.activate(current_signal)
                
                activaciones = np.abs(target_node.activate(current_signal))
                activaciones = np.atleast_1d(activaciones) # Forzamos que sea array incluso si es un escalar
                n_neuronas = activaciones.size

                # 2. LÓGICA DE RESONANCIA ROBUSTA (Sin np.partition para evitar crashes)
                if n_neuronas > 10:
                    # Si la red es grande, usamos el promedio del 10% superior manualmente
                    k = max(1, int(n_neuronas * 0.1))
                    # Usamos sort simple que es más seguro para arrays pequeños/medianos
                    top_k = np.sort(activaciones)[-k:]
                    res_factor_top = np.mean(top_k)
                else:
                    # Si la red es pequeña (como tu caso de 1 sola neurona), promedio directo
                    res_factor_top = np.mean(activaciones)

                # 3. EL FILTRO DE VERDAD (Cuchillo Ontológico)
                # Si el nodo 'Oscuridad' fue entrenado a 0.0, aquí se cortará la señal
                threshold = 0.1  # más permisivo al inicio
                if res_factor_top < threshold:
                    res_factor = 0.01   # atenuar en vez de casi-extinguir
                else:
                    res_factor = 1.0

                # 4. Cálculo de energía final
                next_energy = min(1.0, energy * strength * res_factor)
                
                # Fricción vectorial (pérdida de integridad del mensaje)
                next_signal = self.send_signal(current_coords, target_coords, output_signal)
                
                if next_signal is not None and next_energy >= self.extinction_threshold:
                    queue.append((target_coords, next_signal, next_energy, hop + 1))
            
        # Ordenamos resultados por energía para la tabla final
        return sorted(results.items(), key=lambda x: x[1], reverse=True)

    def get_definitions_by_index(self, index: Tuple[int, ...]):
        """
        Recupera la lista de conceptos definitorios asociados 
        a una coordenada específica en la Matrix.
        """
        # 1. Intentamos obtener el nodo en esa posición
        concept_definitions = self.get(index)
        
        if concept_definitions:
            # 2. Retornamos la definición inmutable guardada en el nodo
            # Usamos list() para asegurar que sea una copia trabajable
            return list(concept_definitions)
        
        # 3. Si no hay nada en esa coordenada, retornamos una lista vacía
        # o podrías lanzar una excepción según prefieras.
        print(f"Aviso: La coordenada {index} está vacía.")
        return []


    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def nnz(self) -> int:
        """Number of stored cells."""
        return len(self._matrix_storage)

    @property
    def density(self) -> float:
        """Fraction of cells that are occupied."""
        total = 1
        for d in self.shape:
            total *= d
        return self.nnz / total

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __getitem__(self, index: Tuple[int, ...]) -> Value:
        return self.get(index)

    def __setitem__(self, index: Tuple[int, ...], value: Value, concept: str):
        self.set(index, value, concept)

    def __repr__(self) -> str:
        return (
            f"ConceptMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"density={self.density:.2e})"
        )

