import numpy as np
import sparse
import plotly.graph_objects as go
from typing import Any, List, Tuple, Union
from src.node import ConceptNode
from src.utils.hashing import coordinates_from_index, posiciones_en_abecedario

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


    def add_concept(self, concept: str, definition: list[str]) -> tuple[int, int, int]:
        concepto_raw_index = posiciones_en_abecedario(concept)
        concept_index = coordinates_from_index(concepto_raw_index)

        
        concept_definition = []

        for c in definition:
            c_array = posiciones_en_abecedario(c)
            coo = coordinates_from_index(c_array)
            concept_definition.append(coo)        
        
        self.set(concept_index, concept, concept_definition)
        
        return concept_index


    def train(self, text: str):
        words = text.lower().split()
        
        # En lugar de crear, RECUPERAMOS
        active_sequence = []
        for word in words:
            coords = self.get_coords(word) # SHA-256 Inmutable
            node = self._node_storage.get(coords)
            
            if node:
                active_sequence.append(node)
            else:
                # Si aparece una palabra que NO estaba en el diccionario inglés
                # (un neologismo o nombre propio), aquí es donde nace
                self.add_node(coords, seed_structure=word)
                active_sequence.append(self._node_storage[coords])

        # Fortalecimiento de la Red Existente
        for i, node in enumerate(active_sequence):
            # Conectamos con el contexto inmediato
            for offset in [-1, 1]:
                if 0 <= i + offset < len(active_sequence):
                    neighbor = active_sequence[i + offset]
                    # Aquí no definimos qué ES, sino cómo se USA
                    node.add_pointer(neighbor.index, strength=0.05)


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

    def __setitem__(self, index: Tuple[int, ...], value: Value):
        self.set(index, value)

    def __repr__(self) -> str:
        return (
            f"ConceptMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"density={self.density:.2e})"
        )

