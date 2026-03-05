import numpy as np
import sparse
from typing import Any, Tuple


class ConceptMatrix:
    """
    Sparse N-dimensional matrix backed by COO format.
    Only stores non-zero values; empty cells return 0.
    """

    def __init__(self, shape: Tuple[int, ...], dtype=float):
        self.shape = shape
        self.dtype = dtype
        # dict for O(1) lookup: coord_tuple → value
        self._store: dict[Tuple[int, ...], Any] = {}

    # ------------------------------------------------------------------
    # Validation (internal)
    # ------------------------------------------------------------------

    def _validate(self, index: Tuple[int, ...]):
        if len(index) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)}-D index, got {len(index)}-D.")
        for axis, (i, dim) in enumerate(zip(index, self.shape)):
            if not (0 <= i < dim):
                raise IndexError(f"Index {i} out of bounds for axis {axis} (size {dim}).")

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def set(self, index: Tuple[int, ...], value: Any):
        """Store a value. Removes the entry if value == 0."""
        self._validate(index)
        if value == 0:
            self._store.pop(index, None)
        else:
            self._store[index] = self.dtype(value)

    def get(self, index: Tuple[int, ...]) -> Any:
        """Return the value at index, or 0 if not stored."""
        self._validate(index)
        return self._store.get(index, self.dtype(0))

    def delete(self, index: Tuple[int, ...]):
        """Reset a cell to 0 (remove from storage)."""
        self._validate(index)
        self._store.pop(index, None)

    # ------------------------------------------------------------------
    # Sparse export
    # ------------------------------------------------------------------

    def to_coo(self) -> sparse.COO:
        """Export to a sparse.COO matrix."""
        if not self._store:
            return sparse.COO(
                coords=np.zeros((len(self.shape), 0), dtype=int),
                data=np.array([], dtype=self.dtype),
                shape=self.shape,
            )
        coords = np.array(list(self._store.keys()), dtype=int).T
        data = np.array(list(self._store.values()), dtype=self.dtype)
        return sparse.COO(coords=coords, data=data, shape=self.shape)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def nnz(self) -> int:
        """Number of stored (non-zero) elements."""
        return len(self._store)

    @property
    def density(self) -> float:
        """Fraction of cells that are non-zero."""
        total = 1
        for d in self.shape:
            total *= d
        return self.nnz / total

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __getitem__(self, index: Tuple[int, ...]) -> Any:
        return self.get(index)

    def __setitem__(self, index: Tuple[int, ...], value: Any):
        self.set(index, value)

    def __repr__(self) -> str:
        return (
            f"ConceptMatrix(shape={self.shape}, dtype={self.dtype.__name__}, "
            f"nnz={self.nnz}, density={self.density:.2e})"
        )


# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    cm = ConceptMatrix(shape=(1000, 1000, 1000))

    cm[0, 0, 0]       = 10.5
    cm[500, 500, 500] = 20.0
    cm[999, 999, 999] = 99.9

    print(cm)
    print(f"get (500,500,500) = {cm[500, 500, 500]}")
    print(f"get (1,1,1)       = {cm[1, 1, 1]}  <- empty")

    cm.delete((0, 0, 0))
    print(f"after delete      -> nnz={cm.nnz}")

    coo = cm.to_coo()
    print(f"COO matrix        -> {coo}")