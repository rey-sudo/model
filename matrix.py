import numpy as np
import sparse
import plotly.graph_objects as go
from typing import Any, List, Tuple, Union

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
        self._store: dict[Tuple[int, ...], Value] = {}

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

    def set(self, index: Tuple[int, ...], value: Value):
        """
        Store any value at index.
          cm.set((x,y,z), 3.14)                    # scalar
          cm.set((x,y,z), [(4,3,3), (8,9,0)])      # ordered coord list
          cm.set((x,y,z), "label")                  # string
          cm.set((x,y,z), None)                     # removes the cell
        """
        self._validate(index)
        if value is None:
            self._store.pop(index, None)
        elif self._is_coord_list(value):
            self._store[index] = [tuple(c) for c in value]  # preserve order
        else:
            self._store[index] = value

    def get(self, index: Tuple[int, ...]) -> Value:
        """Return stored value, or _EMPTY sentinel if the cell is empty."""
        self._validate(index)
        return self._store.get(index, _EMPTY)

    def is_empty(self, index: Tuple[int, ...]) -> bool:
        """True if the cell has no stored value."""
        return self.get(index) is _EMPTY

    def delete(self, index: Tuple[int, ...]):
        """Remove a cell (reset to empty)."""
        self._validate(index)
        self._store.pop(index, None)

    # ------------------------------------------------------------------
    # Sparse export (scalars only)
    # ------------------------------------------------------------------

    def to_coo(self) -> sparse.COO:
        """
        Export numeric cells to sparse.COO.
        Cells containing coord lists or strings are skipped.
        """
        scalar_items = {
            k: v for k, v in self._store.items()
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

    # ------------------------------------------------------------------
    # 3D Visualization — Plotly
    # ------------------------------------------------------------------

    def plot(self, title: str = "ConceptMatrix 3D", save_html: str = None):
        """
        Interactive 3D scatter of all stored cells.
        - Scalar cells: color + size encode the value.
        - Coord-list cells: rendered as connected line segments.
        - Hover shows coordinates and stored value.
        Requires a 3-D matrix.
        """
        if len(self.shape) != 3:
            raise NotImplementedError("plot() only supports 3-D matrices.")
        if not self._store:
            print("Nothing to plot: matrix is empty.")
            return

        scalar_keys, scalar_vals = [], []
        coord_list_items = []

        for k, v in self._store.items():
            if isinstance(v, (int, float)):
                scalar_keys.append(k)
                scalar_vals.append(v)
            elif isinstance(v, list):
                coord_list_items.append((k, v))

        traces = []

        # --- Scalar points ---
        if scalar_keys:
            xs = [c[0] for c in scalar_keys]
            ys = [c[1] for c in scalar_keys]
            zs = [c[2] for c in scalar_keys]
            vmin, vmax = min(scalar_vals), max(scalar_vals)
            span = (vmax - vmin) or 1
            sizes = [8 + 16 * (v - vmin) / span for v in scalar_vals]
            hover = [f"x={x}, y={y}, z={z}<br>value={v:.4g}"
                     for x, y, z, v in zip(xs, ys, zs, scalar_vals)]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                name="scalars",
                marker=dict(
                    size=sizes, color=scalar_vals,
                    colorscale="Viridis",
                    colorbar=dict(title="Value"),
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))

        # --- Coord-list cells: one colored trace per target coord ---
        PALETTE = ["cyan", "magenta", "lime", "orange", "red", "yellow", "white"]

        for origin, targets in coord_list_items:
            ox, oy, oz = origin
            for i, t in enumerate(targets):
                color = PALETTE[i % len(PALETTE)]
                traces.append(go.Scatter3d(
                    x=[ox, t[0]], y=[oy, t[1]], z=[oz, t[2]],
                    mode="lines+markers",
                    name=f"link@{origin}→{t}",
                    line=dict(color=color, width=2),
                    marker=dict(size=[4, 0], color=color, symbol=["circle", "circle"]),
                    text=[f"origin={origin}", f"target={t}"],
                    hovertemplate="%{text}<extra></extra>",
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(title="X", range=[0, self.shape[0]]),
                yaxis=dict(title="Y", range=[0, self.shape[1]]),
                zaxis=dict(title="Z", range=[0, self.shape[2]]),
                bgcolor="rgb(10,10,20)",
            ),
            paper_bgcolor="rgb(20,20,30)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if save_html:
            fig.write_html(save_html)
            print(f"Saved to {save_html}")

        fig.show()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def nnz(self) -> int:
        """Number of stored cells."""
        return len(self._store)

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

