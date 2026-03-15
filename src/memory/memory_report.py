import sys
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import issparse

N_LABEL    = 64               # bits para codificar el ID entero
MAX_ITER   = 50               # iteraciones máximas de convergencia

def memory_report(bam) -> dict[str, float]:
    """Retorna el uso de memoria en MB de cada variable interna de la BAM."""

    def _mb(obj) -> float:
        if isinstance(obj, lil_matrix):
            return (sys.getsizeof(obj.data) + sys.getsizeof(obj.rows)) / 1024**2
        if issparse(obj):
            return (obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes) / 1024**2
        if hasattr(obj, "nbytes"):
            return obj.nbytes / 1024**2
        return sys.getsizeof(obj) / 1024**2

    def _mb_dict_deep(d: dict) -> float:
        total = sys.getsizeof(d)
        for k, v in d.items():
            total += sys.getsizeof(k) + sys.getsizeof(v)
        return total / 1024**2

    def _mb_pattern(p: dict) -> float:
        arrays = _mb(p["x"]) + _mb(p["x_diff"]) + _mb(p["y"])
        scalars = (sys.getsizeof(p["id"]) + sys.getsizeof(p["label"])
                + sys.getsizeof(p["n_white_new"]) + sys.getsizeof(p)) / 1024**2
        return arrays + scalars


    W = bam.W  # fuerza conversión a CSR si está dirty

    dims = {
        "IMG_WIDTH":  f"{bam.IMG_WIDTH}px",
        "IMG_HEIGHT": f"{bam.IMG_HEIGHT}px",
        "N_PIXELS":   f"{bam.N_PIXELS}px",
        "TOTAL_SIGNS": bam.total_signs
    }

    entries = {
        "W (CSR)":   _mb(W),
        "W (LIL)":   _mb(bam._W_lil),
        "patterns":  sum(_mb_pattern(p) for p in bam.patterns),
        "label_map": _mb_dict_deep(bam.label_map),
        "_dirty":    sys.getsizeof(bam._dirty) / 1024**2,
    }
    entries["TOTAL"] = sum(entries.values())

    return {**dims, **entries}