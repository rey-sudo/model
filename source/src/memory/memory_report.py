import sys
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import issparse

N_LABEL    = 64               # bits para codificar el ID entero
MAX_ITER   = 50               # iteraciones máximas de convergencia

def memory_report(bam) -> dict:

    def _mb_lil(m: lil_matrix) -> float:
        nnz        = sum(len(row) for row in m.data)
        per_value  = sys.getsizeof(m.data[0][0]) if nnz > 0 else 28
        per_index  = 28
        row_shells = (sum(sys.getsizeof(r) for r in m.data) +
                      sum(sys.getsizeof(r) for r in m.rows))
        outer      = sys.getsizeof(m.data) + sys.getsizeof(m.rows)
        return (outer + row_shells + nnz * (per_value + per_index)) / 1024**2

    def _mb(obj) -> float:
        if isinstance(obj, lil_matrix):
            return _mb_lil(obj)
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
        arrays  = sum(_mb(p[k]) for k in ('x', 'x_diff', 'y') if k in p)
        scalars = (sys.getsizeof(p["id"]) + sys.getsizeof(p["label"]) +
                   sys.getsizeof(p.get("n_white_new", 0)) + sys.getsizeof(p)) / 1024**2
        return arrays + scalars

    # Resolución segura de W
    W = None
    if hasattr(bam, '_W_lil'):
        W = bam.W                                          # materializa CSR desde LIL
    elif hasattr(bam, '_W_csr') and bam._W_csr is not None:
        W = bam._W_csr

    dims = {
        "IMG_WIDTH":   f"{bam.IMG_WIDTH}px",
        "IMG_HEIGHT":  f"{bam.IMG_HEIGHT}px",
        "N_PIXELS":    f"{bam.N_PIXELS}px",
        "TOTAL_SIGNS": bam.total_signs,
    }

    entries = {
        "W (CSR)":       _mb(W)           if W is not None              else 0.0,
        "W (LIL)":       _mb(bam._W_lil)  if hasattr(bam, '_W_lil')    else 0.0,
        "patterns":      sum(_mb_pattern(p) for p in bam.patterns),
        "label_map":     _mb_dict_deep(bam.label_map),
        "_dirty":        sys.getsizeof(bam._dirty) / 1024**2,
        "patterns_list": sys.getsizeof(bam.patterns) / 1024**2,
    }
    entries["TOTAL"] = sum(entries.values())

    return {**dims, **entries}