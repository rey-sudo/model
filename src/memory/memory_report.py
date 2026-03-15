import sys
from scipy.sparse import lil_matrix, csr_matrix

N_LABEL    = 64               # bits para codificar el ID entero
MAX_ITER   = 50               # iteraciones máximas de convergencia

def memory_report(self) -> dict:
    """
    Reporte exacto de consumo de RAM de TODAS las variables de la instancia BAM.

    Cubre:
      - Matriz W en formato LIL (aprendizaje) y CSR (recall)
      - Lista self.patterns  (vectores numpy por patrón)
      - Diccionario self.label_map
      - Escalares de configuración
      - Total real de la instancia
    """

    def _sparse_lil_bytes(m: lil_matrix) -> dict:
        """Mide lil_matrix: dos arrays de listas de objetos Python."""
        data_bytes  = sum(sys.getsizeof(row) + sum(sys.getsizeof(v) for v in row)
                          for row in m.data)
        rows_bytes  = sum(sys.getsizeof(row) + sum(sys.getsizeof(i) for i in row)
                          for row in m.rows)
        struct_bytes = sys.getsizeof(m) + sys.getsizeof(m.data) + sys.getsizeof(m.rows)
        return {
            'data_bytes':  data_bytes,
            'rows_bytes':  rows_bytes,
            'struct_bytes': struct_bytes,
            'total_bytes': data_bytes + rows_bytes + struct_bytes,
        }

    def _sparse_csr_bytes(m: csr_matrix) -> dict:
        """Mide csr_matrix: tres arrays numpy (data, indices, indptr)."""
        return {
            'data_bytes':    int(m.data.nbytes),
            'indices_bytes': int(m.indices.nbytes),
            'indptr_bytes':  int(m.indptr.nbytes),
            'struct_bytes':  sys.getsizeof(m),
            'total_bytes':   int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)
                             + sys.getsizeof(m),
        }

    def _patterns_bytes(patterns: list) -> dict:
        """Mide self.patterns: lista de dicts con arrays numpy."""
        per_pattern = []
        for p in patterns:
            x_bytes      = int(p['x'].nbytes)
            x_diff_bytes = int(p['x_diff'].nbytes)
            y_bytes      = int(p['y'].nbytes)
            meta_bytes   = (sys.getsizeof(p)
                            + sys.getsizeof(p['label'])
                            + sys.getsizeof(p['id'])
                            + sys.getsizeof(p['n_white_new']))
            total = x_bytes + x_diff_bytes + y_bytes + meta_bytes
            per_pattern.append({
                'id':           p['id'],
                'label':        p['label'],
                'x_bytes':      x_bytes,
                'x_diff_bytes': x_diff_bytes,
                'y_bytes':      y_bytes,
                'meta_bytes':   meta_bytes,
                'total_bytes':  total,
            })
        list_struct  = sys.getsizeof(patterns)
        grand_total  = list_struct + sum(pp['total_bytes'] for pp in per_pattern)
        return {
            'n_patterns':      len(patterns),
            'list_struct_bytes': list_struct,
            'per_pattern':     per_pattern,
            'total_bytes':     grand_total,
        }

    def _label_map_bytes(lm: dict) -> dict:
        """Mide self.label_map: overhead del dict + claves int + valores str."""
        keys_bytes   = sum(sys.getsizeof(k) for k in lm)
        values_bytes = sum(sys.getsizeof(v) for v in lm.values())
        struct_bytes = sys.getsizeof(lm)
        return {
            'n_entries':     len(lm),
            'keys_bytes':    keys_bytes,
            'values_bytes':  values_bytes,
            'struct_bytes':  struct_bytes,
            'total_bytes':   keys_bytes + values_bytes + struct_bytes,
        }

    def _mb(b: int) -> float:
        return round(b / 1024 ** 2, 4)

    # ── Mediciones ──────────────────────────────────────────────
    W_csr = self.W   # fuerza conversión si _dirty

    lil_info     = _sparse_lil_bytes(self._W_lil)
    csr_info     = _sparse_csr_bytes(W_csr)
    patterns_info = _patterns_bytes(self.patterns)
    lmap_info    = _label_map_bytes(self.label_map)

    # Hipotético denso float32
    dense_bytes  = self.N_PIXELS * N_LABEL * 4

    # Escalares de configuración
    scalars_bytes = (sys.getsizeof(self.N_PIXELS)
                     + sys.getsizeof(self.IMG_WIDTH)
                     + sys.getsizeof(self.IMG_HEIGHT)
                     + sys.getsizeof(self._dirty))

    total_real_bytes = (lil_info['total_bytes']
                        + csr_info['total_bytes']
                        + patterns_info['total_bytes']
                        + lmap_info['total_bytes']
                        + scalars_bytes)

    # ── Reporte final ────────────────────────────────────────────
    report = {
        'config': {
            'N_PIXELS':    self.N_PIXELS,
            'N_LABEL':     N_LABEL,
            'IMG_WIDTH':   self.IMG_WIDTH,
            'IMG_HEIGHT':  self.IMG_HEIGHT,
            'n_patterns':  len(self.patterns),
        },
        'W_lil': {
            #**lil_info,
            'total_MB': _mb(lil_info['total_bytes']),
            'note': 'formato escritura (aprendizaje incremental)',
        },
        'W_csr': {
            #**csr_info,
            'nnz':          int(W_csr.nnz),
            'total_elements': self.N_PIXELS * N_LABEL,
            'density_pct':  round(100 * W_csr.nnz / (self.N_PIXELS * N_LABEL), 4),
            'total_MB':     _mb(csr_info['total_bytes']),
            'note': 'formato lectura (recall)',
        },
        'W_dense_hypothetical': {
            'total_bytes': dense_bytes,
            'total_MB':    _mb(dense_bytes),
            'note': 'float32 denso, nunca se materializa',
        },
        'W_compression': {
            'saving_bytes':     dense_bytes - csr_info['total_bytes'],
            'saving_MB':        _mb(dense_bytes - csr_info['total_bytes']),
            'compression_factor': round(dense_bytes / (csr_info['total_bytes'] + 1e-9), 2),
        },
        'patterns': {
            #**patterns_info,
            'total_MB': _mb(patterns_info['total_bytes']),
        },
        'label_map': {
            #**lmap_info,
            'total_MB': _mb(lmap_info['total_bytes']),
        },
        'scalars_bytes': scalars_bytes,
        'total': {
            'total_bytes': total_real_bytes,
            'total_MB':    _mb(total_real_bytes),
            'breakdown_MB': {
                'W_lil':      _mb(lil_info['total_bytes']),
                'W_csr':      _mb(csr_info['total_bytes']),
                'patterns':   _mb(patterns_info['total_bytes']),
                'label_map':  _mb(lmap_info['total_bytes']),
                'scalars':    _mb(scalars_bytes),
            },
        },
    }

    return report