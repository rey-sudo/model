"""
Memoria Asociativa Bidireccional (BAM - Bidirectional Associative Memory)
=========================================================================
Implementación completa de una BAM que asocia:
  - Imagen de entrada: n×n píxeles
  - Label de salida

La BAM puede:
  1. APRENDER  : almacenar el par (imagen, label)
  2. RECORDAR  : dado el label → reconstruir la imagen
  3. RECONOCER : dada la imagen → recuperar el label

Optimización de memoria (v2):
  - Capa de imagen usa codificación BINARIA {0, 1} en lugar de bipolar {-1, +1}
  - Negro = 0 → su fila en W jamás se actualiza → W es naturalmente dispersa
  - W se almacena como scipy.sparse.csr_matrix (~10x menos RAM para imágenes
    con ~90% de fondo negro)
"""

from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


current_path = Path.cwd()

# ══════════════════════════════════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════════════════════════════════

N_LABEL    = 64               # bits para codificar el ID entero
MAX_ITER   = 50               # iteraciones máximas de convergencia

# ══════════════════════════════════════════════════════════════════════════════
#  Funciones de codificación / decodificación
# ══════════════════════════════════════════════════════════════════════════════

def image_to_binary(img_array: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen (n×n) en un vector BINARIO de longitud N_PIXELS.

    Pasos:
      1. Escala de grises
      2. Umbral en 128  → {0, 1}

    Negro → 0  (no contribuye a W, permite dispersión)
    Blanco → 1

    Retorna float32 para compatibilidad con operaciones dispersas.
    """
    if img_array.ndim == 3:                     # RGB → escala de grises
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(float)

    if gray.max() <= 1.0:                       # normalizar a [0, 255]
        gray = gray * 255.0

    binary = (gray >= 128).astype(np.float32)   # umbral → {0.0, 1.0}
    return binary.flatten()                      


def binary_to_image(height: int, width: int, vec: np.ndarray) -> np.ndarray:
    binary = (vec > 0).reshape(height, width).astype(np.float32)
    return (binary * 255).astype(np.uint8)

def id_to_bipolar(label_id: int) -> np.ndarray:
    """
    Codifica un entero (ID del patrón) en vector bipolar de N_LABEL bits.
    """
    bits = []
    for b in range(N_LABEL - 1, -1, -1):        # MSB primero
        bits.append(1 if (label_id >> b) & 1 else -1)
    return np.array(bits, dtype=np.float32)       # (64,)


def bipolar_to_id(vec: np.ndarray) -> int:
    """
    Decodifica un vector bipolar de N_LABEL bits → entero ID.
    """
    binary = ((np.sign(vec) + 1) / 2).astype(int)  # {-1,+1} → {0,1}
    return int(''.join(binary.astype(str)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Clase BAM
# ══════════════════════════════════════════════════════════════════════════════

class BAM:
    def __init__(self, total_signs: int, sign_size_px: int):
        self.IMG_WIDTH  = sign_size_px * total_signs
        self.IMG_HEIGHT = sign_size_px
        self.N_PIXELS   = self.IMG_WIDTH * self.IMG_HEIGHT
        
        self.total_signs = total_signs
        
        
        # lil_matrix: inserción O(1) por fila, ideal para aprendizaje incremental
        self._W_lil: lil_matrix = lil_matrix((self.N_PIXELS, N_LABEL), dtype=np.float32)
        # csr_matrix: multiplicación rápida, se genera al primer recall
        self._W_csr: csr_matrix | None = None
        self._dirty: bool = True              # True = _W_lil tiene cambios sin congelar

        self.patterns: list[dict] = []        # memoria episódica
        
        self.label_map: dict[int, str] = {}
        
        print(
            f"✅ BAM inicializada  |  "
            f"Capa A: {self.N_PIXELS} neuronas (binaria, dispersa)  |  "
            f"Capa B: {N_LABEL} neuronas (bipolar)"
        )

    # ------------------------------------------------------------------
    #  Propiedad W (acceso unificado a la matriz congelada)
    # ------------------------------------------------------------------
    @property
    def W(self) -> csr_matrix:
        """Devuelve W en formato CSR, reconvirtiéndola solo si hubo aprendizaje nuevo."""
        if self._dirty:
            self._W_csr = self._W_lil.tocsr()
            self._dirty = False
        return self._W_csr
        
    def learn_incremental(self, image: np.ndarray, label_str: str) -> None:
        label_id = len(self.patterns)              # ID = índice correlativo
        
        self.label_map[label_id] = label_str       # guardar string en diccionario

        x_new = image_to_binary(image)

        if len(self.patterns) > 0:
            x_acum = np.zeros(self.N_PIXELS, dtype=np.float32)
            for p in self.patterns:
                x_acum = np.maximum(x_acum, p['x'])
            x_diff = x_new * (1 - x_acum)
        else:
            x_diff = x_new

        if x_diff.sum() == 0:
            print(f"⚠️  '{label_str}' no aporta píxeles nuevos — patrón ignorado")
            return

        y = id_to_bipolar(label_id)                # ← ID entero, no string
        white_pixels = np.nonzero(x_diff)[0]
        self._W_lil[white_pixels, :] += y[np.newaxis, :]
        self._dirty = True

        self.patterns.append({
            'x':      x_new,
            'x_diff': x_diff,
            'y':      y,
            'id':     label_id,
            'label':  label_str,
            'n_white_new': int(x_diff.sum()),
        })

        print(
            f"[{label_id}]"
            f"Píxeles nuevos: {int(x_diff.sum())}  |  "
            f"Acumulados: {int(x_new.sum())}"
        )

    def flush(self):
            _ = self.W 
            
            del self._W_lil

            for p in self.patterns:
                del p['x_diff']
                del p['y']
                del p['n_white_new']
    # ------------------------------------------------------------------
    #  Recuperación: imagen → label
    # ------------------------------------------------------------------
    def recall_label(self, image: np.ndarray) -> tuple[str, int, np.ndarray]:
        x = image_to_binary(image)

        y = np.sign(self.W.T @ x)
        y[y == 0] = 1

        label_id  = bipolar_to_id(y)                                    # entero
        label_str = self.label_map.get(label_id, f"<ID {label_id} desconocido>")
        return label_str, label_id, y
    
    def recall_ranking(self, image: np.ndarray) -> list[dict]:
        x = image_to_binary(image)

        ranking = []
        for p in self.patterns:
            score = self.similarity(x, p['x'])
            ranking.append({
                'id':    p['id'],
                'label': p['label'],
                'score': score,
                'votos': int(np.dot(x, p['x'])),
            })

        ranking.sort(key=lambda d: d['score'], reverse=True)
        
        return ranking

    # ------------------------------------------------------------------
    #  Utilidades
    # ------------------------------------------------------------------

    def similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Similitud coseno entre dos vectores."""
        norm = np.linalg.norm(x1) * np.linalg.norm(x2)
        return float(np.dot(x1, x2) / (norm + 1e-9))

    def memory_report(self) -> dict:
        """
        Devuelve un resumen del uso de memoria de W.

        Compara el tamaño real (CSR) contra el hipotético denso (float32).
        """
        W = self.W
        nnz       = W.nnz
        total     = self.N_PIXELS * N_LABEL
        dense_mb  = total * 4 / 1024**2          # float32 = 4 bytes
        sparse_mb = (
            W.data.nbytes + W.indices.nbytes + W.indptr.nbytes
        ) / 1024**2

        report = {
            'elementos_no_cero':  nnz,
            'elementos_totales':  total,
            'densidad_%':         round(100 * nnz / total, 2),
            'ram_densa_MB':       round(dense_mb, 2),
            'ram_sparse_MB':      round(sparse_mb, 2),
            'ahorro_MB':          round(dense_mb - sparse_mb, 2),
            'factor_compresion':  round(dense_mb / (sparse_mb + 1e-9), 1),
        }

        print("\n📊 Reporte de memoria de W:")
        for k, v in report.items():
            print(f"   {k:<25} {v}")
        return report





