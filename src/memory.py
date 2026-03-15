"""
Memoria Asociativa Bidireccional (BAM - Bidirectional Associative Memory)
=========================================================================
Implementación completa de una BAM que asocia:
  - Imagen de entrada: n×n píxeles
  - Label de salida:   string de una sola palabra (ej. "carro")

La BAM puede:
  1. APRENDER  : almacenar el par (imagen, label)
  2. RECORDAR  : dado el label → reconstruir la imagen
  3. RECONOCER : dada la imagen (o versión ruidosa) → recuperar el label

Optimización de memoria (v2):
  - Capa de imagen usa codificación BINARIA {0, 1} en lugar de bipolar {-1, +1}
  - Negro = 0 → su fila en W jamás se actualiza → W es naturalmente dispersa
  - W se almacena como scipy.sparse.csr_matrix (~10x menos RAM para imágenes
    con ~90% de fondo negro)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import lil_matrix, csr_matrix
import os
import sys
import time
import tracemalloc
import psutil

current_path = Path.cwd()

# ══════════════════════════════════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE   = 90               # píxeles de cada lado  (n × n)
N_PIXELS   = IMG_SIZE ** 2    # neuronas en la capa de imagen  (8100)
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
    return binary.flatten()                      # (8100,)


def binary_to_image(vec: np.ndarray) -> np.ndarray:
    """
    Reconstruye la imagen (IMG_SIZE × IMG_SIZE) desde un vector binario {0, 1}.
    Los valores negativos o cero → negro; positivos → blanco.
    """
    binary = (vec > 0).reshape(IMG_SIZE, IMG_SIZE).astype(np.float32)
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
    """
    Memoria Asociativa Bidireccional con matriz de pesos DISPERSA.

    Arquitectura:
        Capa A  ←──────────────────────────→  Capa B
       (imagen)   W  (n×m)  /  W.T (m×n)    (label)
        n = 8100                              m = 160

    Codificación:
        Capa A: binaria {0, 1}   ← NEGRO = 0, no actualiza W
        Capa B: bipolar {-1, +1} ← mantiene dinámica clásica de BAM

    Aprendizaje (Hebb sobre píxeles blancos):
        Para cada píxel i donde x[i] = 1:
            W[i, :] += y             (equivale a outer(x, y) sin las filas cero)

    Almacenamiento:
        W se construye como lil_matrix (eficiente para escritura incremental)
        y se congela como csr_matrix antes del primer recall (eficiente para
        multiplicación matriz-vector).

    Recuperación:
        y_new = sign(W.T @ x)      imagen → label   (CSR @ dense)
        x_new = sign(W   @ y)      label  → imagen  (CSR.T @ dense)
    """

    def __init__(self):
        # lil_matrix: inserción O(1) por fila, ideal para aprendizaje incremental
        self._W_lil: lil_matrix = lil_matrix((N_PIXELS, N_LABEL), dtype=np.float32)
        # csr_matrix: multiplicación rápida, se genera al primer recall
        self._W_csr: csr_matrix | None = None
        self._dirty: bool = True              # True = _W_lil tiene cambios sin congelar

        self.patterns: list[dict] = []        # memoria episódica
        
        self.label_map: dict[int, str] = {}
        
        print(
            f"✅ BAM inicializada  |  "
            f"Capa A: {N_PIXELS} neuronas (binaria, dispersa)  |  "
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

    # ------------------------------------------------------------------
    #  Aprendizaje
    # ------------------------------------------------------------------
    def learn(self, image: np.ndarray, label: str) -> None:
        """
        Almacena un par (imagen, label) en la memoria.

        image  : array NumPy n×n (uint8 o float)
        label  : string de una sola palabra

        Solo las filas correspondientes a píxeles BLANCOS (x[i] = 1)
        se actualizan en W, manteniendo su dispersión natural.
        """
        x = image_to_binary(image)        # {0, 1} — float32
        y = label_to_bipolar(label)       # {-1, +1} — float32

        # Índices de píxeles blancos (x[i] = 1)
        white_pixels = np.nonzero(x)[0]   # shape: (n_blancos,)

        # Actualizar solo las filas de píxeles blancos
        # Equivale a outer(x, y) pero omite las filas donde x[i]=0
        self._W_lil[white_pixels, :] += y[np.newaxis, :]   # broadcasting (n_blancos, 160)

        self._dirty = True   # invalidar caché CSR

        self.patterns.append({
            'x': x,
            'y': y,
            'image': image.copy(),
            'label': label,
            'n_white': len(white_pixels),
            'sparsity': 1.0 - len(white_pixels) / N_PIXELS,
        })

        print(
            f"📚 Patrón aprendido: '{label}'  |  "
            f"Píxeles blancos: {len(white_pixels)}/{N_PIXELS} "
            f"({100*len(white_pixels)/N_PIXELS:.1f}%)  |  "
            f"Patrones totales: {len(self.patterns)}"
        )
        
    def learn_incremental(self, image: np.ndarray, label_str: str) -> None:
        label_id = len(self.patterns)              # ID = índice correlativo
        
        self.label_map[label_id] = label_str       # guardar string en diccionario

        x_new = image_to_binary(image)

        if len(self.patterns) > 0:
            x_acum = np.zeros(N_PIXELS, dtype=np.float32)
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
            'image':  image.copy(),
            'n_white_new': int(x_diff.sum()),
        })

        print(
            f"[{label_id}]  '{label_str}'  |  "
            f"Píxeles nuevos: {int(x_diff.sum())}  |  "
            f"Acumulados: {int(x_new.sum())}"
        )

    # ------------------------------------------------------------------
    #  Recuperación: imagen → label
    # ------------------------------------------------------------------
    def recall_label(self, image: np.ndarray, noisy: bool = False,
                    noise_level: float = 0.0) -> tuple[str, int, np.ndarray]:
        x = image_to_binary(image)

        if noisy and noise_level > 0:
            x = self._add_noise(x, noise_level)

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
                'score': round(score, 4),
                'votos': int(np.dot(x, p['x'])),
            })

        ranking.sort(key=lambda d: d['score'], reverse=True)

        print(f"\n{'Rank':<6} {'ID':<5} {'Label':<45} {'Score':>8} {'Votos':>8}")
        print('─' * 74)
        for i, d in enumerate(ranking, 1):
            marker = ' ◄' if i == 1 else ''
            print(f"{i:<6} {d['id']:<5} {d['label']:<45} {d['score']:>8.4f} {d['votos']:>8}{marker}")

        return ranking

    # ------------------------------------------------------------------
    #  Recuperación: label → imagen
    # ------------------------------------------------------------------
    def recall_image(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Dado el label → reconstruye la imagen.
        Retorna (imagen_uint8 IMG_SIZE×IMG_SIZE, vector_binario).
        """
        y = label_to_bipolar(label)
        y, x = self._iterate_from_y(y)
        img_array = binary_to_image(x)
        return img_array, x

    # ------------------------------------------------------------------
    #  Dinámica de convergencia (iteraciones)
    # ------------------------------------------------------------------
    def _iterate_from_x(self, x: np.ndarray) -> np.ndarray:
        """Propaga x→y→x→… hasta convergencia. x es binario {0,1}."""
        W = self.W   # una sola llamada a la propiedad (evita re-freeze en bucle)
        for _ in range(MAX_ITER):
            y = np.sign(W.T @ x);  y[y == 0] = 1
            x_new = np.sign(W @ y); x_new[x_new == 0] = 0   # negro=0, blanco=1
            x_new = (x_new > 0).astype(np.float32)
            if np.array_equal(x_new, x):
                break
            x = x_new
        return x

    def _iterate_from_y(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propaga y→x→y→… hasta convergencia. y es bipolar {-1,+1}."""
        W = self.W
        for _ in range(MAX_ITER):
            x = (np.sign(W @ y) > 0).astype(np.float32)   # binario {0,1}
            y_new = np.sign(W.T @ x); y_new[y_new == 0] = 1
            if np.array_equal(y_new, y):
                break
            y = y_new
        return y, x

    # ------------------------------------------------------------------
    #  Utilidades
    # ------------------------------------------------------------------
    @staticmethod
    def _add_noise(vec: np.ndarray, level: float) -> np.ndarray:
        """
        Invierte aleatoriamente `level` fracción de bits binarios.
        0 → 1  y  1 → 0  con probabilidad `level`.
        """
        noisy = vec.copy()
        n_flip = int(len(vec) * level)
        idx = np.random.choice(len(vec), n_flip, replace=False)
        noisy[idx] = 1.0 - noisy[idx]   # flip binario (0↔1)
        return noisy

    def similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Similitud coseno entre dos vectores."""
        norm = np.linalg.norm(x1) * np.linalg.norm(x2)
        return float(np.dot(x1, x2) / (norm + 1e-9))

    def accuracy(self, image: np.ndarray, label: str) -> dict:
        """Evalúa qué tan bien se recuerda el par (imagen, label)."""
        recalled_label, y_rec = self.recall_label(image)
        recalled_img, x_rec   = self.recall_image(label)

        x_orig = image_to_binary(image)
        y_orig = label_to_bipolar(label)

        return {
            'label_correcto':   recalled_label == label,
            'label_recuperado': recalled_label,
            'similitud_imagen': self.similarity(x_orig, x_rec),
            'similitud_label':  self.similarity(y_orig, y_rec),
        }

    def memory_report(self) -> dict:
        """
        Devuelve un resumen del uso de memoria de W.

        Compara el tamaño real (CSR) contra el hipotético denso (float32).
        """
        W = self.W
        nnz       = W.nnz
        total     = N_PIXELS * N_LABEL
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

    # ------------------------------------------------------------------
    #  Monitor de recursos
    # ------------------------------------------------------------------
    def resource_usage(self, verbose: bool = True) -> dict:
        """
        Reporta el uso de recursos del proceso y de la BAM en sí misma.

        Métricas reportadas:
          · RAM del proceso (RSS) en MB
          · RAM del proceso (VMS) en MB
          · Uso de CPU (%) del proceso
          · Memoria ocupada por la matriz W (MB y bytes exactos)
          · Memoria de los patrones almacenados (MB)
          · Tamaño total del objeto BAM en bytes (sys.getsizeof)
          · Pico de RAM medido con tracemalloc durante una operación típica

        Parámetros
        ----------
        verbose : bool
            Si True imprime un reporte formateado en consola.

        Retorna
        -------
        dict con todas las métricas en sus unidades base.
        """
        proc = psutil.Process(os.getpid())

        # ── RAM y CPU del proceso ──────────────────────────────────────
        mem_info  = proc.memory_info()
        ram_rss   = mem_info.rss  / (1024 ** 2)   # MB
        ram_vms   = mem_info.vms  / (1024 ** 2)   # MB
        cpu_pct   = proc.cpu_percent(interval=0.1) # %

        # ── Tamaño de la matriz de pesos W ────────────────────────────
        w_bytes      = self.W.nbytes
        w_mb         = w_bytes / (1024 ** 2)
        w_shape      = self.W.shape
        w_dtype      = str(self.W.dtype)

        # ── Tamaño de los patrones almacenados ────────────────────────
        patterns_bytes = sum(
            p['x'].nbytes + p['y'].nbytes +
            (p['image'].nbytes if isinstance(p['image'], np.ndarray) else 0)
            for p in self.patterns
        )
        patterns_mb = patterns_bytes / (1024 ** 2)

        # ── Objeto BAM completo (Python overhead) ─────────────────────
        bam_sizeof = sys.getsizeof(self)

        # ── Pico de memoria durante recall (tracemalloc) ──────────────
        tracemalloc.start()
        _snapshot_before = tracemalloc.take_snapshot()

        # Operación representativa: un ciclo completo
        if self.patterns:
            _x = self.patterns[0]['x'].copy()
            _y_out = np.sign(self.W.T @ _x); _y_out[_y_out == 0] = 1
            _x_out = np.sign(self.W @ _y_out); _x_out[_x_out == 0] = 1

        _current, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_recall_kb = _peak / 1024

        # ── Capacidad teórica de la BAM (Hopfield bound) ──────────────
        # n_max ≈ 0.15 × min(N_A, N_B) para recuperación sin errores
        theoretical_capacity = int(0.15 * min(N_PIXELS, N_LABEL))

        stats = {
            # Proceso
            'ram_proceso_rss_mb'   : round(ram_rss, 3),
            'ram_proceso_vms_mb'   : round(ram_vms, 3),
            'cpu_proceso_pct'      : round(cpu_pct, 2),
            # Matriz W
            'matriz_W_shape'       : w_shape,
            'matriz_W_dtype'       : w_dtype,
            'matriz_W_bytes'       : w_bytes,
            'matriz_W_mb'          : round(w_mb, 4),
            # Patrones
            'n_patrones'           : len(self.patterns),
            'patrones_bytes'       : patterns_bytes,
            'patrones_mb'          : round(patterns_mb, 4),
            # Objeto Python
            'bam_sizeof_bytes'     : bam_sizeof,
            # Pico durante recall
            'pico_recall_kb'       : round(peak_recall_kb, 3),
            # Capacidad teórica
            'capacidad_teorica'    : theoretical_capacity,
            'carga_pct'            : round(len(self.patterns) / max(theoretical_capacity, 1) * 100, 1),
        }

        if verbose:
            sep  = '─' * 48
            sep2 = '═' * 48
            print(f"\n{sep2}")
            print(f"  📊  RECURSOS  —  Memoria Asociativa Bidireccional")
            print(f"{sep2}")

            print(f"\n  🖥️   Proceso  (PID {os.getpid()})")
            print(f"  {sep}")
            print(f"  RAM  RSS (física)   : {ram_rss:>10.3f} MB")
            print(f"  RAM  VMS (virtual)  : {ram_vms:>10.3f} MB")
            print(f"  CPU  uso actual     : {cpu_pct:>10.2f} %")

            print(f"\n  🧠  Matriz de pesos  W  {w_shape}")
            print(f"  {sep}")
            print(f"  Dtype               : {w_dtype}")
            print(f"  Bytes exactos       : {w_bytes:>10,}")
            print(f"  Megabytes           : {w_mb:>10.4f} MB")
            print(f"  Elementos (n×m)     : {w_shape[0] * w_shape[1]:>10,}")

            print(f"\n  💾  Patrones almacenados")
            print(f"  {sep}")
            print(f"  Cantidad            : {len(self.patterns):>10}")
            print(f"  Bytes totales       : {patterns_bytes:>10,}")
            print(f"  Megabytes           : {patterns_mb:>10.4f} MB")

            print(f"\n  ⚡  Operación de recall  (pico tracemalloc)")
            print(f"  {sep}")
            print(f"  Pico asignado       : {peak_recall_kb:>10.3f} KB")

            print(f"\n  📐  Capacidad teórica  (cota de Hopfield)")
            print(f"  {sep}")
            print(f"  Máx. patrones       : {theoretical_capacity:>10}")
            print(f"  Carga actual        : {stats['carga_pct']:>9.1f} %")
            bar_len  = 30
            filled   = int(bar_len * stats['carga_pct'] / 100)
            bar      = '█' * filled + '░' * (bar_len - filled)
            print(f"  [{bar}] {stats['carga_pct']:.1f}%")

            print(f"\n{sep2}\n")

        return stats

# ══════════════════════════════════════════════════════════════════════════════
#  Visualización completa
# ══════════════════════════════════════════════════════════════════════════════

def visualize_results(bam: BAM, image: np.ndarray, label: str,
                      noise_levels: list = None) -> None:
    """
    Panel de visualización con:
      · Imagen original y su representación binaria
      · Imagen reconstruida desde el label
      · Recuperación con ruido progresivo
      · Matriz de pesos (muestra, densificada para visualización)
      · Métricas de precisión
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

    fig = plt.figure(figsize=(18, 12), facecolor='#0f1117')
    fig.suptitle('Memoria Asociativa Bidireccional  (BAM)',
                 color='white', fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 6, figure=fig,
                           hspace=0.55, wspace=0.45,
                           left=0.04, right=0.97,
                           top=0.92, bottom=0.06)

    ax_style = dict(facecolor='#1a1d27')
    title_kw = dict(color='#a0cfff', fontsize=9, pad=6, fontweight='bold')
    label_kw = dict(color='#888', fontsize=7)

    # ── Fila 0: imagen original y representación binaria ───────────────
    ax0 = fig.add_subplot(gs[0, 0:2], **ax_style)
    ax0.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax0.set_title(f'Imagen Original\n"{label}"', **title_kw)
    ax0.axis('off')

    # Codificación binaria {0, 1} en lugar de bipolar {-1, +1}
    x_bin = image_to_binary(image).reshape(IMG_SIZE, IMG_SIZE)
    ax1 = fig.add_subplot(gs[0, 2:4], **ax_style)
    ax1.imshow(x_bin, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Codificación Binaria\nImagen  {0, 1}', **title_kw)
    ax1.axis('off')

    y_bip = label_to_bipolar(label)
    ax2 = fig.add_subplot(gs[0, 4:6], **ax_style)
    ax2.imshow(y_bip.reshape(1, -1), cmap='RdYlGn', vmin=-1, vmax=1,
               aspect='auto')
    ax2.set_title(f'Codificación Bipolar\nLabel  "{label}"  ({len(y_bip)} bits)',
                  **title_kw)
    ax2.set_xlabel('Índice de bit', **label_kw)
    ax2.set_yticks([])
    ax2.tick_params(colors='#666', labelsize=7)

    # ── Fila 1: reconstrucción label → imagen ──────────────────────────
    recalled_img, x_rec = bam.recall_image(label)
    ax3 = fig.add_subplot(gs[1, 0:2], **ax_style)
    ax3.imshow(recalled_img, cmap='gray', vmin=0, vmax=255)
    ax3.set_title(f'Reconstrucción\n"{label}" → Imagen', **title_kw)
    ax3.axis('off')

    # Diferencia pixel a pixel
    diff = np.abs(image.astype(int) - recalled_img.astype(int))
    ax4 = fig.add_subplot(gs[1, 2:4], **ax_style)
    im4 = ax4.imshow(diff, cmap='hot', vmin=0, vmax=255)
    ax4.set_title('Diferencia\n|Original - Reconstruida|', **title_kw)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046,
                 pad=0.04).ax.tick_params(colors='#888', labelsize=7)

    # Matriz de pesos — densificar solo la muestra 100×100 para imshow
    ax5 = fig.add_subplot(gs[1, 4:6], **ax_style)
    w_sample = bam.W[:100, :100].toarray()          # CSR → dense solo en la muestra
    w_abs_max = np.abs(bam.W.data).max() if bam.W.nnz > 0 else 1.0  # evita toarray() global
    im5 = ax5.imshow(w_sample, cmap='coolwarm',
                     vmin=-w_abs_max, vmax=w_abs_max)
    ax5.set_title('Matriz de Pesos W\n(muestra 100×100)', **title_kw)
    ax5.tick_params(colors='#666', labelsize=7)
    plt.colorbar(im5, ax=ax5, fraction=0.046,
                 pad=0.04).ax.tick_params(colors='#888', labelsize=7)

    # ── Fila 2: recuperación con ruido creciente ───────────────────────
    sims, labels_rec = [], []
    x_orig = image_to_binary(image)                 # binario {0, 1}

    for i, nl in enumerate(noise_levels):
        ax = fig.add_subplot(gs[2, i], **ax_style)
        if nl == 0.0:
            noisy_img = image
            x_in = x_orig.copy()
        else:
            x_in = bam._add_noise(x_orig, nl)      # flip binario 0↔1
            noisy_img = binary_to_image(x_in)       # {0,1} → uint8

        ax.imshow(noisy_img, cmap='gray', vmin=0, vmax=255)

        # Recuperar label desde imagen ruidosa
        x_iter = bam._iterate_from_x(x_in)
        y_out = np.sign(bam.W.T @ x_iter); y_out[y_out == 0] = 1
        lab_out = bipolar_to_label(y_out)

        sim = bam.similarity(x_orig, x_iter)
        sims.append(sim)
        labels_rec.append(lab_out)

        correct = '✓' if lab_out == label else '✗'
        color   = '#4ade80' if lab_out == label else '#f87171'
        ax.set_title(f'Ruido {nl*100:.0f}%\n"{lab_out}" {correct}',
                     color=color, fontsize=8, pad=5, fontweight='bold')
        ax.axis('off')
        ax.text(0.5, -0.06, f'sim={sim:.2f}',
                ha='center', va='top',
                transform=ax.transAxes, **label_kw)

    # ── Métricas resumen (texto) ───────────────────────────────────────
    metrics = bam.accuracy(image, label)

    # Reporte de dispersión de W
    nnz      = bam.W.nnz
    total    = N_PIXELS * N_LABEL
    densidad = 100 * nnz / total

    txt = (
        f"Métricas de precisión\n"
        f"──────────────────────────────\n"
        f"Label recuperado : '{metrics['label_recuperado']}'\n"
        f"Label correcto   : {'Sí ✓' if metrics['label_correcto'] else 'No ✗'}\n"
        f"Similitud imagen : {metrics['similitud_imagen']:.4f}\n"
        f"Similitud label  : {metrics['similitud_label']:.4f}\n"
        f"──────────────────────────────\n"
        f"Patrones en memoria : {len(bam.patterns)}\n"
        f"Neuronas imagen     : {N_PIXELS}\n"
        f"Neuronas label      : {N_LABEL}\n"
        f"Dim. matriz W       : {bam.W.shape}\n"
        f"Densidad W          : {nnz}/{total}  ({densidad:.1f}%)"
    )
    fig.text(0.5, 0.01, txt, ha='center', va='bottom',
             color='#c8d8f0', fontsize=8,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='#1a1d27', edgecolor='#3a4060'))

    plt.savefig(current_path / 'output/bam_results.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("🖼️  Visualización guardada → bam_results.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Demo principal
# ══════════════════════════════════════════════════════════════════════════════
def cargar_con_pillow(ruta):
    img = Image.open(ruta).convert('L')
    return np.array(img)

def main():
    print("=" * 60)
    print("  MEMORIA ASOCIATIVA BIDIRECCIONAL  (BAM)")
    print("=" * 60)

    # 1. Crear imagen de prueba
    image_path = current_path / "test.png"
    car_image = cargar_con_pillow(image_path)
    print(f"\n📷 Imagen creada: {car_image.shape}  dtype={car_image.dtype}")
    print(f"   Rango de valores: [{car_image.min()}, {car_image.max()}]")

    # 2. Inicializar BAM y aprender
    bam = BAM()
    label = "manzana"
    bam.learn(car_image, label)

    # 3. Prueba: label → imagen
    print(f"\n🔄 Prueba 1 — Reconstrucción  '{label}' → imagen")
    rec_img, _ = bam.recall_image(label)
    print(f"   Imagen reconstruida: shape={rec_img.shape}")

    # 4. Prueba: imagen → label
    print(f"\n🔄 Prueba 2 — Reconocimiento  imagen → label")
    rec_label, _ = bam.recall_label(car_image)
    print(f"   Label recuperado  : '{rec_label}'")
    print(f"   Correcto          : {rec_label == label}")

    # 5. Prueba con ruido
    print(f"\n🔄 Prueba 3 — Robustez al ruido")
    for noise in [0.05, 0.10, 0.20, 0.30]:
        rec_label_n, _ = bam.recall_label(car_image, noisy=True, noise_level=noise)
        print(f"   Ruido {noise*100:4.0f}%  → '{rec_label_n}'  "
              f"{'✓' if rec_label_n == label else '✗'}")

    # 6. Métricas completas
    print(f"\n📊 Métricas de precisión:")
    m = bam.accuracy(car_image, label)
    for k, v in m.items():
        print(f"   {k:25s}: {v}")

    # 7. Monitor de recursos
    stats = bam.memory_report()

    # 8. Visualizar
    print(f"\n🎨 Generando visualización...")
    visualize_results(bam, car_image, label)

    print("\n✅ Demo completada")
    return bam, car_image, label


