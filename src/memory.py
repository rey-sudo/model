"""
Memoria Asociativa Bidireccional (BAM - Bidirectional Associative Memory)
=========================================================================
Implementación completa de una BAM que asocia:
  - Imagen de entrada: 63×63 píxeles
  - Label de salida:   string de una sola palabra (ej. "carro")

La BAM puede:
  1. APRENDER  : almacenar el par (imagen, label)
  2. RECORDAR  : dado el label → reconstruir la imagen
  3. RECONOCER : dada la imagen (o versión ruidosa) → recuperar el label
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import time
import tracemalloc
import psutil

current_path = Path.cwd()
# ══════════════════════════════════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE   = 20           # píxeles de cada lado  (63 × 63)
N_PIXELS   = IMG_SIZE ** 2              # 3 969 neuronas en la capa de imagen
CHAR_BITS  = 8                          # bits por carácter (ASCII extendido)
MAX_CHARS  = 20                         # longitud máxima del label
N_LABEL    = CHAR_BITS * MAX_CHARS      # 160 neuronas en la capa de label
MAX_ITER   = 50                         # iteraciones máximas de convergencia


# ══════════════════════════════════════════════════════════════════════════════
#  Funciones de codificación / decodificación
# ══════════════════════════════════════════════════════════════════════════════

def image_to_bipolar(img_array: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen (63×63) en un vector bipolar de longitud 3969.
    Pasos:
      1. Escala de grises
      2. Umbral en 128  → binario {0, 1}
      3. Mapeo bipolar  → {-1, +1}
    """
    if img_array.ndim == 3:                        # RGB → escala de grises
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(float)

    # Normalizar a [0, 255] si es necesario
    if gray.max() <= 1.0:
        gray = gray * 255.0

    binary = (gray >= 128).astype(float)           # umbral
    bipolar = 2 * binary - 1                       # {0,1} → {-1,+1}
    return bipolar.flatten()                        # (3969,)


def bipolar_to_image(vec: np.ndarray) -> np.ndarray:
    """
    Reconstruye la imagen (63×63) desde un vector bipolar.
    """
    binary = ((vec + 1) / 2).reshape(IMG_SIZE, IMG_SIZE)
    return (binary * 255).astype(np.uint8)


def label_to_bipolar(label: str) -> np.ndarray:
    """
    Codifica un string en un vector bipolar de longitud N_LABEL (160).
    Cada carácter → 8 bits en complemento a 2 → {-1, +1}.
    El string se rellena con '\\0' hasta MAX_CHARS.
    """
    padded = label.ljust(MAX_CHARS, '\x00')[:MAX_CHARS]
    bits = []
    for ch in padded:
        val = ord(ch)
        for b in range(CHAR_BITS - 1, -1, -1):    # MSB primero
            bits.append(1 if (val >> b) & 1 else -1)
    return np.array(bits, dtype=float)             # (160,)


def bipolar_to_label(vec: np.ndarray) -> str:
    """
    Decodifica un vector bipolar (160,) de vuelta a string.
    """
    binary = ((np.sign(vec) + 1) / 2).astype(int)  # {-1,+1} → {0,1}
    chars = []
    for i in range(MAX_CHARS):
        byte = binary[i * CHAR_BITS:(i + 1) * CHAR_BITS]
        val = int(''.join(byte.astype(str)), 2)
        if val == 0:
            break
        chars.append(chr(val))
    return ''.join(chars)


# ══════════════════════════════════════════════════════════════════════════════
#  Clase BAM
# ══════════════════════════════════════════════════════════════════════════════

class BAM:
    """
    Memoria Asociativa Bidireccional.

    Arquitectura:
        Capa A  ←──────────────────────────→  Capa B
       (imagen)   W  (n×m)  /  W.T (m×n)    (label)
        n = 3969                              m = 160

    Aprendizaje (regla de Hebb):
        W  +=  x ⊗ y      (producto externo)

    Recuperación:
        y_new = sign(W.T @ x)      imagen → label
        x_new = sign(W   @ y)      label  → imagen
    """

    def __init__(self):
        self.W = np.zeros((N_PIXELS, N_LABEL), dtype=float)
        self.patterns: list[dict] = []             # memoria episódica
        print("✅ BAM inicializada  |  Capa A: {N_PIXELS} neuronas  |  Capa B: {N_LABEL} neuronas"
              .format(N_PIXELS=N_PIXELS, N_LABEL=N_LABEL))

    # ------------------------------------------------------------------
    #  Aprendizaje
    # ------------------------------------------------------------------
    def learn(self, image: np.ndarray, label: str) -> None:
        """
        Almacena un par (imagen, label) en la memoria.
        image  : array NumPy 63×63 (uint8 o float)
        label  : string de una sola palabra
        """
        x = image_to_bipolar(image)       # (3969,)
        y = label_to_bipolar(label)       # (160,)

        # Regla de Hebb:  W += x ⊗ y
        self.W += np.outer(x, y)

        self.patterns.append({'x': x, 'y': y,
                              'image': image.copy(),
                              'label': label})
        print(f"📚 Patrón aprendido: '{label}'  |  Patrones totales: {len(self.patterns)}")

    # ------------------------------------------------------------------
    #  Recuperación: imagen → label
    # ------------------------------------------------------------------
    def recall_label(self, image: np.ndarray, noisy: bool = False,
                     noise_level: float = 0.0) -> tuple[str, np.ndarray]:
        """
        Dado (posiblemente ruidosa) imagen → recupera el label.
        Retorna (label_str, label_vector_bipolar).
        """
        x = image_to_bipolar(image)
        if noisy and noise_level > 0:
            x = self._add_noise(x, noise_level)

        x = self._iterate_from_x(x)
        y = np.sign(self.W.T @ x)
        y[y == 0] = 1                              # desempate
        label = bipolar_to_label(y)
        return label, y

    # ------------------------------------------------------------------
    #  Recuperación: label → imagen
    # ------------------------------------------------------------------
    def recall_image(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Dado el label → reconstruye la imagen.
        Retorna (imagen_uint8 63×63, vector_bipolar).
        """
        y = label_to_bipolar(label)
        y, x = self._iterate_from_y(y)
        img_array = bipolar_to_image(x)
        return img_array, x

    # ------------------------------------------------------------------
    #  Dinámica de convergencia (iteraciones)
    # ------------------------------------------------------------------
    def _iterate_from_x(self, x: np.ndarray) -> np.ndarray:
        """Propaga x→y→x→… hasta convergencia."""
        for _ in range(MAX_ITER):
            y = np.sign(self.W.T @ x);  y[y == 0] = 1
            x_new = np.sign(self.W @ y); x_new[x_new == 0] = 1
            if np.array_equal(x_new, x):
                break
            x = x_new
        return x

    def _iterate_from_y(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propaga y→x→y→… hasta convergencia."""
        for _ in range(MAX_ITER):
            x = np.sign(self.W @ y);    x[x == 0] = 1
            y_new = np.sign(self.W.T @ x); y_new[y_new == 0] = 1
            if np.array_equal(y_new, y):
                break
            y = y_new
        return y, x

    # ------------------------------------------------------------------
    #  Utilidades
    # ------------------------------------------------------------------
    @staticmethod
    def _add_noise(vec: np.ndarray, level: float) -> np.ndarray:
        """Invierte aleatoriamente `level` fracción de bits."""
        noisy = vec.copy()
        n_flip = int(len(vec) * level)
        idx = np.random.choice(len(vec), n_flip, replace=False)
        noisy[idx] *= -1
        return noisy

    def similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Similitud coseno entre dos vectores bipolares."""
        return float(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-9))

    def accuracy(self, image: np.ndarray, label: str) -> dict:
        """Evalúa qué tan bien se recuerda el par (imagen, label)."""
        recalled_label, y_rec = self.recall_label(image)
        recalled_img, x_rec   = self.recall_image(label)

        x_orig = image_to_bipolar(image)
        y_orig = label_to_bipolar(label)

        return {
            'label_correcto': recalled_label == label,
            'label_recuperado': recalled_label,
            'similitud_imagen': self.similarity(x_orig, x_rec),
            'similitud_label' : self.similarity(y_orig, y_rec),
        }

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
      · Imagen original y su representación bipolar
      · Imagen reconstruida desde el label
      · Recuperación con ruido progresivo
      · Matriz de pesos (muestra)
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

    # ── Fila 0: imagen original y representación bipolar ───────────────
    ax0 = fig.add_subplot(gs[0, 0:2], **ax_style)
    ax0.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax0.set_title(f'Imagen Original\n"{label}"', **title_kw)
    ax0.axis('off')

    x_bip = image_to_bipolar(image).reshape(IMG_SIZE, IMG_SIZE)
    ax1 = fig.add_subplot(gs[0, 2:4], **ax_style)
    ax1.imshow(x_bip, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('Codificación Bipolar\nImagen  {-1, +1}', **title_kw)
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

    # Matriz de pesos (muestra 100×100)
    ax5 = fig.add_subplot(gs[1, 4:6], **ax_style)
    w_sample = bam.W[:100, :100]
    im5 = ax5.imshow(w_sample, cmap='coolwarm',
                     vmin=-np.abs(bam.W).max(),
                      vmax=np.abs(bam.W).max())
    ax5.set_title('Matriz de Pesos W\n(muestra 100×100)', **title_kw)
    ax5.tick_params(colors='#666', labelsize=7)
    plt.colorbar(im5, ax=ax5, fraction=0.046,
                 pad=0.04).ax.tick_params(colors='#888', labelsize=7)

    # ── Fila 2: recuperación con ruido creciente ───────────────────────
    sims, labels_rec = [], []
    x_orig = image_to_bipolar(image)

    for i, nl in enumerate(noise_levels):
        ax = fig.add_subplot(gs[2, i], **ax_style)
        if nl == 0.0:
            noisy_img = image
            x_in = x_orig.copy()
        else:
            x_in = bam._add_noise(x_orig, nl)
            noisy_img = bipolar_to_image(x_in)

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
        f"Dim. matriz W       : {bam.W.shape}"
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
    stats = bam.resource_usage(verbose=True)

    # 8. Visualizar
    print(f"\n🎨 Generando visualización...")
    visualize_results(bam, car_image, label)

    print("\n✅ Demo completada")
    return bam, car_image, label


if __name__ == "__main__":
    bam, image, label = main()