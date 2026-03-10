"""
BAN — Bidirectional Associative Network (imagen única por patrón)
=================================================================
A diferencia de BAM Vision (que entrena con augmentation masiva),
BAN almacena pares (imagen → label) individuales y reconstruye
la imagen original exacta usando W_back = B⁺·A.

API
---
    ban = BAN()
    ban.train_from_("carro.png",   "carro")
    ban.train_from_("manzana.png", "manzana")
    ban.train_from_("pera.png",    "pera")

    label, scores = ban.classify_("test.png")
    img           = ban.get_image_from_label("carro")
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Configuración — debe coincidir con bam_vision.py si se usan juntos
# ─────────────────────────────────────────────────────────────────
GRID      = 28 * 10   # retina 280×280 → 78,400 features
LABEL_DIM = 32 * 10   # 320 dims por vector de etiqueta

INPUT_DIR  = Path.cwd() / "input"
OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Preprocesamiento
# ─────────────────────────────────────────────────────────────────
def _autocrop(img: Image.Image, padding: float = 0.05) -> Image.Image:
    """Recorta al bounding box del contenido oscuro + margen."""
    arr       = np.array(img)
    threshold = arr.mean()
    mask      = arr < threshold
    rows      = np.any(mask, axis=1)
    cols      = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    pad_r = int((r_max - r_min) * padding)
    pad_c = int((c_max - c_min) * padding)
    h, w  = arr.shape

    return img.crop((
        max(0, c_min - pad_c),
        max(0, r_min - pad_r),
        min(w, c_max + pad_c),
        min(h, r_max + pad_r),
    ))


def _preprocess(image_input) -> np.ndarray:
    """Imagen → vector bipolar (-1 / +1) de longitud GRID²."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        arr = image_input if image_input.ndim == 2 \
              else np.mean(image_input, axis=2).astype(np.uint8)
        img = Image.fromarray(arr).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError(f"Formato no soportado: {type(image_input)}")

    img = _autocrop(img)
    img = img.resize((GRID, GRID), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    return np.where(arr < arr.mean(), 1.0, -1.0).flatten()


def _encode_label(idx: int, dim: int = LABEL_DIM) -> np.ndarray:
    """Índice → vector bipolar pseudoaleatorio reproducible."""
    rng = np.random.default_rng(seed=idx * 31337)
    return rng.choice([-1.0, 1.0], size=dim)


def _vec_to_image(vec: np.ndarray, size: int = 200) -> Image.Image:
    """Vector bipolar → PIL.Image escalada a size×size."""
    grid     = vec.reshape(GRID, GRID)
    arr      = ((1 - grid) / 2 * 255).astype(np.uint8)
    img_small = Image.fromarray(arr, mode="L")
    img_small = img_small.filter(ImageFilter.GaussianBlur(radius=0.6))
    img_out   = img_small.resize((size, size), Image.LANCZOS)
    arr_out   = np.array(img_out)
    arr_out   = np.where(arr_out < 128, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_out, mode="L")


# ─────────────────────────────────────────────────────────────────
# Clase BAN
# ─────────────────────────────────────────────────────────────────
class BAN:
    """
    Bidirectional Associative Network — un patrón por imagen.

    Matrices
    --------
    W_fwd  = A⁺ · B   →   imagen  → etiqueta  (forward)
    W_back = B⁺ · A   →   etiqueta → imagen   (backward / reconstrucción exacta)

    Attributes
    ----------
    labels      : list[str]           — etiquetas registradas en orden
    label_vecs  : dict[str, ndarray]  — vectores bipolares por etiqueta
    A_mat       : ndarray (N, GRID²)  — matriz de imágenes de entrenamiento
    B_mat       : ndarray (N, LABEL_DIM) — matriz de etiquetas
    W_fwd       : ndarray (GRID², LABEL_DIM)
    W_back      : ndarray (LABEL_DIM, GRID²)
    """

    def __init__(self):
        self.labels     : list[str]            = []
        self.label_vecs : dict[str, np.ndarray] = {}
        self._A_rows    : list[np.ndarray]     = []   # acumula vectores imagen
        self._B_rows    : list[np.ndarray]     = []   # acumula vectores etiqueta
        self.A_mat      : np.ndarray | None    = None
        self.B_mat      : np.ndarray | None    = None
        self.W_fwd      : np.ndarray | None    = None
        self.W_back     : np.ndarray | None    = None
        self._fitted    : bool                 = False

    # ── Entrenamiento ────────────────────────────────────────────
    def train_from_(self, filename: str, label: str,
                    save_output: bool = True) -> "BAN":
        """
        Registra un par (imagen, label) y reajusta las matrices.

        Parámetros
        ----------
        filename    : str  — nombre del archivo en /input  (ej. "carro.png")
        label       : str  — etiqueta asociada             (ej. "carro")
        save_output : bool — guarda la imagen procesada en /output

        Retorno
        -------
        self — permite encadenar llamadas:
            ban.train_from_("a.png","a").train_from_("b.png","b")
        """
        label = label.strip().lower()
        ruta  = INPUT_DIR / filename

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró '{ruta}'.\n"
                f"Archivos en /input: {[f.name for f in INPUT_DIR.iterdir()]}"
            )

        # ── Vector imagen ────────────────────────────────────────
        vec_A = _preprocess(ruta)

        # ── Vector etiqueta ──────────────────────────────────────
        if label not in self.label_vecs:
            idx = len(self.labels)
            self.labels.append(label)
            self.label_vecs[label] = _encode_label(idx)

        vec_B = self.label_vecs[label]

        # ── Acumular par ─────────────────────────────────────────
        self._A_rows.append(vec_A)
        self._B_rows.append(vec_B)

        # ── Reajustar matrices con pseudo-inversa ────────────────
        self._fit()

        # ── Guardar imagen procesada ─────────────────────────────
        if save_output:
            img_out = _vec_to_image(vec_A)
            img_out.save(OUTPUT_DIR / f"{label}_ban_entrenada.png")

        print(f"  ✓ '{label}'  ←  {filename}  |  "
              f"patrones={len(self.labels)}  muestras={len(self._A_rows)}")

        return self  # permite encadenar

    def _fit(self):
        """Recalcula W_fwd y W_back con todos los pares acumulados."""
        self.A_mat  = np.stack(self._A_rows)   # (N, GRID²)
        self.B_mat  = np.stack(self._B_rows)   # (N, LABEL_DIM)
        self.W_fwd  = np.linalg.pinv(self.A_mat) @ self.B_mat
        self.W_back = np.linalg.pinv(self.B_mat) @ self.A_mat
        self._fitted = True

    # ── Inferencia ───────────────────────────────────────────────
    def _forward(self, A: np.ndarray) -> np.ndarray:
        return np.sign(A @ self.W_fwd + 1e-9)

    def classify_(self, image_input,
                  verbose: bool = True) -> tuple[str, dict]:
        """
        Clasifica una imagen y retorna (label, scores).

        Parámetros
        ----------
        image_input : str | Path | PIL.Image | np.ndarray
            Si es str se busca en /input.

        Retorno
        -------
        (label: str, scores: dict[str, float])
        """
        if not self._fitted:
            raise RuntimeError("BAN sin entrenar. Llama train_from_() primero.")

        if isinstance(image_input, str):
            image_input = INPUT_DIR / image_input

        vec   = _preprocess(image_input)
        B_hat = self._forward(vec)

        scores = {}
        for lbl, lv in self.label_vecs.items():
            num          = float(np.dot(B_hat, lv))
            den          = (np.linalg.norm(B_hat) * np.linalg.norm(lv)) + 1e-9
            scores[lbl]  = num / den

        winner = max(scores, key=scores.get)

        if verbose:
            print(f"\n🏆 Label   : {winner}")
            print("📊 Scores  :")
            for lbl, score in sorted(scores.items(), key=lambda x: -x[1]):
                bar    = "█" * int((score + 1) / 2 * 20)
                marker = " ← ganador" if lbl == winner else ""
                print(f"   {lbl:<14} {score:+.4f}  {bar}{marker}")

        return winner, scores

    # ── API inversa: label → imagen ──────────────────────────────
    def get_image_from_label(self, label: str,
                             size: int = 200,
                             save: bool = True) -> Image.Image:
        """
        Reconstruye la imagen original desde el label usando W_back.

        Proceso
        -------
        1. B = label_vecs[label]          (LABEL_DIM,)
        2. Â = sign(B · W_back)           (GRID²,)  ← reconstrucción exacta
        3. Reshape → GRID×GRID → PIL.Image size×size

        Parámetros
        ----------
        label : str  — etiqueta registrada (ej. "carro")
        size  : int  — tamaño de salida en px
        save  : bool — guarda en /output/{label}_ban_reconstruida.png

        Retorno
        -------
        PIL.Image modo "L"
        """
        if not self._fitted:
            raise RuntimeError("BAN sin entrenar. Llama train_from_() primero.")

        label = label.strip().lower()
        if label not in self.label_vecs:
            raise ValueError(
                f"Etiqueta desconocida: '{label}'.\n"
                f"Disponibles: {self.labels}"
            )

        # ── 1. Vector etiqueta ───────────────────────────────────
        B = self.label_vecs[label]                        # (LABEL_DIM,)

        # ── 2. Paso inverso: B · W_back → Â ─────────────────────
        A_hat = np.sign(B @ self.W_back + 1e-9)          # (GRID²,)

        # ── 3. Vector → imagen ───────────────────────────────────
        img = _vec_to_image(A_hat, size=size)

        if save:
            img.save(OUTPUT_DIR / f"{label}_ban_reconstruida.png")
            print(f"  💾 Guardada: output/{label}_ban_reconstruida.png")

        return img

    # ── Utilidades ───────────────────────────────────────────────
    def summary(self):
        """Imprime el estado actual de la BAN."""
        print("\n── BAN Summary ──────────────────────────────────────")
        print(f"   Patrones  : {self.labels}")
        print(f"   Muestras  : {len(self._A_rows)}")
        if self._fitted:
            print(f"   W_fwd     : {self.W_fwd.shape}")
            print(f"   W_back    : {self.W_back.shape}")
        else:
            print("   Estado    : sin entrenar")
        print("─────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────
# Ejemplo de uso
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ban = BAN()
    
    #================================================ TRAIN
    (ban
        .train_from_("1.png",   "banco abierto")
        .train_from_("2.png",   "banco abierto")
        .train_from_("3.png",   "banco abierto")
        .train_from_("4.png",   "carro")
    )

    ban.summary()
    
    #================================================ CLASSIFY

    result = ban.classify_("4.png")
    print(f"clasificacion result: {result}")

    #================================================ REVERSE
    label = "carro"
    img = ban.get_image_from_label(label)