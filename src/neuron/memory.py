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

from typing import Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Configuración — debe coincidir con bam_vision.py si se usan juntos
# ─────────────────────────────────────────────────────────────────
GRID      = 28 * 15   # retina 280×280 → 78,400 features
LABEL_DIM = 32 * 15   # 320 dims por vector de etiqueta

INPUT_DIR  = Path.cwd() / "input"
OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Preprocesamiento
# ─────────────────────────────────────────────────────────────────

def _preprocess(image_input, spatial: bool = False) -> Tuple[np.ndarray, Tuple[int,int]]:
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
    
    #spatial
         
    img = ImageOps.pad(img, (GRID, GRID), color=255)  
    arr = np.array(img, dtype=float)
    size = img.size
    result = np.where(arr < arr.mean(), 1.0, -1.0).flatten()
    
    return result, size


def _encode_label(idx: int, dim: int = LABEL_DIM) -> np.ndarray:
    """Índice → vector bipolar pseudoaleatorio reproducible."""
    rng = np.random.default_rng(seed=idx * 31337)
    return rng.choice([-1.0, 1.0], size=dim)


def _vec_to_image(vec: np.ndarray, size: Tuple[int,int] = (200,200)) -> Image.Image:
    """Vector bipolar → PIL.Image escalada a size×size."""
    grid     = vec.reshape(GRID, GRID)
    arr      = ((1 - grid) / 2 * 255).astype(np.uint8)
    img_small = Image.fromarray(arr, mode="L")
    img_small = img_small.filter(ImageFilter.GaussianBlur(radius=0.6))
    img_out   = img_small.resize((size), Image.LANCZOS)
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
        self._canonical_A: dict[str, np.ndarray] = {}
        self._canonical_A_size: dict[str, tuple[int, int]] = {}
        self._spatial: dict[str, bool] = {}
    # ── Entrenamiento ────────────────────────────────────────────
    def train_from_(self, filename: str, label: str,
                    spatial: bool = False,
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

        vec_A, original_size = _preprocess(ruta, spatial=spatial) 
        
        for existing_vec in self._A_rows:
            if np.array_equal(existing_vec, vec_A):
                print(f"  ⚠️  '{label}' ← {filename} ya registrado, se omite")
                return self
        

        # ── Vector etiqueta ──────────────────────────────────────
        if label not in self.label_vecs:
            idx = len(self.labels)
            self.labels.append(label)
            self.label_vecs[label] = _encode_label(idx)
            self._canonical_A[label]     = vec_A 
            self._canonical_A_size[label]     = original_size 
            self._spatial[label]     = spatial

        vec_B = self.label_vecs[label]

        # ── Acumular par ─────────────────────────────────────────
        self._A_rows.append(vec_A)
        self._B_rows.append(vec_B)

        # ── Reajustar matrices con pseudo-inversa ────────────────
        self._fit()

        # ── Guardar imagen procesada ─────────────────────────────
        if save_output:
            img_out = _vec_to_image(vec_A, original_size)
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
                  spatial: bool = False,
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

        vec, _   = _preprocess(image_input, spatial=spatial)
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
    def get_image_from_label(self, label: str, save: bool = True) -> Image.Image:
        label = label.strip().lower()
        if label not in self._canonical_A:
            raise ValueError(f"Etiqueta desconocida: '{label}'. Disponibles: {self.labels}")

        # Reconstrucción exacta desde el vector canónico almacenado
        img = _vec_to_image(self._canonical_A[label], size=self._canonical_A_size[label])

        if save:
            img.save(OUTPUT_DIR / f"{label}_ban_reconstruida.png")

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
    print(f"clasificacion: {result}")

    #================================================ REVERSE
    label = "banco abierto"
    img = ban.get_image_from_label(label)