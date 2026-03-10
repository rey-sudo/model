import pickle
from typing import Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
import scipy.sparse as sp

GRID      = 28 * 7   # retina features
LABEL_DIM = 32 * 22   #  32 * labels - dims por vector de etiqueta

INPUT_DIR  = Path.cwd() / "input"
OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Preprocesamiento
# ─────────────────────────────────────────────────────────────────

def _preprocess(image_input) -> Tuple[sp.csr_matrix, Tuple[int,int]]:
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
    
         
    img = ImageOps.pad(img, (GRID, GRID), color=255)  
    arr = np.array(img, dtype=float)
    size = img.size
    
    vec = np.where(arr < arr.mean(), 1.0, -1.0).flatten()
    
    result = sp.csr_matrix(vec)
    
    return result, size


def _encode_label(idx: int, dim: int = LABEL_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed=idx * 31337)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)   # ← float32


def _vec_to_image(vec, size: Tuple[int,int] = (200,200)) -> Image.Image:
    """Vector bipolar → PIL.Image escalada a size×size."""
    if sp.issparse(vec):
        vec = vec.toarray().flatten()
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

    Attributes
    ----------
    labels      : list[str]           — etiquetas registradas en orden
    label_vecs  : dict[str, ndarray]  — vectores bipolares por etiqueta
    A_mat       : ndarray (N, GRID²)  — matriz de imágenes de entrenamiento
    B_mat       : ndarray (N, LABEL_DIM) — matriz de etiquetas
    W_fwd       : ndarray (GRID², LABEL_DIM)
    """

    def __init__(self):
        self.labels     : list[str]            = []
        self.label_vecs : dict[str, np.ndarray] = {}
        self._A_rows    : list[np.ndarray]     = []   # acumula vectores imagen
        self._B_rows    : list[np.ndarray]     = []   # acumula vectores etiqueta
        self.A_mat      : np.ndarray | None    = None
        self.B_mat      : np.ndarray | None    = None
        self.W_fwd      : np.ndarray | None    = None
        self._fitted    : bool                 = False
        self._seen_hashes: set = set()
        
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

        vec_A, original_size = _preprocess(ruta) 
        
        _fingerprint = vec_A.toarray().tobytes()
        if _fingerprint in self._seen_hashes:
            print(f"  ⚠️  '{label}' ya registrado, se omite")
            return self
        
        self._seen_hashes.add(_fingerprint)
                

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
            img_out = _vec_to_image(vec_A, original_size)
            img_out.save(OUTPUT_DIR / f"{label}_ban_entrenada.png")

        print(f"  ✓ '{label}'  ←  {filename}  |  "
              f"patrones={len(self.labels)}  muestras={len(self._A_rows)}")

        return self  # permite encadenar

    def _fit(self):
        self.A_mat = sp.vstack(self._A_rows)
        self.B_mat = np.stack(self._B_rows)                          # ← asignar primero

        A_dense    = self.A_mat.toarray().astype(np.float32)
        B_dense    = self.B_mat.astype(np.float32)

        self.W_fwd = np.linalg.pinv(A_dense) @ B_dense

        self._fitted = True
        
        self.A_mat = None  
        self.B_mat = None

    # ── Inferencia ───────────────────────────────────────────────
    def _forward(self, A) -> np.ndarray:
        # ── Convertir sparse a denso antes de multiplicar ────────────
        if sp.issparse(A):
            A = A.toarray().flatten()
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

        vec, _   = _preprocess(image_input)
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
                marker = " ← ganador" if lbl == winner else ""
                print(f"   {lbl:<14} {score:+.5f} {marker}")

        return winner, scores

    # ── Utilidades ───────────────────────────────────────────────
    def summary(self):
        """Imprime el estado actual de la BAN."""
        print("\n── BAN Summary ──────────────────────────────────────")
        print(f"   Patrones  : {self.labels}")
        print(f"   Muestras  : {len(self._A_rows)}")
        if self._fitted:
            print(f"   W_fwd     : {self.W_fwd.shape}")
        else:
            print("   Estado    : sin entrenar")
        print("─────────────────────────────────────────────────────\n")


    def memory_usage(self) -> dict:
        def mb(arr): return arr.nbytes / 1024 / 1024 if arr is not None else 0
        def mb_sparse(m): return m.data.nbytes / 1024 / 1024 if m is not None else 0

        A_rows_mb    = sum(mb_sparse(v) for v in self._A_rows)
        B_rows_mb    = sum(v.nbytes for v in self._B_rows)      / 1024 / 1024

        total = (A_rows_mb + B_rows_mb  +
                mb(self.W_fwd) +
                mb_sparse(self.A_mat) if sp.issparse(self.A_mat) else mb(self.A_mat) +
                mb(self.B_mat))

        report = {
            "_A_rows"      : f"{A_rows_mb:.2f} MB",
            "_B_rows"      : f"{B_rows_mb:.2f} MB",
            "W_fwd"        : f"{mb(self.W_fwd):.2f} MB",
            "A_mat"        : f"{mb_sparse(self.A_mat):.2f} MB" if sp.issparse(self.A_mat) else f"{mb(self.A_mat):.2f} MB",
            "B_mat"        : f"{mb(self.B_mat):.2f} MB",
            "TOTAL"        : f"{total:.2f} MB",
        }

        print("\n── BAN Memory Usage ─────────────────────────────────")
        for k, v in report.items():
            sep = "─" * 40 if k == "TOTAL" else ""
            if sep: print(sep)
            print(f"   {k:<16} {v:>10}")
        print("─────────────────────────────────────────────────────\n")

        return report

    def save(self, path: str | Path) -> None:
        """
        Guarda la instancia BAN completa en un archivo binario.

        Uso
        ---
        ban.save("models/ban_v1.pkl")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  💾 BAN guardada → {path}  ({path.stat().st_size / 1024 / 1024:.2f} MB)")


    @staticmethod
    def load(path: str | Path) -> "BAN":
        """
        Carga una instancia BAN desde un archivo binario.

        Uso
        ---
        ban = BAN.load("models/ban_v1.pkl")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró '{path}'.")
        with open(path, "rb") as f:
            instance = pickle.load(f)
        print(f"  ✅ BAN cargada  ← {path}  |  patrones={instance.labels}")
        return instance


