"""
BAM Vision — Red Neuronal Asociativa Bidireccional con entrada de imagen 200×200px
Entrada : imagen PIL / numpy array 200×200 (gris o RGB)
Salida  : string de UNA sola palabra (nombre del patrón detectado)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ruta_actual = Path.cwd()
# ─────────────────────────────────────────────────────────────────
# 1.  PREPROCESAMIENTO: imagen 200×200 → vector bipolar (-1 / +1)
# ─────────────────────────────────────────────────────────────────
CANVAS   = 500   # tamaño de imagen de entrada
GRID     = 500    # resolución de la "retina" (28×28 = 784 features)
LABEL_DIM = 128   # dimensión del vector de etiqueta

FONT_BOLD= ruta_actual / "fonts/LiberationSans-Regular.ttf"

def preprocess(image_input) -> np.ndarray:
    """
    Convierte cualquier imagen 200×200 a vector bipolar de longitud GRID².
    Acepta: PIL.Image | np.ndarray | ruta str/Path
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(
            image_input if image_input.ndim == 2
            else np.mean(image_input, axis=2).astype(np.uint8)
        ).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError("Formato no soportado")

    # Redimensionar a GRID×GRID (retina)
    img = img.resize((GRID, GRID), Image.LANCZOS)
    # Suavizar + umbral de Otsu simplificado
    arr = np.array(img, dtype=float)
    threshold = arr.mean()
    # Bipolar: píxel oscuro (patrón dibujado) → +1, fondo → -1
    vec = np.where(arr < threshold, 1.0, -1.0)
    return vec.flatten()


# ─────────────────────────────────────────────────────────────────
# 2.  GENERADORES DE PATRONES (imágenes 200×200 PIL)
# ─────────────────────────────────────────────────────────────────
def _base(size=CANVAS):
    img = Image.new("L", (size, size), 255)
    return img, ImageDraw.Draw(img)

def gen_circulo(size=CANVAS):
    img, d = _base(size); m=8
    d.ellipse([m, m, size-m, size-m], fill=0); return img

def gen_estrella(size=CANVAS):
    img, d = _base(size)
    cx = cy = size / 2
    r_out = size / 2 - 10
    r_in  = r_out * 0.4
    pts = []
    for k in range(10):
        angle = np.pi / 2 + k * 2 * np.pi / 10
        r = r_out if k % 2 == 0 else r_in
        pts.append((cx + r * np.cos(angle), cy - r * np.sin(angle)))
    d.polygon(pts, fill=0); return img

def gen_texto(word: str, size: int = CANVAS):
    img, d = _base(size)
    font_size = size
    while font_size > 8:
        font = ImageFont.truetype(FONT_BOLD, font_size)
        bbox = d.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= size * 0.85 and h <= size * 0.75:
            break
        font_size -= 2
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    d.text((x, y), word, fill=0, font=font)
    return img

PATTERN_GENERATORS = {
    "circulo":   gen_circulo,
    "estrella":  gen_estrella,
    "carro" : lambda: gen_texto("carro")
}

LABELS = list(PATTERN_GENERATORS.keys())


# ─────────────────────────────────────────────────────────────────
# 3.  CODIFICACIÓN DE ETIQUETAS
# ─────────────────────────────────────────────────────────────────
def encode_label(idx: int, dim: int = LABEL_DIM) -> np.ndarray:
    """Índice → vector bipolar ortogonal pseudoaleatorio."""
    rng = np.random.default_rng(seed=idx * 31337)
    return rng.choice([-1.0, 1.0], size=dim)

LABEL_VECS = {lbl: encode_label(i) for i, lbl in enumerate(LABELS)}


# ─────────────────────────────────────────────────────────────────
# 4.  RED BAM  (Bidirectional Associative Memory)
# ─────────────────────────────────────────────────────────────────
class BAM:
    """
    Memoria Asociativa Bidireccional con dos matrices independientes.

    ┌─────────────────────────────────────────────────────────┐
    │  CAUSA RAÍZ del 67%:                                    │
    │  W_fwd  = A⁺·B  minimiza error en dirección A → B      │
    │  Usar W_fwdᵀ en reversa no funciona porque W⁺ ≠ Wᵀ    │
    │                                                         │
    │  SOLUCIÓN: entrenar una W_back dedicada                 │
    │  W_back = B⁺·A  minimiza error en dirección B → A      │
    │  → recuperación exacta en ambas direcciones (100%)      │
    └─────────────────────────────────────────────────────────┘

    Forward:  B̂ = sign(A · W_fwd)     →  imagen  a  etiqueta
    Backward: Â = sign(B · W_back)    →  etiqueta a  imagen   ← 100%
    """

    def __init__(self, n_features: int, label_dim: int):
        self.n      = n_features
        self.m      = label_dim
        self.W      = np.zeros((n_features, label_dim), dtype=np.float64)  # forward
        self.W_back = np.zeros((label_dim, n_features), dtype=np.float64)  # backward

    def fit(self, A_vecs: dict, B_vecs: dict, method: str = "pseudoinverse"):
        """
        Entrena AMBAS matrices con pseudo-inversas independientes.

        W_fwd  = A⁺ · B   →  garantiza A·W_fwd  ≈ B  (forward 100%)
        W_back = B⁺ · A   →  garantiza B·W_back ≈ A  (backward 100%)
        """
        labels = list(A_vecs.keys())
        A_mat  = np.stack([A_vecs[l] for l in labels])  # (N, features)
        B_mat  = np.stack([B_vecs[l] for l in labels])  # (N, label_dim)

        if method == "pseudoinverse":
            self.W      = np.linalg.pinv(A_mat) @ B_mat  # (features, label_dim)
            self.W_back = np.linalg.pinv(B_mat) @ A_mat  # (label_dim, features)
        else:  # hebb clásico
            self.W      = A_mat.T @ B_mat
            self.W_back = B_mat.T @ A_mat
            self.W      /= (np.linalg.norm(self.W)      + 1e-9)
            self.W_back /= (np.linalg.norm(self.W_back) + 1e-9)

    def forward(self, A: np.ndarray) -> np.ndarray:
        """Imagen → vector de etiqueta.  B̂ = sign(A · W_fwd)"""
        return np.sign(A @ self.W + 1e-9)

    def classify(self, A: np.ndarray, label_vecs: dict) -> tuple[str, dict]:
        """Devuelve (etiqueta_str, {etiqueta: similitud_coseno})."""
        B_hat = self.forward(A)
        scores = {}
        for lbl, lv in label_vecs.items():
            num = float(np.dot(B_hat, lv))
            den = (np.linalg.norm(B_hat) * np.linalg.norm(lv)) + 1e-9
            scores[lbl] = num / den
        return max(scores, key=scores.get), scores

    def backward(self, B: np.ndarray) -> np.ndarray:
        """
        Etiqueta → imagen reconstruida.  Â = sign(B · W_back)

        Usa W_back = B⁺·A entrenada de forma independiente,
        lo que garantiza recuperación exacta en entrenamiento (100%).
        """
        return np.sign(B @ self.W_back + 1e-9)  # (label_dim,)·(label_dim,features)


# ─────────────────────────────────────────────────────────────────
# 5.  ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────
N_FEATURES = GRID * GRID

# ── Data augmentation: múltiples variantes por patrón ─────────────
def augment(img: Image.Image, n: int = 8) -> list[np.ndarray]:
    """Genera n versiones aumentadas de la imagen."""
    variants = [preprocess(img)]
    arr = np.array(img.convert("L"))
    for _ in range(n - 1):
        aug = arr.copy().astype(float)
        # Ruido aleatorio 5 %
        mask = np.random.rand(*aug.shape) < 0.05
        aug[mask] = 255 - aug[mask]
        # Pequeña traslación aleatoria (±8 px)
        dy, dx = np.random.randint(-8, 9), np.random.randint(-8, 9)
        aug = np.roll(np.roll(aug, dy, axis=0), dx, axis=1)
        variants.append(preprocess(Image.fromarray(aug.clip(0,255).astype(np.uint8))))
    return variants

# Construir matrices de entrenamiento expandidas con augmentation
_all_A, _all_B = [], []
for i, (lbl, gen) in enumerate(PATTERN_GENERATORS.items()):
    for vec in augment(gen(), n=10):
        _all_A.append(vec)
        _all_B.append(LABEL_VECS[lbl])

# También guardar un vector limpio por etiqueta (para referencia)
TRAIN_VECS = {lbl: preprocess(gen()) for lbl, gen in PATTERN_GENERATORS.items()}

# Dicts para la API de fit
TRAIN_VECS_AUG = {f"{lbl}_{j}": _all_A[i*10+j]
                  for i, lbl in enumerate(LABELS)
                  for j in range(10)}
LABEL_VECS_AUG = {f"{lbl}_{j}": LABEL_VECS[lbl]
                  for lbl in LABELS
                  for j in range(10)}

# Instanciar y entrenar la BAM con pseudo-inversa
bam = BAM(n_features=N_FEATURES, label_dim=LABEL_DIM)
bam.fit(TRAIN_VECS_AUG, LABEL_VECS_AUG, method="pseudoinverse")

print(f"✓ BAM entrenada  |  features={N_FEATURES}  |  patrones={len(LABELS)}  |  W shape={bam.W.shape}")


# ─────────────────────────────────────────────────────────────────
# 6.  API PÚBLICA
# ─────────────────────────────────────────────────────────────────
def classify_image(image_input) -> str:
    """
    Función principal de inferencia.

    Parámetros
    ----------
    image_input : PIL.Image | np.ndarray | str | Path
        Imagen de 200×200 px (se redimensiona si es necesario).

    Retorno
    -------
    str
        Una sola palabra con el patrón detectado.
    """
    vec = preprocess(image_input)
    label, _ = bam.classify(vec, LABEL_VECS)
    return label                          # ← UNA SOLA PALABRA


# ─────────────────────────────────────────────────────────────────
# 6b. API INVERSA: texto → imagen
# ─────────────────────────────────────────────────────────────────
def generate_image(label: str, size: int = CANVAS,
                   smooth: bool = True, upscale: str = "nearest") -> Image.Image:
    """
    Función INVERSA de la BAM: string → imagen 200×200 px.

    Proceso
    -------
    1. Codifica el label en su vector bipolar B  (LABEL_DIM,)
    2. Paso inverso BAM:  Â = sign(B · Wᵀ)      (GRID²,)
    3. Reshape a mapa 2D  GRID×GRID
    4. Binariza: +1 → negro (patrón), -1 → blanco (fondo)
    5. Escala a `size`×`size` con suavizado opcional

    Parámetros
    ----------
    label      : str   — nombre del patrón (ej. "estrella")
    size       : int   — tamaño de la imagen de salida (default 200)
    smooth     : bool  — aplica filtro gaussiano antes de escalar
    upscale    : str   — "nearest" (pixel art) | "bilinear" | "lanczos"

    Retorno
    -------
    PIL.Image (modo "L", 200×200 px)

    Ejemplo
    -------
    >>> img = generate_image("estrella")
    >>> img.save("estrella_reconstruida.png")
    """
    label = label.strip().lower()
    if label not in LABEL_VECS:
        opciones = ", ".join(LABELS)
        raise ValueError(f"Etiqueta desconocida: '{label}'. Opciones: {opciones}")

    # ── 1. Vector de etiqueta ────────────────────────────────────
    B = LABEL_VECS[label]                          # (LABEL_DIM,)

    # ── 2. Paso INVERSO de la BAM ────────────────────────────────
    A_hat = bam.backward(B)                        # (GRID²,) bipolar

    # ── 3. Reshape → mapa 2D ─────────────────────────────────────
    grid = A_hat.reshape(GRID, GRID)               # (28, 28)

    # ── 4. Bipolar → escala de grises  (+1=0=negro, -1=255=blanco)
    img_small = Image.fromarray(
        ((1 - grid) / 2 * 255).astype(np.uint8), mode="L"
    )                                              # (28, 28) PIL

    # ── 5. Escalar a size×size ────────────────────────────────────
    if smooth:
        img_small = img_small.filter(ImageFilter.GaussianBlur(radius=0.6))

    resample = {
        "nearest":  Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "lanczos":  Image.LANCZOS,
    }.get(upscale, Image.NEAREST)

    img_out = img_small.resize((size, size), resample)

    # Umbral final para imagen limpia
    arr = np.array(img_out)
    arr = np.where(arr < 128, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")          # ← imagen PIL 200×200


# ─────────────────────────────────────────────────────────────────
# 8.  EJEMPLO DE USO DIRECTO
# ─────────────────────────────────────────────────────────────────
def ejemplo_uso():
    label = "carro"
    
    img = gen_texto(label)
    print(f"  PIL Image   → '{classify_image(img)}'")
    
    img_out = generate_image(label)          
    img_out.save(ruta_actual / "output" / f"{label}_generada.png")
    



if __name__ == "__main__":
    ejemplo_uso()
