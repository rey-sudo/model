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
from pathlib import Path
import os

ruta_actual = Path.cwd()
# ─────────────────────────────────────────────────────────────────
# 1.  PREPROCESAMIENTO: imagen 200×200 → vector bipolar (-1 / +1)
# ─────────────────────────────────────────────────────────────────
CANVAS   = 200 * 2   # tamaño de imagen de entrada
GRID     = 28  * 2  # resolución de la "retina" (28×28 = 784 features)
LABEL_DIM = 32 * 2  # dimensión del vector de etiqueta

FONT_BOLD= ruta_actual / "fonts/LiberationSans-Regular.ttf"

def autocrop(img: Image.Image, padding: float = 0.05) -> Image.Image:
    """
    Recorta la imagen al bounding box del contenido oscuro,
    añadiendo un margen proporcional.
    """
    arr = np.array(img)
    threshold = arr.mean()

    # Máscara de píxeles con contenido (oscuros)
    mask = arr < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img  # imagen vacía, no recortar

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Añadir padding
    pad_r = int((r_max - r_min) * padding)
    pad_c = int((c_max - c_min) * padding)
    h, w  = arr.shape

    r_min = max(0, r_min - pad_r)
    r_max = min(h, r_max + pad_r)
    c_min = max(0, c_min - pad_c)
    c_max = min(w, c_max + pad_c)

    return img.crop((c_min, r_min, c_max, r_max))


def preprocess(image_input) -> np.ndarray:
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

    # ── Autocrop: elimina canvas vacío alrededor del contenido ──
    img = autocrop(img)

    img = img.resize((GRID, GRID), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    threshold = arr.mean()
    vec = np.where(arr < threshold, 1.0, -1.0)
    return vec.flatten()


# ─────────────────────────────────────────────────────────────────
# 2.  GENERADORES DE PATRONES (imágenes 200×200 PIL)
# ─────────────────────────────────────────────────────────────────
def _base(size=CANVAS):
    img = Image.new("L", (size, size), 255)
    return img, ImageDraw.Draw(img)

def gen_texto(word: str, size: int = CANVAS) -> Image.Image:
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

def gen_image(path: str, size: int = CANVAS) -> Image.Image:
    img = Image.open(ruta_actual / "input" / path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    return img

PATTERN_GENERATORS = {
    "carro" : lambda: gen_texto("carro"),
    "manzana" : lambda: gen_texto("manzana"),
    "pera" : lambda: gen_texto("pera"),
    "banano" : lambda: gen_texto("banano"),
    "puta": lambda: gen_texto("puta"),
    "banco abierto": lambda: gen_texto("banco abierto")
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



def get_fonts() -> list[Path]:
    """Busca todas las fuentes disponibles en el sistema."""
    font_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        ruta_actual / "fonts",
    ]
    fonts = []
    for d in font_dirs:
        if d.exists():
            fonts += list(d.rglob("*.ttf")) + list(d.rglob("*.otf"))
    return fonts if fonts else [FONT_BOLD]  # fallback a la fuente default


def gen_texto_multifont(word: str, size: int = CANVAS) -> list[Image.Image]:
    """
    Genera imágenes del texto combinando todas las fuentes disponibles
    con múltiples tamaños relativos al canvas.
    """
    # Tamaños como fracción del canvas: 40%, 55%, 70%, 85%
    size_factors = [0.40, 0.55, 0.70, 0.85]

    imagenes = []
    for font_path in [FONT_BOLD]: #get_fonts
        for factor in size_factors:
            try:
                img, d = _base(size)
                target_w = size * factor
                target_h = size * factor

                font_size = size
                while font_size > 8:
                    font = ImageFont.truetype(str(font_path), font_size)
                    bbox = d.textbbox((0, 0), word, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w <= target_w and h <= target_h:
                        break
                    font_size -= 2

                x = (size - w) // 2 - bbox[0]
                y = (size - h) // 2 - bbox[1]
                d.text((x, y), word, fill=0, font=font)
                imagenes.append(img)
            except Exception:
                continue

    if not imagenes:
        imagenes.append(gen_texto(word, size))  # fallback

    if verbose := True:
        print(f"    {len(get_fonts())} fuentes × {len(size_factors)} tamaños = {len(imagenes)} imágenes base")

    return imagenes


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

def train(n_augment: int = 10, method: str = "pseudoinverse", verbose: bool = True) -> BAM:
    global LABELS, LABEL_VECS

    LABELS     = list(PATTERN_GENERATORS.keys())
    LABEL_VECS = {lbl: encode_label(i) for i, lbl in enumerate(LABELS)}

    all_A, all_B = {}, {}
    key_idx = 0

    for i, (label, gen) in enumerate(PATTERN_GENERATORS.items()):
        img_base = gen()
        
        img_base.save(Path.cwd() / "output" / f"{label}_generada.png")
        
        # ── Variantes por fuente ─────────────────────────────────
        imagenes = gen_texto_multifont(label) if label in LABELS else [img_base]

        total_variantes = 0
        for img in imagenes:
            for vec in augment(img, n=n_augment):
                all_A[f"{label}_{key_idx}"] = vec
                all_B[f"{label}_{key_idx}"] = encode_label(i)
                key_idx += 1
                total_variantes += 1
            
            
        if verbose:
            print(f"  ✓ '{label}'  →  {len(imagenes)} fuentes × {n_augment} aug = {total_variantes} muestras")

    net = BAM(n_features=GRID * GRID, label_dim=LABEL_DIM)
    net.fit(all_A, all_B, method=method)

    if verbose:
        print(f"\n✅ BAM entrenada")
        print(f"   Patrones  : {LABELS}")
        print(f"   Muestras  : {len(all_A)}")
        print(f"   Método    : {method}")
        print(f"   W_fwd     : {net.W.shape}")
        print(f"   W_back    : {net.W_back.shape}")

    return net






# Instanciar y entrenar la BAM con pseudo-inversa
bam = BAM(n_features=N_FEATURES, label_dim=LABEL_DIM)
bam = train(n_augment=10)

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
    result = bam.classify(vec, LABEL_VECS)
    return result                          # ← UNA SOLA PALABRA


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
    return Image.fromarray(arr, mode="L")         

def classify_from_input(filename: str, verbose: bool = True):
    """
    Toma una imagen de la carpeta /input y devuelve el label identificado.

    Parámetros
    ----------
    filename : str  — nombre del archivo (ej. "manzana_test.png")
    verbose  : bool — imprime scores de similitud por etiqueta

    Retorno
    -------
    str — etiqueta detectada
    """
    ruta = ruta_actual / "input" / filename
    
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró '{ruta}'. Archivos disponibles: "
                                f"{[f.name for f in (ruta_actual / 'input').iterdir()]}")

    img = Image.open(ruta).convert("L")
    
    vec = preprocess(img)
    label, scores = bam.classify(vec, LABEL_VECS)
    
    print(label, scores)
    
    if verbose:
        print(f"\n📂 Imagen  : {filename}")
        print(f"🏆 Label   : {label}")
        print("📊 Scores  :")
        for lbl, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int((score + 1) / 2 * 20)  # score ∈ [-1,1] → barra 0-20
            marker = " ← ganador" if lbl == label else ""
            print(f"   {lbl:<12} {score:+.4f}  {bar}{marker}")

    return label, scores, vec




def main():
    label, scores, vec = classify_from_input("1.png")
    print(label)
    
    label = "banco abierto"
    img_out = generate_image(label)          
    img_out.save(ruta_actual / "output" / f"{label}_generada.png")
    
if __name__ == "__main__":
    main()
