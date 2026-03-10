La solución es deshabilitar autocrop y preservar la posición espacial original:
Con autocrop:                    Sin autocrop (spatial=True):
"carro fruta" → recorta todo     "carro fruta" → resize directo
→ carro en posición variable     → carro siempre en misma posición
Agregar spatial como parámetro en _preprocess y train_from_:
pythondef _preprocess(image_input, spatial: bool = False) -> np.ndarray:
    """
    spatial=False  →  autocrop + resize  (invariante a posición)
    spatial=True   →  resize directo     (preserva posición física)
    """
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

    # ── Spatial: skip autocrop ───────────────────────────────────
    if not spatial:
        img = _autocrop(img)

    img = img.resize((GRID, GRID), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    return np.where(arr < arr.mean(), 1.0, -1.0).flatten()
Y en train_from_ y classify_:
pythondef train_from_(self, filename: str, label: str,
                spatial: bool = False,
                save_output: bool = True) -> "BAN":

    label = label.strip().lower()
    ruta  = INPUT_DIR / filename
    vec_A = _preprocess(ruta, spatial=spatial)  # ← pasa spatial

    # guard duplicados
    for existing_vec in self._A_rows:
        if np.array_equal(existing_vec, vec_A):
            print(f"  ⚠️  '{label}' ← {filename} ya registrado, se omite")
            return self

    if label not in self.label_vecs:
        idx = len(self.labels)
        self.labels.append(label)
        self.label_vecs[label]   = _encode_label(idx)
        self._canonical_A[label] = vec_A
        self._spatial[label]     = spatial  # ← recuerda el modo por label

    ...

def classify_(self, image_input,
              spatial: bool = False,
              verbose: bool = True) -> tuple[str, dict]:

    if isinstance(image_input, str):
        image_input = INPUT_DIR / image_input

    vec   = _preprocess(image_input, spatial=spatial)  # ← mismo modo que entrenamiento
    ...
Añadir _spatial al __init__:
pythonself._spatial: dict[str, bool] = {}  # recuerda el modo de cada label
```

**Resultado con `spatial=True`:**
```
Imagen 1: "carro fruta"  spatial=True
→ resize directo 280×280
→ carro ocupa bits [0 : 39200]    ← posición fija
→ fruta ocupa bits [39200 : 78400]

Imagen 2: "carro perro"  spatial=True  
→ resize directo 280×280
→ carro ocupa bits [0 : 39200]    ← misma posición ✅
→ perro ocupa bits [39200 : 78400]

W_fwd = pinv([vec1, vec2]) @ [vec_B, vec_B]
→ bits [0:39200] de "carro" se refuerzan   ✅
→ bits de "fruta"/"perro" se cancelan      ✅
ModoCuándo usarlospatial=False (default)texto de tamaño/posición variablespatial=Trueobjetos anclados a una región fija de la imagen