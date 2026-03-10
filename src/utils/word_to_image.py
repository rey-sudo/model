"""
frase_a_imagen.py
Convierte una palabra o frase en una imagen.

Argumentos:
    path     : directorio o ruta completa de destino
    frase    : texto a renderizar
    filename : nombre del archivo de salida (sin o con extensión)
    formato  : formato de salida — "PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF"
    padding  : espacio en píxeles alrededor del texto
               si es 1 la imagen se ajusta exactamente al contenido con 1 px de margen
    wrap     : si True, hace wrap del texto para que la imagen sea cuadrada
    size     : (ancho, alto) o int para cuadrado — define un lienzo fijo.
               Con size, todas las imágenes tienen el mismo tamaño y el texto
               siempre empieza en las mismas coordenadas (padding, padding),
               garantizando que las palabras comunes queden alineadas pixel a pixel.
               wrap=True + size → el texto hace wrap dentro del ancho disponible
               del lienzo en lugar de buscar proporción cuadrada.

Dependencias:
    pip install Pillow
"""

import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def word_to_image(
    path: str,
    frase: str,
    filename: str,                           # nombre del archivo de salida
    formato: str = "PNG",
    padding: int = 20,
    wrap: bool = False,                      # True → wrap del texto
    size: tuple[int, int] | int | None = None,  # lienzo fijo; sobreescribe dimensiones calculadas
    # ── Opciones de estilo ──────────────────────────────────────
    fuente_path: str | None = None,          # ruta a un .ttf/.otf
    fuente_size: int = 16,
    color_texto: str | tuple = "black",
    color_fondo: str | tuple = "white",
) -> Path:
    """
    Genera una imagen que contiene `frase` y la guarda en `path/filename`.

    Cuando `size` está definido:
      - El lienzo tiene siempre ese tamaño exacto.
      - El texto se ancla en (padding, padding) — esquina superior izquierda —
        de modo que palabras comunes quedan en las mismas coordenadas en todas
        las imágenes generadas con el mismo `size` y `padding`.
      - Si `wrap=True`, el texto hace wrap dentro del ancho del lienzo.
      - El texto que supere los límites del lienzo queda recortado.

    Cuando `size=None`:
      - Sin wrap → imagen ajustada al contenido.
      - Con wrap → imagen cuadrada calculada automáticamente.

    Returns
    -------
    Path  Ruta absoluta del archivo creado.
    """

    # ── Validar formato ─────────────────────────────────────────
    FORMATOS_SOPORTADOS = {"PNG", "JPEG", "JPG", "WEBP", "BMP", "GIF", "TIFF"}
    fmt = formato.upper()
    if fmt not in FORMATOS_SOPORTADOS:
        raise ValueError(
            f"Formato '{formato}' no soportado. "
            f"Usa uno de: {', '.join(sorted(FORMATOS_SOPORTADOS))}"
        )
    fmt_pillow = "JPEG" if fmt == "JPG" else fmt

    # ── Normalizar size ──────────────────────────────────────────
    if size is not None:
        if isinstance(size, int):
            canvas_w, canvas_h = size, size
        else:
            canvas_w, canvas_h = int(size[0]), int(size[1])
        if canvas_w < 1 or canvas_h < 1:
            raise ValueError(f"size debe ser >= 1 px, recibido: {size}")
    else:
        canvas_w = canvas_h = None

    # ── Resolver dest ────────────────────────────────────────────
    dest = Path(path)
    extension = fmt.lower() if fmt != "JPG" else "jpg"
    fn = Path(filename)
    nombre_archivo = fn.stem + (fn.suffix if fn.suffix else f".{extension}")

    if dest.is_dir() or not dest.suffix:
        dest = dest / nombre_archivo
    else:
        dest = dest.parent / nombre_archivo

    # ── Cargar fuente ───────────────────────────────────────────
    if fuente_path:
        fp = Path(fuente_path)
        if not fp.exists():
            raise FileNotFoundError(f"No se encontró la fuente: {fp}")
        font = ImageFont.truetype(str(fp), fuente_size)
    else:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fuente_size
            )
        except (IOError, OSError):
            font = ImageFont.load_default()

    # ── Helper: medir texto multilínea ──────────────────────────
    _dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def medir(texto: str) -> tuple[int, int, int, int]:
        """Devuelve (w, h, offset_x, offset_y) del bbox del texto."""
        bb = _dummy_draw.multiline_textbbox((0, 0), texto, font=font, spacing=4)
        return bb[2] - bb[0], bb[3] - bb[1], -bb[0], -bb[1]

    # ── Helper: wrap por píxeles (respeta el ancho real del lienzo) ──
    def wrap_por_pixels(texto: str, ancho_max: int) -> str:
        """Parte el texto en líneas que quepan en ancho_max píxeles."""
        palabras = texto.split()
        lineas: list[str] = []
        linea = ""
        for palabra in palabras:
            candidata = (linea + " " + palabra).strip()
            w, _, _, _ = medir(candidata)
            if w <= ancho_max or not linea:
                linea = candidata
            else:
                lineas.append(linea)
                linea = palabra
        if linea:
            lineas.append(linea)
        return "\n".join(lineas)

    # ── Helper: wrap por cuadrado (minimiza diferencia w/h) ─────
    def wrap_cuadrado(texto: str) -> tuple[str, int, int]:
        """Devuelve (texto_con_saltos, ancho, alto) más cercano a cuadrado."""
        palabras = texto.split()
        total_chars = len(texto)
        mejor_texto = texto
        mejor_diff  = float("inf")
        w0, h0, _, _ = medir(texto)
        mejor_w, mejor_h = w0, h0

        def wrap_chars(max_c: int) -> str:
            ls, l = [], ""
            for p in palabras:
                c = (l + " " + p).strip()
                if len(c) <= max_c or not l:
                    l = c
                else:
                    ls.append(l); l = p
            if l: ls.append(l)
            return "\n".join(ls)

        for mc in range(1, total_chars + 1):
            cand = wrap_chars(mc)
            w, h, _, _ = medir(cand)
            if w == 0 or h == 0:
                continue
            diff = abs(w - h)
            if diff < mejor_diff:
                mejor_diff = diff
                mejor_texto = cand
                mejor_w, mejor_h = w, h

        return mejor_texto, max(mejor_w, mejor_h, 1), max(mejor_w, mejor_h, 1)

    # ════════════════════════════════════════════════════════════
    # ── Calcular layout según combinación size / wrap ─────────
    # ════════════════════════════════════════════════════════════

    if canvas_w is not None:
        # ── MODO LIENZO FIJO ─────────────────────────────────────
        img_w, img_h = canvas_w, canvas_h
        area_w = canvas_w - padding * 2   # ancho disponible para el texto

        if wrap and area_w > 0:
            texto_final = wrap_por_pixels(frase, area_w)
        else:
            texto_final = frase

        # Posición fija: siempre (padding, padding) → mismas coordenadas en todas las imágenes
        _, _, offset_x, offset_y = medir(texto_final)
        pos_x = padding + offset_x
        pos_y = padding + offset_y
        align  = "left"

    elif wrap:
        # ── MODO AUTO-CUADRADO ───────────────────────────────────
        texto_final, mejor_w, mejor_h = wrap_cuadrado(frase)
        lado = mejor_w + padding * 2
        img_w = img_h = lado
        tw, th, offset_x, offset_y = medir(texto_final)
        pos_x  = (img_w - tw) // 2 + offset_x
        pos_y  = (img_h - th) // 2 + offset_y
        align  = "center"

    else:
        # ── MODO AUTO-RECTANGULAR (sin wrap, sin size) ───────────
        text_w, text_h, offset_x, offset_y = medir(frase)
        texto_final = frase
        img_w = max(text_w + padding * 2, 1)
        img_h = max(text_h + padding * 2, 1)
        pos_x  = padding + offset_x
        pos_y  = padding + offset_y
        align  = "left"

    # ── Crear imagen ─────────────────────────────────────────────
    mode = "RGB" if fmt_pillow in ("JPEG", "BMP", "GIF") else "RGBA"
    fondo = color_fondo
    if mode == "RGB" and isinstance(fondo, tuple) and len(fondo) == 4 and fondo[3] == 0:
        fondo = "white"

    img  = Image.new(mode, (img_w, img_h), fondo)
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (pos_x, pos_y), texto_final,
        font=font, fill=color_texto,
        align=align, spacing=4,
    )

    # ── Guardar ─────────────────────────────────────────────────
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"quality": 95} if fmt_pillow == "JPEG" else {}
    if fmt_pillow == "JPEG":
        img = img.convert("RGB")

    img.save(dest, format=fmt_pillow, **save_kwargs)

    modo_str = (
        f"lienzo fijo {img_w}×{img_h}" if canvas_w else
        ("cuadrada auto" if wrap else "rectangular auto")
    )
    #print(f"✅ [{modo_str}] {dest.resolve()}  ({img_w}×{img_h} px)")
    return dest.resolve()


