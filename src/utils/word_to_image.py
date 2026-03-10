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

Dependencias:
    pip install Pillow
"""

import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def word_to_image(
    path: str,
    frase: str,
    filename: str,                  # nombre del archivo de salida (sin o con extensión)
    formato: str = "PNG",
    padding: int = 20,
    wrap: bool = False,             # True → wrap del texto para imagen cuadrada
    # ── Opciones de estilo ──────────────────────────────────────
    fuente_path: str | None = None, # ruta a un .ttf/.otf; None = fuente por defecto
    fuente_size: int = 48,          # tamaño en puntos
    color_texto: str | tuple = "black",
    color_fondo: str | tuple = "white",
) -> Path:
    """
    Genera una imagen que contiene `frase` y la guarda en `path/filename`.

    Con wrap=True el texto se distribuye en múltiples líneas buscando
    la proporción más cercana a un cuadrado.

    Returns
    -------
    Path
        Ruta absoluta del archivo creado.
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

    # ── Helper: medir un bloque de texto ────────────────────────
    _dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def medir(texto: str) -> tuple[int, int, int, int]:
        """Devuelve (w, h, offset_x, offset_y) del texto."""
        bb = _dummy_draw.multiline_textbbox((0, 0), texto, font=font, spacing=4)
        return bb[2] - bb[0], bb[3] - bb[1], -bb[0], -bb[1]

    # ── Calcular layout ──────────────────────────────────────────
    if not wrap:
        # ── Sin wrap: una sola línea ─────────────────────────────
        text_w, text_h, offset_x, offset_y = medir(frase)
        texto_final = frase
        img_w = text_w + padding * 2
        img_h = text_h + padding * 2

    else:
        # ── Con wrap: buscar el ancho de línea que produce imagen más cuadrada
        palabras = frase.split()

        def wrap_en(max_chars: int) -> str:
            """Hace wrap simple por número máximo de caracteres por línea."""
            lineas, linea = [], ""
            for palabra in palabras:
                candidata = (linea + " " + palabra).strip()
                if len(candidata) <= max_chars or not linea:
                    linea = candidata
                else:
                    lineas.append(linea)
                    linea = palabra
            if linea:
                lineas.append(linea)
            return "\n".join(lineas)

        total_chars = len(frase)
        mejor_texto = frase
        mejor_diff  = float("inf")
        mejor_w = mejor_h = 0

        # Medir la frase sin wrap como baseline
        w0, h0, _, _ = medir(frase)
        mejor_w, mejor_h = w0, h0

        # Probar distintos anchos de línea (de 1 palabra a toda la frase)
        for max_chars in range(1, total_chars + 1):
            candidato = wrap_en(max_chars)
            w, h, _, _ = medir(candidato)
            if w == 0 or h == 0:
                continue
            diff = abs(w - h)
            if diff < mejor_diff:
                mejor_diff  = diff
                mejor_texto = candidato
                mejor_w, mejor_h = w, h

        texto_final = mejor_texto
        _, _, offset_x, offset_y = medir(texto_final)
        # El lado del cuadrado es el mayor de los dos ejes + padding
        # Mínimo garantizado: siempre >= texto + 2*padding
        lado = max(mejor_w, mejor_h, 1) + padding * 2
        img_w = img_h = lado  # imagen cuadrada

    # ── Crear imagen ─────────────────────────────────────────────
    mode = "RGB" if fmt_pillow in ("JPEG", "BMP", "GIF") else "RGBA"
    fondo = color_fondo
    if mode == "RGB" and isinstance(fondo, tuple) and len(fondo) == 4 and fondo[3] == 0:
        fondo = "white"

    img  = Image.new(mode, (img_w, img_h), fondo)
    draw = ImageDraw.Draw(img)

    if wrap:
        # Centrar el bloque de texto dentro del cuadrado
        tw, th, offset_x, offset_y = medir(texto_final)
        pos_x = (img_w - tw) // 2 + offset_x
        pos_y = (img_h - th) // 2 + offset_y
        draw.multiline_text(
            (pos_x, pos_y), texto_final,
            font=font, fill=color_texto,
            align="center", spacing=4,
        )
    else:
        draw.text(
            (padding + offset_x, padding + offset_y),
            texto_final, font=font, fill=color_texto,
        )

    # ── Guardar ─────────────────────────────────────────────────
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"quality": 95} if fmt_pillow == "JPEG" else {}
    if fmt_pillow == "JPEG":
        img = img.convert("RGB")

    img.save(dest, format=fmt_pillow, **save_kwargs)
    shape = "cuadrada" if wrap else "rectangular"
    print(f"✅ Imagen {shape} guardada en: {dest.resolve()}  ({img_w}×{img_h} px)")
    return dest.resolve()


