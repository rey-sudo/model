"""
frase_a_imagen.py
Convierte una palabra o frase en una imagen.

Argumentos:
    path    : ruta de destino donde se guarda la imagen (ej. "salida/imagen.png")
    frase   : texto a renderizar
    formato : formato de salida — "PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF"
    padding  : espacio en píxeles alrededor del texto
               si es 1 la imagen se ajusta exactamente al contenido con 1 px de margen
    filename : nombre del archivo de salida (sin o con extensión).
               Si se omite, se genera desde la frase.

Dependencias:
    pip install Pillow
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def word_to_image(
    path: str,
    frase: str,
    filename: str,                 # nombre del archivo de salida (sin o con extensión)
    formato: str = "PNG",
    padding: int = 20,
    # ── Opciones de estilo ──────────────────────────────────────
    fuente_path: str | None = None,   # ruta a un .ttf/.otf; None = fuente por defecto
    fuente_size: int = 48,            # tamaño en puntos (ignorado con fuente por defecto)
    color_texto: str | tuple = "black",
    color_fondo: str | tuple = "white",
) -> Path:
    """
    Genera una imagen que contiene `frase` y la guarda en `path`.

    Returns
    -------
    Path
        Ruta absoluta del archivo creado.

    Raises
    ------
    ValueError
        Si el formato no está soportado.
    FileNotFoundError
        Si `fuente_path` apunta a un archivo inexistente.
    """

    # ── Validar formato ─────────────────────────────────────────
    FORMATOS_SOPORTADOS = {"PNG", "JPEG", "JPG", "WEBP", "BMP", "GIF", "TIFF"}
    fmt = formato.upper()
    if fmt not in FORMATOS_SOPORTADOS:
        raise ValueError(
            f"Formato '{formato}' no soportado. "
            f"Usa uno de: {', '.join(sorted(FORMATOS_SOPORTADOS))}"
        )
    # Pillow usa "JPEG" para .jpg
    fmt_pillow = "JPEG" if fmt == "JPG" else fmt

    # ── Resolver path y filename ────────────────────────────────
    import re
    dest = Path(path)
    extension = fmt.lower() if fmt != "JPG" else "jpg"

    if filename:
        # Usar el nombre proporcionado; añadir extensión si no la tiene
        fn = Path(filename)
        nombre_archivo = fn.stem + (fn.suffix if fn.suffix else f".{extension}")
    else:
        # Derivar nombre desde la frase sanitizando caracteres no válidos
        nombre_base = re.sub(r'[^\w\-]', '_', frase.strip()).strip('_')
        nombre_base = re.sub(r'_+', '_', nombre_base)
        nombre_archivo = f"{nombre_base}.{extension}"

    # Si path es directorio (o ruta sin extensión), combinar con el nombre
    if dest.is_dir() or not dest.suffix:
        dest = dest / nombre_archivo
    elif filename:
        # path tiene extensión pero se pasó filename → usar directorio de path + filename
        dest = dest.parent / nombre_archivo

    # ── Cargar fuente ───────────────────────────────────────────
    if fuente_path:
        fp = Path(fuente_path)
        if not fp.exists():
            raise FileNotFoundError(f"No se encontró la fuente: {fp}")
        font = ImageFont.truetype(str(fp), fuente_size)
    else:
        try:
            # Intenta cargar DejaVuSans si está disponible en el sistema
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fuente_size)
        except (IOError, OSError):
            # Última opción: fuente bitmap incorporada en Pillow
            font = ImageFont.load_default()

    # ── Calcular tamaño del texto ────────────────────────────────
    # Usamos un canvas temporal para medir con precisión
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)

    bbox   = draw.textbbox((0, 0), frase, font=font)
    # bbox → (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # Offset para posicionar el texto (algunos glyphs tienen descenso negativo)
    offset_x = -bbox[0]
    offset_y = -bbox[1]

    # ── Tamaño final de la imagen ────────────────────────────────
    img_w = text_w + padding * 2
    img_h = text_h + padding * 2

    # ── Crear imagen y dibujar texto ─────────────────────────────
    # JPEG no soporta transparencia → forzar RGB
    mode = "RGB" if fmt_pillow in ("JPEG", "BMP", "GIF") else "RGBA"

    # Si el fondo es transparente y el modo es RGB, usar blanco
    fondo = color_fondo
    if mode == "RGB" and isinstance(fondo, tuple) and len(fondo) == 4 and fondo[3] == 0:
        fondo = "white"

    img  = Image.new(mode, (img_w, img_h), fondo)
    draw = ImageDraw.Draw(img)

    pos_x = padding + offset_x
    pos_y = padding + offset_y
    draw.text((pos_x, pos_y), frase, font=font, fill=color_texto)

    # ── Guardar ─────────────────────────────────────────────────
    dest.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if fmt_pillow == "JPEG":
        save_kwargs["quality"] = 95
        # JPEG no admite RGBA
        img = img.convert("RGB")

    img.save(dest, format=fmt_pillow, **save_kwargs)
    #print(f"✅ Imagen guardada en: {dest.resolve()}  ({img_w}×{img_h} px)")
    return dest.resolve()


# ── Demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ejemplos = [
        # padding = 1  →  imagen ajustada al texto con 1 px de margen
        dict(path="demo/ajustado.png",     frase="¡Hola, mundo!",            formato="PNG",  padding=1),
        # padding generoso, fondo oscuro, texto blanco
        dict(path="demo/oscuro.png",       frase="Python es genial",         formato="PNG",  padding=30,
             color_fondo="#1e1e2e", color_texto="#cdd6f4", fuente_size=64),
        # JPEG
        dict(path="demo/comprimido.jpg",   frase="Texto en JPEG",            formato="JPEG", padding=16),
        # Fondo transparente
        dict(path="demo/transparente.png", frase="Fondo transparente",       formato="PNG",  padding=20,
             color_fondo=(0, 0, 0, 0), color_texto="navy"),
        # Frase larga
        dict(path="demo/frase_larga.png",  frase="Convertir texto en imagen es muy útil",
             formato="PNG", padding=12, fuente_size=36),
    ]

    for kw in ejemplos:
        frase_a_imagen(**kw)