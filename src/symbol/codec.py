import io
from pathlib import Path
from PIL import Image
import subprocess
import cairosvg
import numpy as np
import zxingcpp
import os

def word_to_aztec(path: Path, unicode_id: int, nombre_archivo):
    """
    Genera un Aztec Code usando Zint en modo binario puro.
    Optimizado para IDs de hasta 4.294.967.295 (4 bytes).
    """
    # 1. Preparar los datos binarios (4 bytes cubren hasta 4 mil millones)
    datos_bytes = unicode_id.to_bytes(4, byteorder='big')
    
    temp_bin = f"temp_aztec_{unicode_id}.bin"
    runa_svg = path / f"{nombre_archivo}.svg"
    runa_png = path / f"{nombre_archivo}.png"
    
    try:
        with open(temp_bin, "wb") as f:
            f.write(datos_bytes)

        # 2. Ejecutar Zint
        # --barcode=92: Aztec Code
        # --binary: Fuerza modo binario (Base256)
        # --scale=4: Tamaño del módulo (ajusta según la resolución de tu BAM)
        # --secure=0: Nivel de corrección de errores (0 es auto, puedes subirlo a 4+)
        subprocess.run([
            "zint",
            "--barcode=92", 
            "--binary",
            f"--input={temp_bin}",
            f"--output={runa_svg}",
            "--scale=4",
            "--notext"
        ], check=True, capture_output=True)
        
        svg_data = cairosvg.svg2png(url=str(runa_svg), output_width=120, output_height=120)
        
        img = Image.open(io.BytesIO(svg_data))
        img_resized = img.convert("L").resize((20, 20), Image.NEAREST)
        img_resized.save(runa_png)
            
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar Zint: {e.stderr.decode()}")
    finally:
        if os.path.exists(temp_bin):
            os.remove(temp_bin)

    return runa_svg

def decodificar_aztec(ruta_o_array):
    if isinstance(ruta_o_array, (str, Path)):
        img = Image.open(ruta_o_array)
    else:
        img = Image.fromarray(ruta_o_array.astype(np.uint8))
    
    resultados = zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.Aztec)
    if not resultados:
        return None
    
    try:
        return int.from_bytes(resultados[0].bytes, byteorder='big')
    except:
        return None
    
    
    