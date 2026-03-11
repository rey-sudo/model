from PIL import Image
import numpy as np
from skimage import measure

def image_to_svg_paths(input_path, output_path):
    # 1. Cargar la imagen y convertirla a escala de grises
    img = Image.open(input_path).convert("L")
    width, height = img.size
    
    # Convertir a array de numpy (0-255)
    # Para texto negro sobre fondo blanco, el texto son valores bajos (~0)
    data = np.array(img)
    
    # 2. Encontrar contornos
    # El valor 128 es el umbral (puedes ajustarlo si el texto es muy tenue)
    # fully_connected='high' ayuda a que las letras no se corten
    contours = measure.find_contours(data, level=128, fully_connected='high')

    # 3. Generar el SVG
    with open(output_path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n')
        
        # Usamos un solo path con evenodd para los huecos de las letras
        f.write('  <path fill-rule="evenodd" d="')
        
        for contour in contours:
            # Los contornos de skimage vienen como [y, x]
            if len(contour) < 3:
                continue
                
            # Mover al primer punto (intercambiando y, x para el estándar SVG)
            f.write(f"M{contour[0][1]},{contour[0][0]} ")
            
            # Dibujar líneas al resto de puntos
            for point in contour[1:]:
                f.write(f"L{point[1]},{point[0]} ")
            
            f.write("Z ")
            
        f.write('" fill="black" stroke="none" />\n')
        f.write('</svg>')
