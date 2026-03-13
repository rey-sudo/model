import treepoem
import blake3

def generar_nodo_svg(unicode_id:int, nombre_archivo):
    # 1. Normalización y Hash (10 bytes para una matriz compacta)
    # Usamos BLAKE3 por su velocidad y seguridad
    #hash_bytes = blake3.blake3(elemento_unicode.encode('utf-8')).digest(length=10)
    #4 bytes: El valor máximo es $2^{32} - 1 = \mathbf{4.294.967.295}$. 
    datos_bytes = unicode_id.to_bytes(4, byteorder='big') 
    # 2. Generar el código de barras
    # Usamos treepoem para obtener la representación vectorial
    barcode = treepoem.generate_barcode(
        barcode_type='datamatrix',
        data=datos_bytes,
        options={"parse": True, "square": True } # Permite manejar datos binarios correctamente
    )
    
    # 3. Guardar como SVG
    # Treepoem devuelve un objeto Image de Pillow, pero podemos forzar el guardado vectorial
    # si usamos la opción de exportación directa de BWIPP o convertimos el postscript.
    with open(f"{nombre_archivo}.png", "w") as f:
        # Nota: treepoem genera por defecto un objeto de imagen. 
        # Para SVG puro, lo ideal es usar la salida de PostScript (EPS) y renombrar o convertir.
        barcode.save(f"{nombre_archivo}.png") # Salida estándar en raster

