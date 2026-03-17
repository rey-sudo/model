import hashlib

def entero_a_coordenadas_3d(n, limite_min=0, limite_max=100):
    """
    Convierte un entero de cualquier longitud en coordenadas (x, y, z)
    con distribución uniforme dentro de un rango definido.
    """
    # 1. Convertimos el entero a una cadena de bytes para el hash
    n_bytes = str(n).encode('utf-8')
    hash_obj = hashlib.sha256(n_bytes).digest()
    
    # 2. Dividimos el hash (32 bytes) en 3 partes de 8 bytes cada una (64 bits)
    # Usamos los primeros 24 bytes del hash para x, y, z
    parte_x = int.from_bytes(hash_obj[0:8], byteorder='big')
    parte_y = int.from_bytes(hash_obj[8:16], byteorder='big')
    parte_z = int.from_bytes(hash_obj[16:24], byteorder='big')
    
    # Valor máximo posible para 8 bytes (2^64 - 1)
    max_64bit = 0xFFFFFFFFFFFFFFFF
    
    # 3. Normalizamos a un rango [0, 1] y luego al rango deseado
    rango = limite_max - limite_min
    
    x = limite_min + (parte_x / max_64bit) * rango
    y = limite_min + (parte_y / max_64bit) * rango
    z = limite_min + (parte_z / max_64bit) * rango
    
    return x, y, z

# --- Ejemplo de uso ---
numero_largo = 982374982374982374982374987234987234
coord = entero_a_coordenadas_3d(numero_largo, 0, 1000)

print(f"Entero: {numero_largo}")
print(f"Coordenadas: X={coord[0]:.2f}, Y={coord[1]:.2f}, Z={coord[2]:.2f}")