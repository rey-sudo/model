from blake3 import blake3
import struct

def posiciones_en_abecedario(palabra):
    abecedario = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
              'n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    palabra = palabra.lower()
    posiciones = []
    
    for letra in palabra:
        if letra in abecedario:
            posicion = abecedario.index(letra)
            posiciones.append(posicion)
        else:
            posiciones.append(None)
    
    return posiciones

def int_to_3d(number: int) -> tuple[int, int, int]:
    # 1. Aseguramos que sea múltiplo de 3 antes de procesar
    # Esto garantiza que el número 'caiga' en un nodo válido de la matrix
    number = number - (number % 3)
    
    digits = str(abs(number))
    n = len(digits)

    # Padding para asegurar que siempre haya al menos 3 dígitos
    if n < 3:
        digits = digits.zfill(3)
        n = 3

    # 2. División equitativa
    # Usamos n // 3 para los dos primeros, y el resto para el último
    chunk_size = n // 3
    
    x = int(digits[0 : chunk_size])
    y = int(digits[chunk_size : chunk_size * 2])
    z = int(digits[chunk_size * 2 :]) # Toma todo lo restante (pueden ser más dígitos)
    
    return (x, y, z)

def coordinates_from_index(arr: list[int]) -> tuple[int, int, int]:
    # 1. Convertimos el array a string para el hash
    s = ','.join(map(str, arr))
    
    # 2. Generamos el hash de 16 bytes
    h_bytes = blake3(s.encode()).digest(length=10)
    hash_entero = int.from_bytes(h_bytes, 'big')
    

    coordinates = int_to_3d(hash_entero)
    
    return tuple(coordinates)

