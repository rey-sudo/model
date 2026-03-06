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
    digits = str(abs(number))
    n = len(digits)

    if n < 3:
        digits = digits.zfill(3)  # padding: 1 → "001", 99 → "099"
        n = 3

    chunk_size = n // 3
    
    x = int(digits[0 : chunk_size])
    y = int(digits[chunk_size : chunk_size * 2])
    z = int(digits[chunk_size * 2 :])
    return (x, y, z)

def coordinates_from_index(arr: list[int]) -> tuple[int, int, int]:
    # 1. Convertimos el array a string para el hash
    s = ','.join(map(str, arr))
    
    # 2. Generamos el hash de 16 bytes
    h_bytes = blake3(s.encode()).digest(length=10)
    hash_entero = int.from_bytes(h_bytes, 'big')
    
    coordinates = int_to_3d(hash_entero)
    
    return tuple(coordinates)

