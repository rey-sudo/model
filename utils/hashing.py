import hashlib
import base64
import string
from blake3 import blake3

def hash_array(arr):
    # 1. Convertimos el array a string para el hash
    s = ','.join(map(str, arr))
    
    # 2. Generamos el hash de 16 bytes
    h_bytes = blake3(s.encode()).digest(length=10)
    hash_entero = int.from_bytes(h_bytes, 'big')
    
    return hash_entero * 3


def encode_base62(num):
    # Definimos el alfabeto: 0-9 + a-z + A-Z
    chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
    base = len(chars)
    
    if num == 0:
        return chars[0]
        
    res = []
    while num > 0:
        num, rem = divmod(num, base)
        res.append(chars[rem])
    
    # Invertimos la lista porque los residuos se obtienen del final al principio
    return "".join(reversed(res))