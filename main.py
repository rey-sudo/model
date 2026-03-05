from matrix import ConceptMatrix
from utils.hashing import encode_base62, hash_array


def posiciones_en_abecedario(palabra):
    abecedario = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
              'n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    palabra = palabra.lower()
    posiciones = []
    
    for letra in palabra:
        if letra in abecedario:
            posicion = abecedario.index(letra)
            posiciones.append(posicion + 1)
        else:
            posiciones.append(None)
    
    return posiciones


concepto = "cerebro"
definicion = ["animal", "de", "dos", "patas"]

raw_index = posiciones_en_abecedario(concepto)

print(f"raw_index: {raw_index}")

deterministic_index = hash_array(raw_index)

print(f"index: {deterministic_index}")

encoded = encode_base62(deterministic_index)

print(f"base62: {encoded}")


cm = ConceptMatrix(shape=(10000, 10000, 10000))

cm[0, 0, 0]       = 10.5

print(cm[0, 0, 0])