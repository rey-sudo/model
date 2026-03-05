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


palabras = [] #Cindices

for palabra in palabras:
    resultado = posiciones_en_abecedario(palabra)
    print(f"  Lista: {resultado}")

    