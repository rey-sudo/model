from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"
GRID = 28 * 7
RETINA = (GRID, GRID)

RED1= {}
frase1 = "a car is a road vehicle"

def preprocesar_texto(frase):
    palabras = frase.split()
    resultado_tuplas = [] # Aquí guardaremos las tuplas (img, label)
    
    for i in range(1, len(palabras) + 1):
        # 1. Construimos el fragmento de la frase
        frs = " ".join(palabras[:i])
        
        # 2. Creamos la tupla y la añadimos a la lista
        nombre_archivo = f"{i}.png"
        resultado_tuplas.append((nombre_archivo, frs))
        
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=2, wrap=True, size=(RETINA), fuente_size=12)

    return resultado_tuplas


            
def entrenar_frase1():
    # Obtenemos [("1.png", "a"), ("2.png", "a car"), ...]
    resultado = preprocesar_texto(frase1)
    
    for i in range(1, len(resultado) + 1):
        RED1[i] = BAN()
        print(f"\n--- Entrenando BAN{i} con toda la secuencia ---")
        
        for _step_idx, (img, label) in enumerate(resultado, 1):
            # Solo podemos usar como upstream los BANs que ya terminaron 
            # de crearse en el diccionario RED1
            nodos_contexto = [RED1[j] for j in range(1, i) if j in RED1]
                
            if nodos_contexto:
                RED1[i].train_from_upstream_(img, label, upstream=nodos_contexto)
            else:
                # Si es el BAN1, no tiene upstream, sigue entrenando de forma lineal
                RED1[i].train_from_(img, label)

        
def detectar_frase():
    red = list(RED1.values())
    ultima_ban = red[-1]

    
    winner, scores, intermediate_scores = ultima_ban.classify_chained_("3.png",  upstream=red)
    print(winner)
    
    ultima_ban.summary()
    ultima_ban.memory_usage()
    



entrenar_frase1()
detectar_frase()












    #ban3.save("models/ban_v1.pkl")


def reconstruir_frase(clasificacion):
    prefix, frases_scores = clasificacion
    # add temperature
    # obtener frases válidas
    sentences = [s for s in frases_scores.keys() if s.strip()]

    split_sentences = [s.split() for s in sentences]
    max_len = max(len(s) for s in split_sentences)

    result = []

    for i in range(max_len):
        words = [s[i] for s in split_sentences if len(s) > i]
        word = Counter(words).most_common(1)[0][0]
        result.append(word)

    return result, " ".join(result)