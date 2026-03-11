from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"
GRID = 28 * 7
RETINA = (GRID, GRID)

frase = "the car is"
chunks = []


ban1 = BAN()
ban2 = BAN()
ban3 = BAN()

def preprocesar_texto(frase):
    palabras = frase.split()
    
    for i in range(0, len(palabras) + 1):
        frs = " ".join(palabras[:i])
        chunks.append(frs)
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=2, wrap=True, size=(RETINA), fuente_size=12)
        
def entrenar_memoria():
    for i, p in enumerate(chunks):
        
        if i == 0:
            continue
        
        if i == 1:
            ban1.train_from_(filename=f"{i}.png", label=p)

        if i == 2:
            ban2.train_from_upstream_(filename=f"{i}.png", label=p, upstream=ban1)
          
        if i == 3:
            ban3.train_from_upstream_(filename=f"{i}.png", label=p, upstream=[ban1, ban2])

        
    #ban1.summary()
    #ban1.memory_usage()
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

def detectar_frase():
    result = ban3.classify_chained_("1.png",  upstream=[ban1, ban2, ban3])
    
    print(result)


preprocesar_texto(frase)
entrenar_memoria()
detectar_frase()


