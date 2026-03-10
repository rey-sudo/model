from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"


ban = BAN(bidirectional=False)

frase = "a car is a road vehicle that is powered by an engine and is able to carry a small number of people."
chunks = []

RETINA_W = 28 * 8

RETINA = (RETINA_W, RETINA_W)

def preprocesar_texto(frase):
    palabras = frase.split()
    
    for i in range(0, len(palabras) + 1):
        frs = " ".join(palabras[:i])
        chunks.append(frs)
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=2, wrap=True, size=(RETINA), fuente_size=12)
    

def entrenar_memoria():
    for i, p in enumerate(chunks):
        ban.train_from_(filename=f"{i}.png", label=p, spatial=True)
        
    ban.summary()
    ban.memory_usage()
    ban.save("models/ban_v1.pkl")

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
    ban = BAN.load("models/ban_v1.pkl")
    ban.memory_usage()
    
    result = ban.classify_("3.png")
    words, sentence = reconstruir_frase(result)
    print(f"{sentence}")    


preprocesar_texto(frase)
entrenar_memoria()
detectar_frase()



