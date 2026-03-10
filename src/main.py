from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"


ban = BAN()

frase = "el perro es un animal que kaka koko cucu keke"
chuncks = []

def construir_frases(frase):
    palabras = frase.split()
    
    for i in range(0, len(palabras) + 1):
        frs = " ".join(palabras[:i])
        chuncks.append(frs)
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=1)
    
    print(chuncks)    
     

def entrenar_memoria():
    for i, p in enumerate(chuncks):
        ban.train_from_(f"{i}.png",   p)
        
    ban.summary()


def detectar_frase():
    result = ban.classify_("input.png")
    print(f"clasificacion: {result}")    


construir_frases(frase)
entrenar_memoria()


