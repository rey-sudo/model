from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"


ban = BAN()

frase = "a car is a road vehicle that is powered by an engine and is able to carry a small number of people."
chuncks = []

RETINA_W = 28 * 8

RETINA = (RETINA_W, RETINA_W)

def construir_frases(frase):
    palabras = frase.split()
    
    for i in range(0, len(palabras) + 1):
        frs = " ".join(palabras[:i])
        chuncks.append(frs)
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=10, wrap=True, size=(RETINA), fuente_size=16)
    
    print(chuncks)    
     

def entrenar_memoria():
    for i, p in enumerate(chuncks):
        ban.train_from_(filename=f"{i}.png", label=p, spatial=True)
        
    ban.summary()
    ban.memory_usage()


def detectar_frase():
    result = ban.classify_("3.png")
    print(f"clasificacion: {result}")    

def extraer_imagen_label():
    ban.get_image_from_label("a car is a road vehicle")

construir_frases(frase)
entrenar_memoria()
detectar_frase()
extraer_imagen_label()
