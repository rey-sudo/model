from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"
GRID = 28 * 7
RETINA = (GRID, GRID)

REDES = {} 

documento = [
    "a car is a road vehicle",
    "a car is used for transportation",
    "a car has four wheels",
]

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

def entrenar_frase(red: dict, frase: str):
    phrase_chunks = preprocesar_texto(frase)

    for i in range(1, len(phrase_chunks) + 1):
        red[i] = BAN()
        print(f"\n--- Entrenando BAN{i} con toda la secuencia ---")

        for img, label in phrase_chunks:
            nodos_contexto = [red[j] for j in range(1, i) if j in red]

            if nodos_contexto:
                red[i].train_from_upstream_(img, label, upstream=nodos_contexto)
            else:
                red[i].train_from_(img, label)


def entrenar_documento():
    for p_idx, parrafo in enumerate(documento, 1):
        print(f"\n{'═'*55}")
        print(f"  PÁRRAFO {p_idx}: '{parrafo}'")
        print(f"{'═'*55}")
        REDES[p_idx] = {}
        entrenar_frase(REDES[p_idx], parrafo)



def clasificar_documento(imagen: str, verbose: bool = True):
    print(f"\n{'═'*55}")
    print(f"  CONSULTA: {imagen}")
    print(f"{'═'*55}")

    for p_idx, red in REDES.items():
        print(f"\n  PÁRRAFO {p_idx}")
        print(f"  {'─'*40}")

        n_bans = len(red)

        for i in range(1, n_bans + 1):
            upstream = [red[j] for j in range(1, i) if j in red]

            if upstream:
                label, scores, _ = red[i].classify_chained_(
                    imagen, upstream=upstream, verbose=False
                )
            else:
                label, scores = red[i].classify_(imagen, verbose=False)

            score_val = scores[label]
            bar = "█" * int(abs(score_val) * 20)
            print(f"  BAN{i}  {score_val:+.4f}  {bar:<20}  \"{label}\"")

    print(f"\n{'═'*55}")
    

entrenar_documento()
print(REDES)
clasificar_documento("3.png")






    
