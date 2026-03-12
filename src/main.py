from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"
GRID = 28 * 6
RETINA = (GRID, GRID)

lineas = [
    "a cat with four wheels",
    "a car is used for transportation",
    "a car has four wheels",
]

LINEAS = {}   # { idx: BAN }

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

def entrenar_lineas():
    for i, linea in enumerate(lineas, 1):
        resultado = preprocesar_texto(linea)   # genera imágenes + tuplas

        LINEAS[i] = BAN()

        for img, label in resultado:
            LINEAS[i].train_from_(img, label)

        print(f"  ✓ LINEA{i}  '{linea}'  |  labels={len(resultado)}")
        #LINEAS[i].memory_usage()

def clasificar_lineas(imagen: str, verbose: bool = True) -> tuple:

    def consultar(i: int, ban: BAN) -> tuple:
        winner, scores = ban.classify_(imagen, verbose=False)
        return i, winner, scores[winner], scores

    resultados = {}

    with ThreadPoolExecutor(max_workers=len(LINEAS)) as executor:
        futuros = {
            executor.submit(consultar, i, ban): i
            for i, ban in LINEAS.items()
        }
        for futuro in as_completed(futuros):
            i, winner, score, scores = futuro.result()
            resultados[i] = {
                "linea"  : lineas[i - 1],
                "winner" : winner,
                "score"  : score,
                "scores" : scores,
            }

    # ── ranking de líneas por score ──────────────────────────────
    ranking = sorted(resultados.items(), key=lambda x: -x[1]["score"])

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  CONSULTA: {imagen}")
        print(f"{'═'*60}")
        for pos, (i, res) in enumerate(ranking, 1):
            bar    = "█" * int(abs(res["score"]) * 20)
            winner = res["winner"]
            score  = res["score"]

            # segundo label dentro de esa BAN
            second = sorted(res["scores"].items(), key=lambda x: -x[1])
            second = second[1] if len(second) > 1 else None

            linea  = f"  {pos}°  L{i}  {score:+.4f}  {bar:<20}  \"{winner}\""
            if second:
                linea += f"   2°: \"{second[0]}\" {second[1]:+.4f}"
            print(linea)

        mejor = ranking[0]
        print(f"\n  ➤  Línea más cercana : L{mejor[0]}")
        print(f"  ➤  '{mejor[1]['linea']}'")
        print(f"  ➤  Label             : \"{mejor[1]['winner']}\"")
        print(f"  ➤  Score             : {mejor[1]['score']:+.4f}")
        print(f"{'═'*60}\n")

    mejor_idx = ranking[0][0]
    return mejor_idx, ranking[0][1]["winner"], resultados





entrenar_lineas()
clasificar_lineas("2.png")




