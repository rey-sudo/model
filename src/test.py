from pathlib import Path
from src.memory import BAM, cargar_con_pillow
from src.sign.codec import block_to_individual_rows

ruta_actual= Path.cwd()

bam = BAM()

block = { 
        0: "el",
        1: "carro",
        2: "es",
        3: "un",
        4: "vehiculo",
        5: "con",
        6: "cuatro",
        7: "ruedas",
        8: "y",
        9: "transporta"
        }

def imprimir_indices_acumulados(diccionario):
    # Obtenemos solo las llaves (los números 0, 1, 2...)
    indices = list(diccionario.keys())
    acc = []
    
    for i in range(1, len(indices) + 1):
        # Tomamos la porción de índices hasta i
        chunk = indices[:i]
        acc.append(chunk)
        
        block_index = i - 1
        
        cascade = block_to_individual_rows(acc=acc, index=block_index, sign_size=9, block_length=len(block))
        cascade.save(f"cascada_{block_index}.png")        
        
        cascade_ = cargar_con_pillow(f"cascada_{block_index}.png")
        
        resultado = [str(i) for i in chunk]
        label = "_".join(resultado)
        
        print(f"label_{block_index}->{label}")
        bam.learn_incremental(cascade_, label)
      
 
def imprimir_tabla_traducida(lista_datos, diccionario):
    # Encabezado de la tabla
    header = f"{'Rank':<6} {'Traducción':<50} {'Score':<10} {'Votos':<8}"
    print(header)
    print("─" * len(header))

    for item in lista_datos:
        # 1. Traducir el label a palabras
        indices = item['label'].split("_")
        palabras = [diccionario[int(i)] for i in indices if i.isdigit()]
        frase = " ".join(palabras)
        
        # 2. Formatear valores
        rank = item['rank']
        score = f"{item['score']:.4f}"
        votos = item['votos']
        
        # Marcador especial para el primer lugar (opcional, como el ◄ de tu ejemplo)
        marcador = " ◄" if rank == 1 else ""
        
        # 3. Imprimir fila con anchos fijos: 
        # <6 (6 espacios izquierda), <50 (50 espacios para la frase), etc.
        print(f"{rank:<6} {frase:<50} {score:<10} {votos:<8}{marcador}")


      
      
imprimir_indices_acumulados(block) 


input_mage = cargar_con_pillow(f"cascada_0.png")

input_label = bam.recall_ranking(input_mage)

print(imprimir_tabla_traducida(input_label, block))



