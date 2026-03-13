from pathlib import Path
from src.memory import BAM, cargar_con_pillow
from src.sign.codec import block_to_canvas, index_to_sign, sign_to_index

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
        
        print(i)
        
        cascade = block_to_canvas(acc=acc, sign_size=9, block_length=len(block))
        cascade.save(f"cascada_{i}.png")        
        
        cascade_ = cargar_con_pillow(f"cascada_{i}.png")
        bam.learn_incremental(cascade_, f"index_{i}")
      
 
      
      
      
      
imprimir_indices_acumulados(block) 


input_mage = cargar_con_pillow(f"cascada_3.png")

rec_label = bam.recall_ranking(input_mage)
print(f"   Label recuperado  : '{rec_label}'")



