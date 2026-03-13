from pathlib import Path
from src.sign.codec import block_to_canvas, index_to_sign, sign_to_index

ruta_actual= Path.cwd()



"""
nodo_va = index_to_sign(1_198_765)
#nodo_va.show()
#nodo_va.save(f"test.png")

result = sign_to_index(nodo_va)
print(f"output: {result}")

cascada = crear_atlas_cascada(2)
cascada.show()
cascada.save(f"test.png")

"""

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
    
    for i in range(1, len(indices) + 1):
        # Tomamos la porción de índices hasta i
        chunk = indices[:i]
        print(chunk)
        
        cascade = block_to_canvas(chunk=chunk, sign_size=9, block_size=len(block))
        cascade.save(f"cascada_{i}.png")        
        
      
imprimir_indices_acumulados(block) 
      


