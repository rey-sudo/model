from pathlib import Path
from src.sign.codec import index_to_sign, sign_to_index
from src.symbol.codec import decodificar_aztec, word_to_aztec
from PIL import Image




ruta_actual= Path.cwd()




nodo_va = index_to_sign(100_000)
nodo_va.show()
nodo_va.save(f"test.png")

result = sign_to_index(nodo_va)
print(f"output: {result}")



#word_to_aztec(ruta_actual, 1_200_000, "test")
#print(decodificar_aztec("test.png"))
