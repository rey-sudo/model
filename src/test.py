from pathlib import Path
from src.sign.codec import crear_atlas_cascada, index_to_sign, sign_to_index



ruta_actual= Path.cwd()




nodo_va = index_to_sign(1_198_765)
nodo_va.show()
nodo_va.save(f"test.png")

result = sign_to_index(nodo_va)
print(f"output: {result}")

cascada = crear_atlas_cascada(100)
cascada.show()
cascada.save(f"test.png")