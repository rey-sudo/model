from pathlib import Path
from src.symbol.codec import decodificar_aztec, word_to_aztec





ruta_actual= Path.cwd()



word_to_aztec(ruta_actual, 1_200_000, "test")
print(decodificar_aztec("test.png"))
