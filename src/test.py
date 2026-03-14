from pathlib import Path
from src.dicts.signs import SIGN_COLLECTION_RAW, SignManager
from src.memory import BAM, cargar_con_pillow
from src.sign.codec import block_to_individual_rows

current_path = Path.cwd()

sign_manager = SignManager(SIGN_COLLECTION_RAW)
sign_manager.build()    

bam = BAM()


def create_block_from_raw(block_raw, sign_manager):
    """
    Convierte una lista de palabras en un diccionario mapeado
    a sus índices correspondientes mediante el sign_manager.
    """
    # Usamos enumerate para obtener (0, "el"), (1, "carro"), etc.
    block = {
        i: sign_manager.get_index_from_sign(word) 
        for i, word in enumerate(block_raw)
    }
    
    return block

block_raw = ["the", "car", "is", "a", "vehicle", "with", "four", "wheels", "and", "transports"]

block = create_block_from_raw(block_raw, sign_manager)

def translate_string(id_string, block_dict):
    """
    Convierte un string de IDs separados por guiones bajos
    en una frase legible.
    """
    # 1. Separamos el string por '_' para obtener una lista ['0', '1', ...]
    parts = id_string.split('_')
    
    # 2. Buscamos cada número en el diccionario
    # Usamos int(p) porque las llaves del dict son enteros
    words = [block_dict[int(p)] for p in parts if p.isdigit()]
    
    # 3. Unimos las palabras con un espacio
    return " ".join(words)



def train(diccionario):
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
        
        print(f"traduce-> {label}")
        bam.learn_incremental(cascade_, label)
      
 

train(block) 

print("=" * 50)

input_mage = cargar_con_pillow(f"cascada_0.png")
input_label = bam.recall_ranking(input_mage)
print(input_label)

bam.memory_report()


