from pathlib import Path
from src.dicts.signs import SIGN_COLLECTION_RAW, SignManager
from src.memory import BAM, cargar_con_pillow
from src.sign.codec import block_to_individual_rows

current_path = Path.cwd()

sign_manager = SignManager(SIGN_COLLECTION_RAW)
sign_manager.build()    

def create_bam_dict_from_block(block_raw, sign_manager):
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

def apply_get_index(array):
    return [sign_manager.get_index_from_sign(word) for word in array]

def create_bam_dict(array):
    acc = []
    
    for i in range(1, len(array) + 1):
        chunk = array[:i]
        acc.append(chunk)
        
    return {i: v for i, v in enumerate(acc)}   

def train(array):
    acc = []
    
    for i in range(1, len(array) + 1):
        chunk = array[:i]
        acc.append(chunk)
        
    print(acc)


    
    
block_raw = ["the", "car", "is", "a", "vehicle", "with", "four", "wheels", "and", "transports"]    
block = apply_get_index(block_raw)
bam_dict=create_bam_dict(block)   


print(block_raw)
print(block)
print(bam_dict)

bam = BAM(bam_dict)    
    
    
    
    
    
    
    
"""         
        block_index = i - 1
        
        cascade = block_to_individual_rows(acc=acc, index=block_index, sign_size=9, block_length=len(bam_dict))
        cascade.save(f"cascada_{block_index}.png")        
        
        cascade_ = cargar_con_pillow(f"cascada_{block_index}.png")
        
        resultado = [str(i) for i in chunk]
        label = "_".join(resultado)
        
        print(f"traduce-> {label}")
        bam.learn_incremental(cascade_, label)
        
    """
    
    
    
#train(block) 





""" 

print("=" * 50)

input_mage = cargar_con_pillow(f"cascada_0.png")
input_label = bam.recall_ranking(input_mage)
print(input_label)

bam.memory_report()

"""
