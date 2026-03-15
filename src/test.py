from pathlib import Path
from src.dicts.signs import SIGN_COLLECTION_RAW, SignManager
from src.memory import BAM, cargar_con_pillow
from src.sign.codec import create_canvas_row

current_path = Path.cwd()

sign_manager = SignManager(SIGN_COLLECTION_RAW)
sign_manager.build()    

def train(bam_dict):
    for i, value in bam_dict.items():
        print(f"Llave {i} contiene: {value}")
        
        cascade = create_canvas_row(acc=bam_dict, index=i, sign_size=9, block_length=len(bam_dict))
        cascade.save(f"cascada_{i}.png")    
        
        cascade_img = cargar_con_pillow(f"cascada_{i}.png")
        
        bam.learn_incremental(cascade_img, str(i))       
    
    
block_raw = ["the", "car", "is", "a", "vehicle", "with", "four", "wheels", "and", "transports"]    
block = sign_manager.apply_index_to_block(block_raw)
bam_dict=sign_manager.block_to_bam_dict(block)   

print(block_raw)
print(block)
print(bam_dict)

bam = BAM()    #bam_dict
    
train(bam_dict)    
    
    
""" 

print("=" * 50)

input_mage = cargar_con_pillow(f"cascada_0.png")
input_label = bam.recall_ranking(input_mage)
print(input_label)

bam.memory_report()

"""
