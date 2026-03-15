import json
from pathlib import Path
from src.dicts.signs import SignManager
from src.memory import BAM
from src.dicts.codec import create_canvas_row
from PIL import Image
import re
from src.memory import memory_report, BAM

current_path = Path.cwd()

INPUT_PATH = Path("input")
SIGN_SIZE_PX = 9

collection_paths = [Path("dicts/english/words/alpha.txt"), Path(Path("dicts/english/words/custom.txt"))]

sign_manager = SignManager(collection_paths=collection_paths)
sign_manager.build()    

def train(bam, bam_dict):
    for i, value in bam_dict.items():
        total_items = len(bam_dict)
        
        cascade = create_canvas_row(value=value, total_items=total_items, sign_size=SIGN_SIZE_PX)

        label = ",".join(map(str, value))
        bam.learn_incremental(cascade, label)  
            
        #Image.fromarray(cascade).save(f"cascada_{i}.png")
        
    bam.flush() 
    
    
         
         
paragraph = sign_manager.load_block_file(path=INPUT_PATH / "block.md")
paragraph_ = re.findall(r'\b\d{4}\b|[a-zA-Z]{2,}', paragraph)

block = sign_manager.apply_index_to_block(paragraph_)
bam_dict = sign_manager.block_to_bam_dict(block)   

bam = BAM(total_signs=len(bam_dict), sign_size_px=SIGN_SIZE_PX)  

train(bam, bam_dict)   
 
print(json.dumps(memory_report(bam), indent=4, ensure_ascii=False))

def imprimir_ranking(datos):
    # Definir los encabezados y el ancho de las columnas
    header = f"{'ID':<4} | {'Label':<30} | {'Score':<10} | {'Votos':<6}"
    print(header)
    print("-" * len(header))

    for fila in datos:
        # :<N alinea a la izquierda con N espacios
        # :.4f reduce el score a 4 decimales para que no rompa la tabla
        id_val = fila['id']
        label = sign_manager.decode_labels(fila['label'])[-30:]
        score = fila['score']
        votos = fila['votos']
        
        print(f"{id_val:<4} | {label} | {score:<10.4f} | {votos:<6}")



#input_mage = cargar_con_pillow(f"cascada_2.png")
#ranking = bam.recall_ranking(input_mage)
#imprimir_ranking(ranking)

