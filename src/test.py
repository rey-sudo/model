from pathlib import Path
import re
from src.dicts.signs import SIGN_COLLECTION_RAW, SignManager
from src.memory import BAM, cargar_con_pillow
from src.dicts.codec import create_canvas_row, index_to_sign
from PIL import Image

current_path = Path.cwd()

SIGN_SIZE = 9

sign_manager = SignManager(SIGN_COLLECTION_RAW)
sign_manager.build()    

def train(bam, bam_dict):
    for i, value in bam_dict.items():
        total_items = len(bam_dict)
        print(i, value)
        
        cascade = create_canvas_row(value=value, total_items=total_items, sign_size=SIGN_SIZE)

        label = ",".join(map(str, value))
        bam.learn_incremental(cascade, label)  
            
        Image.fromarray(cascade).save(f"cascada_{i}.png")
         
         
         
paragraph = "Automobiles have fundamentally transformed modern civilization, evolving from simple mechanical carriages into sophisticated feats of engineering. They provide unparalleled freedom, allowing individuals to traverse vast distances with ease and efficiency. Beyond mere utility, cars represent a fusion of art and technology, where sleek aesthetics meet cutting-edge innovations like autonomous driving and electric propulsion. As the industry shifts toward sustainability, the focus has moved to reducing carbon footprints through battery-powered motors. Whether it is a rugged off-road vehicle or a high-performance sports car, these machines remain essential symbols of personal mobility, continuous innovation, and the human desire"

block_raw = re.findall(r'\w+', paragraph)
block = sign_manager.apply_index_to_block(block_raw)
bam_dict=sign_manager.block_to_bam_dict(block)   


bam = BAM()  

train(bam, bam_dict)    
bam.memory_report()    
    

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



input_mage = cargar_con_pillow(f"cascada_2.png")
ranking = bam.recall_ranking(input_mage)
imprimir_ranking(ranking)

