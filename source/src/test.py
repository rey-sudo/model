import json
from pathlib import Path
from PIL import Image
from dicts.signs import SignManager
from memory.memory import BAM
from memory import memory_report
from dicts.codec import create_canvas_row



INPUT_PATH = Path("input")
SIGN_SIZE_PX = 9
CONTEXT_LENGTH = 100

sign_manager = SignManager()

block = sign_manager.load_block_file(path=INPUT_PATH / "block.md")
imap, smap, cascade = sign_manager.get_cascade_from_block(block)

print(imap)
print(smap)
print(cascade)
bam = BAM(total_signs=CONTEXT_LENGTH, sign_size_px=SIGN_SIZE_PX)  

def train(bam, cascade):
    for i, value in cascade.items():
        total_items = len(cascade)
        
        canvas = create_canvas_row(value=value, sign_size_px=SIGN_SIZE_PX, total_signs=CONTEXT_LENGTH)

        label = ",".join(map(str, value))
        bam.learn_incremental(canvas, label) 
            
        Image.fromarray(canvas).save(f"cascada_{i}.png")
        
    bam.flush() 
    

train(bam, cascade)   
print(json.dumps(memory_report(bam), indent=4, ensure_ascii=False))

#============================================================================

sign_input = sign_manager.block_to_canvas(block="what is the animal domesticated", smap=smap, sign_size_px=bam.sign_size_px, total_signs=CONTEXT_LENGTH)
ranking = bam.recall_ranking(sign_input)


def imprimir_ranking(datos):
    # Definir los encabezados y el ancho de las columnas
    header = f"{'ID':<4} | {'Label':<30} | {'Score':<10} | {'Votos':<6}"
    print(header)
    print("-" * len(header))

    for fila in datos:
        # :<N alinea a la izquierda con N espacios
        # :.4f reduce el score a 4 decimales para que no rompa la tabla
        id_val = fila['id']
        label = sign_manager.decode_labels(fila['label'], imap)[-40:]
        score = fila['score']
        votos = fila['votos']
        
        print(f"{id_val:<4} | {label} | {score:<10.4f} | {votos:<6}")



imprimir_ranking(ranking)













"""  
nlp = spacy.load("en_core_web_sm")
doc = nlp(block)
tags_interes = ["NOUN", "PROPN", "VERB", "ADJ"] # "PRON", "ADV", "ADP", "DET", "AUX"
data_nlp = {
    tag: list(set(token.lemma_ for token in doc if token.pos_ == tag))
    for tag in tags_interes
}
json_output = json.dumps(data_nlp, indent=4, ensure_ascii=False)
#print(json_output)
"""


