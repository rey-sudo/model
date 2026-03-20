import json
from pathlib import Path
from PIL import Image
from dicts.signs import SignManager
from memory.memory import BAM, FirmaSemantica, RespuestaBAM
from memory import memory_report
from dicts.codec import create_canvas_row



INPUT_PATH = Path("input")
OUTPUT_PATH = Path("output")

SIGN_SIZE_PX = 9
CONTEXT_LENGTH = 100

sign_manager = SignManager()

block = sign_manager.load_block_file(path=INPUT_PATH / "block.md")
smap, cascade = sign_manager.get_cascade_from_block(block)


print(smap)
print(cascade)

bam = BAM("cat", total_signs=CONTEXT_LENGTH, sign_size_px=SIGN_SIZE_PX)  

def train(bam, cascade):
    for i, value in cascade.items():
        canvas = create_canvas_row(value=value, sign_size_px=SIGN_SIZE_PX, total_signs=CONTEXT_LENGTH)

        label = ",".join(map(str, value))
        
        firma= FirmaSemantica.desde_binario({
            "animal": 1, "nature": 1, "feline": 1
        })
        
        bam.learn_incremental(canvas, label, firma) 
            
        Image.fromarray(canvas).save(OUTPUT_PATH / f"cascada_{i}.png")
        
    bam.flush() 
    print(json.dumps(memory_report(bam), indent=4, ensure_ascii=False))

train(bam, cascade)   

#============================================================================

sign_input = sign_manager.block_to_canvas(block="the", smap=smap, sign_size_px=bam.sign_size_px, total_signs=CONTEXT_LENGTH)
ranking = bam.recall_ranking(sign_input)


def print_ranking(ranking: list[RespuestaBAM], titulo: str = "recall_ranking") -> None:
    print(f"\n{titulo}")
    print(f"{'ID':<6} {'Label':<30} {'Score':<12} {'Votos':<8} {'BAM':<15} {'Confiable'}")
    print("─" * 85)
    for i, r in enumerate(ranking):
        confiable = "✅" if r.confiable else "❌"
        firma_str = ", ".join(f"{k}={round(v,1)}" for k, v in r.firma.bits.items())
        print(f"{i:<6} {r.label:<30} {r.score:<12.4f} {r.votos:<8} {r.bam_id:<15} {confiable}")
        print(f"       firma: [{firma_str}]")
    print("─" * 85)
    print(f"total: {len(ranking)} resultados  |  confiables: {sum(1 for r in ranking if r.confiable)}")


print_ranking(ranking)


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


