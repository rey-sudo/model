import json
from dicts.hashing import string_to_coords_3d
from dicts.signs import SignManager
import spacy




block = "The West Bank"

sign_manager = SignManager()
cascade = sign_manager.get_cascade_from_block(block)
print(cascade)





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


