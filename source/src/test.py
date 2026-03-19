import json
from dicts.hashing import string_to_coords_3d
from dicts.signs import SignManager
import spacy




block = ("The West Bank is a landlocked territory near the Mediterranean coast of Western Asia, bordered by Jordan to the east and Israel to the north, south, and west. It is a region of immense historical and religious significance, housing ancient cities like Jericho, Hebron, and Bethlehem. Its landscape is characterized by rugged limestone hills and vast olive groves that have been cultivated for generations. Despite its cultural richness, the area is defined by a complex geopolitical reality and a fragmented administrative structure. This ongoing situation creates significant challenges for the daily lives and mobility of its residents, keeping the region at the center of international diplomatic focus.")
block = ["what", "is", "the", "west", "bank", "?"]


sign_manager = SignManager()


result = sign_manager.apply_coords_to_block(block)
print(result)











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


