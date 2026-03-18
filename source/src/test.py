# Load English tokenizer, tagger, parser and NER
import json
import spacy


nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("The West Bank is a landlocked territory near the Mediterranean coast of Western Asia, bordered by Jordan to the east and Israel to the north, south, and west. It is a region of immense historical and religious significance, housing ancient cities like Jericho, Hebron, and Bethlehem. Its landscape is characterized by rugged limestone hills and vast olive groves that have been cultivated for generations. Despite its cultural richness, the area is defined by a complex geopolitical reality and a fragmented administrative structure. This ongoing situation creates significant challenges for the daily lives and mobility of its residents, keeping the region at the center of international diplomatic focus.")
prompt= "what is the west bank?"

doc = nlp(text)


data_nlp = {
    "NOUN": list(set(token.lemma_ for token in doc if token.pos_ == "NOUN")),
    "PROPN": list(set(token.lemma_ for token in doc if token.pos_ == "PROPN")),
    "PRON": list(set(token.lemma_ for token in doc if token.pos_ == "PRON")),
    "VERB": list(set(token.lemma_ for token in doc if token.pos_ == "VERB"))
}

json_output = json.dumps(data_nlp, indent=4)
print(json_output)







""" 
print("ADJ:", [token.lemma_ for token in doc if token.pos_ == "ADJ"])
print("ADV:", [token.lemma_ for token in doc if token.pos_ == "ADV"])
print("ADP:", [token.lemma_ for token in doc if token.pos_ == "ADP"])
print("DET:", [token.lemma_ for token in doc if token.pos_ == "DET"])
print("AUX:", [token.lemma_ for token in doc if token.pos_ == "AUX"])
"""
