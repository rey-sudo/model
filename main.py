from matrix import ConceptMatrix
from utils.hashing import coordinates_from_index

MATRIX_SHAPE= 1_000_000_000

def posiciones_en_abecedario(palabra):
    abecedario = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
              'n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    palabra = palabra.lower()
    posiciones = []
    
    for letra in palabra:
        if letra in abecedario:
            posicion = abecedario.index(letra)
            posiciones.append(posicion)
        else:
            posiciones.append(None)
    
    return posiciones



cm = ConceptMatrix(shape=(MATRIX_SHAPE, MATRIX_SHAPE, MATRIX_SHAPE))

ecosistema_lexico = [
    {
        "concept": "bosque",
        "definition": [
            "sustantivo", "masculino", "ecosistema", "donde", "la", "vegetación", 
            "predominante", "son", "los", "árboles", "y", "matas", "que", "cubre", 
            "una", "extensión", "grande", "de", "terreno"
        ]
    },
    {
        "concept": "árbol",
        "definition": [
            "sustantivo", "masculino", "planta", "perenne", "de", "tronco", 
            "leñoso", "y", "elevado", "que", "se", "ramifica", "a", "cierta", 
            "altura", "del", "suelo", "formando", "una", "copa"
        ]
    },
    {
        "concept": "suelo",
        "definition": [
            "sustantivo", "masculino", "superficie", "de", "la", "corteza", 
            "terrestre", "biológicamente", "activa", "que", "proviene", "de", 
            "la", "desintegración", "de", "las", "rocas", "y", "residuos", "orgánicos"
        ]
    },
    {
        "concept": "fauna",
        "definition": [
            "sustantivo", "femenino", "conjunto", "de", "los", "animales", "de", 
            "un", "país", "región", "o", "medio", "determinado", "que", "viven", 
            "en", "un", "estado", "temporal", "o", "permanente"
        ]
    },
    {
        "concept": "clima",
        "definition": [
            "sustantivo", "masculino", "conjunto de", "condiciones", "atmosféricas", 
            "propias", "de", "un", "lugar", "constituido", "por", "la", "cantidad", 
            "y", "frecuencia", "de", "lluvias", "humedad", "y", "temperatura"
        ]
    }
]


for item in ecosistema_lexico:
    concepto_raw_index = posiciones_en_abecedario(item["concept"])
    print(f"concepto_raw_index: {concepto_raw_index}")


    concept_index = coordinates_from_index(concepto_raw_index)
    print(f"concept_index: {concept_index}")

    concept_definition = []

    for c in item["definition"]:
        cri = posiciones_en_abecedario(c)
        coo = coordinates_from_index(cri)
        concept_definition.append(coo)
    

    cm.set(concept_index, concept_definition)

print(cm.get((8510649, 8428853, 1706681)))

cm.plot(title="ConceptMatrix", save_html="concept_matrix_3d.html")