from src.matrix import ConceptMatrix

MATRIX_SHAPE= 1_000_000_000

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
    result = cm.add_concept(item["concept"], item["definition"])
    print(result)



cm.plot(title="ConceptMatrix", save_html="concept_matrix_3d.html")