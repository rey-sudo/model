import hashlib

class ConceptoFirmado:
    def __init__(self, nombre, coords):
        self.nombre = nombre
        self.coords = coords
        # La firma es inmutable, solo depende del nombre
        self.firma = hashlib.sha256(nombre.encode()).hexdigest()

    def actualizar_coords(self, nuevas_coords):
        self.coords = nuevas_coords
        print(f"🔄 Concepto '{self.nombre}' movido. Firma {self.firma} se mantiene constante.")

# --- ESCENARIO ---
m = {}
# Creamos el concepto
c1 = ConceptoFirmado("bank_finance", (10, 0, 0))
m[c1.firma] = c1

# Un párrafo guardado en la base de datos de la IA
parrafo_db = {
    "texto": "El banco central bajó las tasas.",
    "entidades": [c1.firma] # Solo guardamos la firma
}

# 1 año después... las coordenadas cambian en la matriz
m[c1.firma].actualizar_coords((15, -2, 0))

# El LLM recupera el párrafo y consulta la matriz
id_recuperado = parrafo_db["entidades"][0]
print(f"🔍 Recuperando significado para {id_recuperado}...")
print(f"📍 Coordenadas actuales: {m[id_recuperado].coords}")


def procesar_documento_medico(matriz_global, texto):
    # 1. Extraer palabras clave (esto lo haría el LLM)
    keywords = extraer_keywords(texto) 
    
    # 2. Crear una "Sub-Matriz" o instancia dedicada
    print("🔬 Escaneando densidad semántica...")
    
    puntuacion_medica = sum(matriz_global.cosine_sim(k, "biomedicina") for k in keywords if matriz_global.get(k))

    if puntuacion_medica > threshold:
        print("✅ Documento identificado como MÉDICO.")
        # Re-calibramos el concepto 'bank' para este hilo de conversación
        contexto_local = "bank_blood"
    
    # 3. Mapeo de nuevos conceptos del documento
    # Si el texto habla de una "arteria obstruida", la matriz calcula:
    # arteria (biomedicina) + obstruida (bloqueo) = nueva coordenada 3D
    for termino_nuevo in keywords_desconocidas:
        matriz_global.add(termino_nuevo, (0,0,0), [contexto_local, "patologia"], drift_mode="auto")
        
        
        
        
        
        
        
m = SemanticMatrix3D()

# --- POLOS DE GRAVEDAD JURÍDICA (Primitivos) ---
m.add("obligacion", (15.0, 0.0, 0.0))  # Eje X: Carga/Deber
m.add("penalidad", (0.0, 15.0, 0.0))   # Eje Y: Castigo/Consecuencia
m.add("territorio", (0.0, 0.0, 15.0))  # Eje Z: Alcance Geográfico

# --- ENTIDADES FIRMADAS (Conceptos con Nombre Único) ---
# Cada uno genera su firma hash automáticamente basada en el nombre
m.add("clausula_rescision", (0,0,0), ["obligacion", "penalidad"], drift_mode="auto")
m.add("fuero_judicial", (0,0,0), ["territorio"], drift_mode="auto")
m.add("fuerza_mayor", (0,0,0), ["obligacion"], drift_mode="auto")




def entrenar_documento_legal(texto, matriz):
    print(f"📄 Procesando Documento Legal...")
    
    # El LLM identifica términos y los vincula a la Firma de la Matriz
    # (Simulamos la detección de conceptos clave)
    mapeo_documento = [
        {"token": "rescisión", "firma": matriz.get("clausula_rescision").signature},
        {"token": "Madrid", "firma": matriz.get("fuero_judicial").signature}
    ]
    
    # Guardamos el párrafo "firmado" en la base de datos de conocimiento
    return {
        "contenido": texto,
        "firmas": mapeo_documento,
        "timestamp": "2026-03-16"
    }

doc_entrenado = entrenar_documento_legal(
    "Rescisión del contrato bajo las leyes de Madrid", m
)

def auditoria_logica(doc_firmado, matriz):
    for item in doc_firmado["firmas"]:
        concepto = matriz.get_by_signature(item["firma"])
        
        print(f"🧐 Auditando concepto: {concepto.name}")
        
        # Verificamos si la cláusula está cerca del polo de 'penalidad'
        dist = matriz.distance(concepto.name, "penalidad")
        
        if concepto.name == "clausula_rescision" and dist > 10:
            print(f"  ⚠️ ALERTA DE RIESGO: La cláusula de rescisión parece 'débil'.")
            print(f"  Distancia a penalidad: {dist:.2f} (Demasiado lejos)")
        else:
            print(f"  ✅ Consistencia lógica verificada.")

auditoria_logica(doc_entrenado, m)