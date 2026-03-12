class BANSemantica:
    """
    BAN + diccionario de primitivos + espacio 3D.
    Todo persistido en un solo .pkl
    """

    def __init__(self):
        self.ban      = BAN()
        self.dic      : dict[str, list[str]] = {}
        self.espacio  : EspacioSemantico3D | None = None

    def definir(self, concepto: str,
                definicion: list[str]) -> "BANSemantica":
        # validar que la definición solo use primitivos
        for rasgo in definicion:
            if rasgo in self.dic:
                raise ValueError(
                    f"'{rasgo}' es un concepto compuesto — "
                    f"usa solo primitivos en la definición"
                )
        self.dic[concepto] = definicion
        return self

    def construir(self) -> "BANSemantica":
        """Construye el espacio semántico desde el diccionario."""
        self.espacio = EspacioSemantico3D(self.dic)
        print(f"  ✅ espacio construido")
        print(f"     conceptos  : {len(self.dic)}")
        print(f"     primitivos : {self.espacio.P}")
        print(f"     matriz     : {self.espacio.M.shape}")
        return self

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
```

---

### Ejemplo concreto — qué ve la matriz
```
primitivos ordenados:
  0:asiento  1:credito  2:deposito  3:dinero  4:dos
  5:edificio 6:exterior 7:madera    8:motor   9:movil
  10:cuatro  11:ruedas

matriz M (conceptos × primitivos):
               as  cr  dep din dos edi ext mad mot mov cua rue
banco_fin    [  0,  1,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0 ]
banco_mueble [  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0 ]
carro        [  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1 ]
moto         [  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1 ]

distancia(carro, moto)        = 1.41  ← cercanos  ✅
distancia(carro, banco_fin)   = 2.83  ← lejanos   ✅
distancia(banco_fin,banco_mueble) = 2.83  ← muy lejanos ✅
```

---

### Conclusión
```
Primitivos puros en la definición →
  cada concepto = punto exacto en R^P
  sin ambigüedad posible
  sin referencias circulares
  sin degradación por centroide de centroides

La matriz M (C × P) es:
  el diccionario en forma de coordenadas
  un espacio semántico navegable
  la base para inferencia por álgebra de conjuntos
  proyectable a 3D para visualización

El índice puede ser compuesto  →  maneja ambigüedad léxica
La definición solo primitivos  →  coordenada exacta en el espacio

Es la arquitectura semántica más limpia posible
porque separa completamente
  forma  (el índice — puede ser ambiguo)
  significado (la definición — siempre exacto)






# concepto que no existe en el diccionario
# pero se puede construir desde primitivos

firma_nuevo = sign(
    firma_movil    +
    firma_agua     +
    firma_motor    +
    firma_personas
)
# → vecinos más cercanos: [barco, lancha, ferry]
# BAN infiere el concepto más cercano
# sin haberlo visto nunca                   ✅
```

---

### 11. Compatibilidad directa con BAN
```
primitivos → firmas bipolares deterministas (hash del nombre)
definicion → suma ponderada de firmas de primitivos
concepto   → firma bipolar final

esta firma ES la coordenada en el espacio BAN
el mismo vector se usa para:
  clasificar   →  cosine con W_fwd
  generar      →  W_back
  desambiguar  →  Hadamard con concepto activo
  razonar      →  álgebra de conjuntos sobre M
                                                ✅
```

---

### 12. Persistencia trivial
```
el diccionario es un dict de Python
la matriz M es un ndarray

pickle.dump → todo guardado ✅
pickle.load → todo restaurado ✅

no hay estructura compleja que reconstruir
no hay índices que sincronizar
no hay consistencia que verificar
```

---

### Resumen — ventajas únicas de este sistema
```
Ventaja                         Ningún otro sistema lo tiene así
──────────────────────────────  ─────────────────────────────────────
Coordenada exacta               WordNet no / Embeddings no
Sin degradación jerárquica      Ontologías no
Inferencia sin reglas           Grafos necesitan reglas explícitas
Distancia semántica real        Embeddings son aproximaciones
Taxonomía automática            WordNet requiere mantenimiento
Detección de inconsistencias    Ninguno
Composición de conceptos nuevos Transformers sí pero con GPU
Actualización O(1)              Transformers no
Persistencia trivial            Grafos no
Compatible con BAN directamente Cualquier otro necesita adaptación
```

---

### La ventaja más profunda — el significado ES la posición
```
En todos los demás sistemas:
  el significado está CODIFICADO en parámetros
  no puedes inspeccionarlo directamente
  no puedes modificarlo sin reentrenar

En este sistema:
  el significado ES la fila en la matriz M
  puedes leerlo    →  M[carro]  = [0,0,0,0,0,0,0,0,1,1,1,1]
  puedes editarlo  →  M[carro, idx("agua")] = 1
  puedes compararlo → distancia(M[carro], M[moto])
  puedes componerlo → sign(M[carro] + M[barco])

el significado se vuelve matemáticamente
manipulable sin reentrenar
sin backprop
sin GPU



# si dos conceptos tienen coordenadas idénticas
if distancia("carro", "automovil") == 0.0:
    print("duplicado — misma definición, índice distinto")  ⚠️

# si un concepto está muy lejos de todos los demás
if min(distancias("ovni")) > umbral:
    print("concepto aislado — primitivos muy distintos")  ⚠️

# si dos conceptos considerados iguales están lejos
if distancia("banco_a", "banco_financiero") > 0.0:
    print("inconsistencia — deberían ser el mismo concepto") ⚠️
```

El espacio semántico hace visibles los errores del diccionario.

---

### 9. Sección por primitivo — taxonomía automática
```
plano("ruedas"):
  [carro, moto, bicicleta, camion, tractor]
  → taxonomía de vehículos con ruedas

plano("motor"):
  [carro, moto, avion, barco, tractor]
  → taxonomía de vehículos motorizados

plano("agua"):
  [barco, pez, ballena, submarino]
  → taxonomía de conceptos acuáticos

cada primitivo es un criterio de clasificación
la taxonomía emerge automáticamente
sin árbol definido manualmente              ✅

# ¿qué tiene ruedas Y motor pero NO cuatro?
interseccion(["ruedas", "motor"]) → [carro, moto, camion]
excluir(["cuatro"])               → [moto]              ✅

# ¿qué conceptos son similares a carro pero sin motor?
vecinos("carro") → [moto, camion, bicicleta]
filtrar sin "motor" → [bicicleta]               ✅

# razonamiento sin grafo, sin reglas, sin LLM
# solo álgebra sobre la matriz M
```

---

### 5. Sin degradación jerárquica
```
SISTEMA CON REFERENCIAS CRUZADAS
──────────────────────────────────────────────────
carro    → ["vehiculo", "cuatro_ruedas", "motor"]
vehiculo → ["objeto",   "movil",         "transporta"]
objeto   → ["existente","perceptible"]

centroide(carro) usa centroide(vehiculo) usa centroide(objeto)
→ señal se diluye en cada nivel   ❌

SISTEMA CON PRIMITIVOS PUROS
──────────────────────────────────────────────────
carro → ["existente", "movil", "cuatro", "ruedas", "motor"]

un solo nivel — sin degradación
la señal es siempre directa       ✅
```

---

### 6. Actualización incremental O(1)
```
agregar nuevo concepto:
  1. definir con primitivos conocidos    O(P)
  2. agregar fila a la matriz M          O(P)
  3. coordenada disponible inmediatamente

agregar nuevo primitivo:
  1. agregar columna a la matriz M       O(C)
  2. recalcular SVD para proyección 3D   O(C·P²)
     solo si quieres visualizar

el espacio semántico crece sin reentrenar
sin tocar los conceptos existentes        ✅
```

---

### 7. Proyección 3D — visualización real del espacio semántico
```
SVD sobre M (C × P) → (C × 3)

cada concepto es un punto en 3D
conceptos similares → puntos cercanos
conceptos opuestos  → puntos lejanos

puedes VER la estructura semántica
no es una metáfora — es geometría real

clusters visibles:
  vehiculos agrupados en una región
  muebles agrupados en otra
  financiero en otra
  sin superposición si los primitivos son buenos






  # ── 1. construir el espacio semántico global — UNA SOLA VEZ ─────
es = EspacioSemantico.instancia()

es.definir("objeto",         ["existente",  "perceptible"])
es.definir("movil",          ["desplazable","energia"])
es.definir("vehiculo",       ["movil",      "transporta",  "personas"])
es.definir("carro",          ["vehiculo",   "cuatro",      "ruedas",   "motor"])
es.definir("moto",           ["vehiculo",   "dos",         "ruedas",   "motor"])
es.definir("banco_fin",      ["edificio",   "dinero",      "deposito", "credito"])
es.definir("banco_mueble",   ["mueble",     "asiento",     "madera",   "exterior"])

es.save()   # persiste el espacio completo
es.summary()

# ── 2. entrenar BANs — todas usan el mismo espacio ───────────────
ban1 = BAN()   # internamente: self._espacio = EspacioSemantico.instancia()
ban2 = BAN()   # misma referencia ✅
ban3 = BAN()   # misma referencia ✅

ban1.train_from_("carro.png",   "carro")      # usa firma del espacio
ban1.train_from_("moto.png",    "moto")       # usa firma del espacio

ban2.train_from_upstream_("carro.png", "carro",    upstream=ban1)
ban2.train_from_upstream_("moto.png",  "moto",     upstream=ban1)

ban3.train_from_upstream_("carro.png", "carro",    upstream=[ban1, ban2])

# ── 3. clasificar — el espacio da contexto semántico ─────────────
firma_query  = ban1._forward(_preprocess("query.png"))
concepto, _  = es.clasificar_firma(firma_query)
# el concepto activo desambigua la clasificación en ban2 y ban3

# ── 4. inferencia directa sobre el espacio ───────────────────────
es.interseccion(["ruedas", "motor"])          # → ["carro", "moto"]
es.excluir(["ruedas", "motor"], ["cuatro"])   # → ["moto"]
es.distancia("carro", "banco_fin")            # → 2.83
```

---

### Por qué es correcto que sea singleton
```
OPCIÓN A — espacio dentro de cada BAN
  BAN_1._espacio = EspacioSemantico()   copia
  BAN_2._espacio = EspacioSemantico()   copia diferente
  → inconsistencia si una actualiza y la otra no  ❌
  → duplicación de memoria                         ❌

OPCIÓN B — espacio global singleton
  BAN_1._espacio → EspacioSemantico.instancia()
  BAN_2._espacio → EspacioSemantico.instancia()
  BAN_3._espacio → EspacioSemantico.instancia()
  → todos ven exactamente el mismo espacio         ✅
  → actualizar el espacio afecta a todas las BANs  ✅
  → un solo .pkl para el espacio                   ✅
  → BANs persisten solo W_fwd y W_back             ✅
```

---

### Persistencia separada — dos archivos
```
models/
  espacio_semantico.pkl    ← EspacioSemantico singleton
                              diccionario, primitivos,
                              firmas, matriz M

  ban_1.pkl                ← solo W_fwd, W_back, labels
  ban_2.pkl                ← solo W_fwd, W_back, labels
  ban_3.pkl                ← solo W_fwd, W_back, labels

al cargar:
  EspacioSemantico.load("models/espacio_semantico.pkl")
  ban1 = BAN.load("models/ban_1.pkl")
  # ban1._espacio apunta automáticamente al singleton restaurado ✅
```

---

### Conclusión
```
EspacioSemantico.instancia()
  → existe una sola vez en memoria
  → todas las BANs lo referencian
  → definir un concepto nuevo lo hace visible a todas
  → persiste en un archivo independiente
  → las BANs persisten solo sus pesos W_fwd y W_back

el espacio semántico ES la inteligencia compartida
las BANs SON los patrones perceptuales específicos
la separación es limpia, sin duplicación,
sin inconsistencia posible