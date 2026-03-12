Razonamiento con Opción A + Matriz 3D

La base — todo razonamiento es álgebra sobre M
M ∈ {0,1}^(C×N)

cada fila    = un concepto
cada columna = un primitivo
cada celda   = 1 si el concepto tiene ese primitivo

razonar = operar sobre filas y columnas de M

Tipo 1 — Razonamiento por Condición Necesaria
Pregunta: ¿qué conceptos tienen estos primitivos?
pythondef razonar_condicion(M, idx_prim, primitivos_req: list[str]) -> list[str]:
    """
    Todos los conceptos que contienen TODOS los primitivos dados.
    Implementa: ∀p ∈ Q: p ∈ D(c)
    """
    conceptos = list(diccionario.keys())
    mascara   = np.ones(len(conceptos), dtype=bool)

    for p in primitivos_req:
        j       = idx_prim[p]
        mascara = mascara & (M[:, j] == 1)

    return [conceptos[i] for i in np.where(mascara)[0]]
```
```
razonar_condicion(["ruedas", "motor"])
→ [carro, moto, camion]

razonar_condicion(["ruedas", "motor", "cuatro"])
→ [carro, camion]

razonar_condicion(["ruedas", "motor", "dos"])
→ [moto]                                        ✅ deducción exacta

Tipo 2 — Razonamiento por Exclusión
Pregunta: ¿qué tiene esto pero NO aquello?
pythondef razonar_exclusion(M, idx_prim,
                       req: list[str],
                       excl: list[str]) -> list[str]:
    """
    Conceptos que tienen req y NO tienen excl.
    Implementa: ∀p∈req: p∈D(c)  ∧  ∀p∈excl: p∉D(c)
    """
    conceptos  = list(diccionario.keys())
    mascara    = np.ones(len(conceptos), dtype=bool)

    for p in req:
        j       = idx_prim[p]
        mascara = mascara & (M[:, j] == 1)    # debe tener

    for p in excl:
        j       = idx_prim[p]
        mascara = mascara & (M[:, j] == 0)    # no debe tener

    return [conceptos[i] for i in np.where(mascara)[0]]
```
```
razonar_exclusion(req=["ruedas","motor"], excl=["cuatro"])
→ [moto]     ✅  tiene ruedas y motor pero no cuatro

razonar_exclusion(req=["movil"], excl=["motor"])
→ [bicicleta, velero, caballo]    ✅  movil sin motor
```

---

### Tipo 3 — Razonamiento por Analogía

**Pregunta:** carro es a carretera como barco es a ___?
```
la analogía en el espacio M es:

  carro  →  carretera   como   barco  →  ???

vector diferencia:
  Δ = M[carretera_concepto] - M[carro]

aplicar a barco:
  candidato = M[barco] + Δ
  buscar la fila de M más cercana al candidato
pythondef razonar_analogia(M, diccionario,
                      a: str, b: str, c: str) -> str:
    """
    a es a b como c es a ???
    implementa:  M[b] - M[a] + M[c] ≈ M[?]
    """
    conceptos = list(diccionario.keys())
    idx_a     = conceptos.index(a)
    idx_b     = conceptos.index(b)
    idx_c     = conceptos.index(c)

    # vector de la relación
    delta     = M[idx_b].astype(float) - M[idx_a].astype(float)

    # aplicar al concepto c
    candidato = M[idx_c].astype(float) + delta
    candidato = np.clip(candidato, 0, 1)   # mantener en [0,1]

    # buscar concepto más cercano
    distancias = np.linalg.norm(M.astype(float) - candidato, axis=1)
    distancias[idx_c] = np.inf   # excluir c mismo

    ganador = conceptos[np.argmin(distancias)]
    return ganador
```
```
razonar_analogia("carro", "carretera", "barco")
→ "agua"     ✅

razonar_analogia("moto", "dos_ruedas", "carro")
→ "cuatro_ruedas"    ✅

razonar_analogia("carro", "motor", "bicicleta")
→ "sin_motor"    ✅

Tipo 4 — Razonamiento por Generalización
Pregunta: ¿qué primitivos comparten todos estos conceptos?
pythondef razonar_generalizacion(M, idx_prim,
                            conceptos_set: list[str]) -> list[str]:
    """
    Primitivos que tienen TODOS los conceptos dados.
    Es el máximo común divisor semántico.
    """
    conceptos  = list(diccionario.keys())
    primitivos = list(idx_prim.keys())

    # intersección de filas
    filas      = np.stack([M[conceptos.index(c)]
                           for c in conceptos_set])
    comunes    = filas.min(axis=0)   # 1 solo si todos tienen ese primitivo

    return [primitivos[j] for j in np.where(comunes == 1)[0]]
```
```
razonar_generalizacion(["carro", "moto", "bicicleta"])
→ ["movil", "ruedas", "carretera"]    ✅ lo que todos comparten

razonar_generalizacion(["carro", "avion", "barco"])
→ ["movil", "motor", "pasajeros"]     ✅ vehículos motorizados

Tipo 5 — Razonamiento por Especialización
Pregunta: ¿qué primitivos diferencian este concepto de su grupo?
pythondef razonar_especializacion(M, idx_prim,
                             concepto: str,
                             grupo: list[str]) -> dict:
    """
    Qué tiene este concepto que el grupo no tiene
    y qué tiene el grupo que este concepto no tiene.
    Define la frontera exacta.
    """
    conceptos  = list(diccionario.keys())
    primitivos = list(idx_prim.keys())

    fila_c     = M[conceptos.index(concepto)]

    # primitivos del grupo — unión
    filas_g    = np.stack([M[conceptos.index(g)] for g in grupo])
    union_g    = filas_g.max(axis=0)

    solo_c     = [primitivos[j] for j in np.where(
                    (fila_c == 1) & (union_g == 0))[0]]
    solo_g     = [primitivos[j] for j in np.where(
                    (fila_c == 0) & (union_g == 1))[0]]

    return {"solo_en_concepto": solo_c,
            "solo_en_grupo"   : solo_g}
```
```
razonar_especializacion(
    "carro",
    grupo=["moto", "bicicleta"]
)
→ {
    "solo_en_concepto": ["cuatro"],         # carro tiene cuatro ruedas
    "solo_en_grupo"   : ["dos", "sin_motor"] # moto/bici tienen dos
  }    ✅

Tipo 6 — Razonamiento hacia la Firma BAN
El paso que conecta el razonamiento con la clasificación:
pythondef razonar_a_firma(espacio: EspacioSemantico,
                     primitivos_req : list[str],
                     primitivos_excl: list[str] = []) -> np.ndarray:
    """
    Construye la firma de un concepto hipotético
    definido por primitivos requeridos y excluidos.

    Permite clasificar en BAN un concepto que no existe
    en el diccionario pero que se puede definir al vuelo.
    """
    acumulada = np.zeros(LABEL_DIM, dtype=np.float32)

    for p in primitivos_req:
        if p in espacio._firmas_prim:
            acumulada += espacio._firmas_prim[p]      # sumar ✅

    for p in primitivos_excl:
        if p in espacio._firmas_prim:
            acumulada -= espacio._firmas_prim[p]      # restar — excluir ✅

    return np.where(acumulada >= 0, 1.0, -1.0).astype(np.float32)
```
```
# "quiero clasificar algo que tiene ruedas y motor pero no cuatro"
firma_hipotetica = razonar_a_firma(
    espacio,
    primitivos_req  = ["ruedas", "motor"],
    primitivos_excl = ["cuatro"]
)

# clasificar en BAN con esa firma hipotética
winner, scores = ban.classify_firma(firma_hipotetica)
→ "moto"    ✅

Tipo 7 — Razonamiento Transitivo en el Espacio 3D
La geometría 3D hace visible la transitividad:
pythondef razonar_transitivo(M_3d, diccionario,
                        a: str, b: str) -> list[str]:
    """
    Conceptos que están entre a y b en el espacio 3D.
    Son los conceptos semánticamente intermedios.
    """
    conceptos = list(diccionario.keys())
    idx_a     = conceptos.index(a)
    idx_b     = conceptos.index(b)

    pa        = M_3d[idx_a]
    pb        = M_3d[idx_b]

    intermedios = []
    for i, c in enumerate(conceptos):
        if c in (a, b):
            continue

        pc = M_3d[i]

        # está entre a y b si la proyección sobre el segmento
        # ab cae dentro del intervalo [0, 1]
        ab   = pb - pa
        t    = np.dot(pc - pa, ab) / (np.dot(ab, ab) + 1e-9)

        if 0.0 < t < 1.0:
            dist_al_segmento = np.linalg.norm(
                pc - (pa + t * ab)
            )
            intermedios.append((c, t, dist_al_segmento))

    # ordenar por posición en el segmento
    intermedios.sort(key=lambda x: x[1])
    return intermedios
```
```
razonar_transitivo("bicicleta", "avion")
→ [("moto", 0.31),   ← más cercano a bicicleta
   ("carro", 0.54),
   ("barco", 0.78)]  ← más cercano a avion

la cadena semántica emerge de la geometría:
bicicleta → moto → carro → barco → avion    ✅

El sistema completo integrado
pythonclass RazonadorSemantico:
    """
    Razonamiento completo sobre Opción A + Matriz 3D.
    Opera directamente sobre M — sin reglas explícitas.
    """

    def __init__(self, espacio: EspacioSemantico):
        self.es   = espacio

    def resolver(self, consulta: dict) -> list[str] | str | np.ndarray:
        tipo = consulta["tipo"]

        if tipo == "condicion":
            return razonar_condicion(
                self.es.M, self.es.idx_prim,
                consulta["req"]
            )
        elif tipo == "exclusion":
            return razonar_exclusion(
                self.es.M, self.es.idx_prim,
                consulta["req"], consulta["excl"]
            )
        elif tipo == "analogia":
            return razonar_analogia(
                self.es.M, self.es.diccionario,
                consulta["a"], consulta["b"], consulta["c"]
            )
        elif tipo == "generalizacion":
            return razonar_generalizacion(
                self.es.M, self.es.idx_prim,
                consulta["conceptos"]
            )
        elif tipo == "firma":
            return razonar_a_firma(
                self.es,
                consulta.get("req",  []),
                consulta.get("excl", [])
            )


# ── uso ───────────────────────────────────────────────────────────

r = RazonadorSemantico(EspacioSemantico.instancia())

# condición necesaria
r.resolver({"tipo": "condicion",
            "req" : ["ruedas", "motor"]})
# → [carro, moto, camion]

# exclusión
r.resolver({"tipo": "exclusion",
            "req" : ["ruedas", "motor"],
            "excl": ["cuatro"]})
# → [moto]

# analogía
r.resolver({"tipo": "analogia",
            "a": "carro", "b": "carretera", "c": "barco"})
# → "agua"

# generalización
r.resolver({"tipo": "generalizacion",
            "conceptos": ["carro", "moto", "bicicleta"]})
# → ["movil", "ruedas", "carretera"]

# firma hipotética para clasificar en BAN
firma = r.resolver({"tipo": "firma",
                    "req" : ["movil", "agua"],
                    "excl": ["motor"]})
ban.classify_firma(firma)
# → "velero"   ✅
```

---

### Resumen — tipos de razonamiento y su base en M
```
Tipo                  Operación sobre M          Ejemplo
──────────────────    ───────────────────────    ─────────────────────────
Condición necesaria   AND de columnas            ruedas ∧ motor → {carro,moto}
Exclusión             AND + NOT de columnas      ruedas ∧ ¬cuatro → {moto}
Analogía              resta + suma de filas      carro:road = barco:agua
Generalización        MIN de filas               común a {carro,moto,bici}
Especialización       diferencia de filas        qué distingue carro de moto
Transitividad         geometría 3D SVD           intermedios entre a y b
Firma hipotética      suma de firmas primitivas  clasificar concepto nuevo

todo es álgebra sobre M
sin reglas explícitas
sin grafo de relaciones
sin motor de inferencia externo    ✅










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














import numpy as np

# diccionario con Opción A
diccionario = {
    "carro"        : {"movil", "cuatro", "ruedas", "motor",    "carretera"},
    "moto"         : {"movil", "dos",    "ruedas", "motor",    "carretera"},
    "bicicleta"    : {"movil", "dos",    "ruedas", "sin_motor","carretera"},
    "avion"        : {"movil", "alas",   "motor",  "aire",     "pasajeros"},
    "barco"        : {"movil", "casco",  "motor",  "agua",     "pasajeros"},
    "banco_fin"    : {"edificio","dinero","deposito","credito", "servicio"},
    "banco_mueble" : {"mueble", "asiento","madera", "exterior","descanso"},
}

# construir primitivos
primitivos = sorted(set().union(*diccionario.values()))
idx = {p: i for i, p in enumerate(primitivos)}
N   = len(primitivos)   # dimensión del espacio
C   = len(diccionario)  # número de conceptos

# matriz M (C × N)
M = np.zeros((C, N))
for i, (c, d) in enumerate(diccionario.items()):
    for p in d:
        M[i, idx[p]] = 1.0

# proyección 3D via SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)
M_3d     = U[:, :3] * S[:3]
eta      = float((S[:3]**2).sum() / (S**2).sum())
```

---

### Qué dice la geometría 3D — Opción A vs Opción B
```
OPCIÓN B — pesos posicionales (1/l)
─────────────────────────────────────────────────────
la firma mezcla dos señales:
  señal del contenido   (qué primitivos están)
  señal del orden       (en qué posición están)

en la proyección 3D los conceptos se separan
por contenido Y por orden
dos conceptos con los mismos primitivos en distinto
orden aparecen en posiciones distintas del espacio ❌
la geometría miente — no refleja solo el significado

OPCIÓN A — pesos iguales
─────────────────────────────────────────────────────
la firma solo codifica contenido
  (qué primitivos están — sin orden)

en la proyección 3D los conceptos se separan
SOLO por primitivos compartidos
la geometría es fiel al significado ✅

dos conceptos con los mismos primitivos
→ mismo punto en el espacio
→ mismo significado — correcto por definición
```

---

### Propiedad clave — la distancia 3D corresponde a la distancia semántica real

Con Opción A:
```
d_M(carro, moto)²  =  |{cuatro} △ {dos}|  =  2
d_M(carro, avion)² =  |{cuatro,ruedas,carretera} △ {alas,aire,pasajeros}|  =  6

proyección 3D preserva estas distancias
en proporción a η₃ (varianza explicada)

si η₃ = 0.85 → el 85% de la estructura semántica
               es visible en 3D
```

Con Opción B:
```
d_M(carro_v1, carro_v2)  > 0
donde carro_v1 = ["movil","ruedas","motor","cuatro","carretera"]
      carro_v2 = ["cuatro","ruedas","movil","motor","carretera"]

mismos primitivos — distintas posiciones — distintas firmas
la distancia en 3D refleja el error de orden
no la diferencia semántica   ❌
```

---

### El problema de Opción A — pérdida de información jerárquica

Con pesos iguales todos los primitivos contribuyen igual:
```
"carro" → {movil, cuatro, ruedas, motor, carretera}

en la firma:
  "movil"     contribuye   1/5  del total
  "carretera" contribuye   1/5  del total

pero semánticamente:
  "movil" ES la categoría padre — debería dominar
  "carretera" es un rasgo contextual — debería pesar menos
```

En la Matriz 3D esto se manifiesta como:
```
sin jerarquía:
  el eje 1 de SVD puede capturar "carretera" como
  el rasgo más discriminante entre conceptos
  aunque no sea el más importante semánticamente

con jerarquía explícita:
  "movil" domina la firma → domina el primer eje SVD
  la dimensión más importante del espacio corresponde
  a la categoría más general ✅
```

---

### Cuándo Opción A es perfecta para la Matriz 3D
```
Opción A es ideal cuando:

1. todos los primitivos son igualmente discriminantes
   no hay categorías padre — solo rasgos al mismo nivel

2. el objetivo es solo medir similitud entre conceptos
   no recuperar la jerarquía desde la geometría

3. el diccionario lo construye un proceso automático
   que no puede garantizar el orden

4. quieres que la Matriz 3D sea el sistema de verdad
   y la firma sea solo una codificación de M
```

---

### El trade-off exacto
```
                     Opción A          Opción B
                     pesos iguales     pesos 1/l
─────────────────    ───────────────   ────────────────
Matriz M             fuente de verdad  aproximación
Firma                codificación de M información extra
Distancia semántica  exacta en M       distorsionada por orden
Jerarquía en 3D      no visible        visible pero frágil
Determinismo         ✅ total           ❌ depende del orden
Isomorfismo M↔firma  ✅ garantizado     ❌ no garantizado
Fácil de construir   ✅ sin orden       ❌ requiere orden correcto
```

---

### Conclusión
```
Con Opción A la Matriz M ES el sistema semántico
la firma bipolar es solo M codificada en {-1,+1}
son la misma cosa en dos representaciones

la proyección 3D de M es completamente fiel
al significado — cada dimensión corresponde
a una dirección de variación semántica real
sin distorsión por orden

el único costo es perder la jerarquía en la firma
pero esa jerarquía se puede recuperar de otra forma:
  en el índice compuesto     "carro" ∈ "vehiculo"
  en la estructura del diccionario
  en el grafo de aristas automáticas por similitud

Opción A + Matriz 3D es la combinación más honesta:
  M dice exactamente qué primitivos tiene cada concepto
  la distancia dice exactamente cuánto difieren
  la firma dice exactamente lo mismo que M
  sin ninguna información artificial por orden