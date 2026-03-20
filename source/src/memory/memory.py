"""
Memoria Asociativa Bidireccional (BAM v3)
=========================================
Mejoras implementadas:
  1. Actualización formal de firma entre ciclos (decay + recálculo)
  2. Agregación formal del OPERATOR con resolución de contradicciones
  3. Umbral de confianza — el sistema puede responder "no sé"
  4. x_diff corregido para patrones con prefijos compartidos
  5. Recall iterativo hasta convergencia (energía de Lyapunov)
  6. Enrutamiento graduado de BAMs por relevancia semántica
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


current_path = Path.cwd()

# ══════════════════════════════════════════════════════════════════════════════
#  Constantes
# ══════════════════════════════════════════════════════════════════════════════

N_LABEL           = 64      # bits para codificar el ID entero
MAX_ITER          = 50      # iteraciones máximas de convergencia BAM
CONFIDENCE_THRESH = 0.35    # score mínimo para que una respuesta sea válida
FIRMA_DECAY       = 0.85    # decaimiento de bits de firma entre ciclos
FIRMA_DECAY_MIN   = 0.10    # bit de firma se apaga si cae bajo este umbral
ROUTING_THRESH    = 0.20    # relevancia mínima para enrutar prompt a una BAM


# ══════════════════════════════════════════════════════════════════════════════
#  Estructuras de datos
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FirmaSemantica:
    """
    Firma semántica como mapa de bits ponderado.

    En lugar de {0,1} puro, cada concepto tiene un peso flotante [0,1].
    Esto permite:
      - Decaimiento gradual de conceptos distantes en el ciclo
      - Umbral de apagado (FIRMA_DECAY_MIN) para limpiar ruido
      - Operaciones de similitud continua (no solo AND binario)
    """
    bits: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def desde_binario(cls, bitmap: dict[str, int]) -> "FirmaSemantica":
        """Construye desde un diccionario {concepto: 0/1}."""
        return cls(bits={k: float(v) for k, v in bitmap.items()})

    # ------------------------------------------------------------------
    def overlap(self, otra: "FirmaSemantica") -> float:
        """
        Relevancia graduada entre dos firmas.
        overlap = |A ∩ B| / |A|   (fracción de conceptos de A cubiertos por B)
        Devuelve 0.0 si A está vacía.
        """
        conceptos_A = {k for k, v in self.bits.items() if v > FIRMA_DECAY_MIN}
        if not conceptos_A:
            return 0.0
        suma = sum(
            min(self.bits[k], otra.bits.get(k, 0.0))
            for k in conceptos_A
        )
        return suma / len(conceptos_A)

    # ------------------------------------------------------------------
    def decaer(self, decay: float = FIRMA_DECAY) -> "FirmaSemantica":
        """
        Aplica decaimiento exponencial y apaga bits bajo FIRMA_DECAY_MIN.
        Usado entre ciclos del OPERATOR para modelar foco cambiante.
        """
        nuevos = {
            k: v * decay
            for k, v in self.bits.items()
            if v * decay > FIRMA_DECAY_MIN
        }
        return FirmaSemantica(bits=nuevos)

    # ------------------------------------------------------------------
    def fusionar(self, otra: "FirmaSemantica", peso: float = 1.0) -> "FirmaSemantica":
        """
        Fusiona dos firmas ponderando la nueva con `peso`.
        Usado para incorporar firma de la palabra recién generada.
        """
        resultado = dict(self.bits)
        for k, v in otra.bits.items():
            resultado[k] = resultado.get(k, 0.0) + v * peso
        # renormalizar a [0, 1]
        max_val = max(resultado.values(), default=1.0)
        return FirmaSemantica(bits={k: v / max_val for k, v in resultado.items()})

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        activos = {k: round(v, 2) for k, v in self.bits.items()
                   if v > FIRMA_DECAY_MIN}
        return f"Firma({activos})"


@dataclass
class RespuestaBAM:
    """Resultado que una BAM devuelve al OPERATOR."""
    label:      str
    score:      float
    firma:      FirmaSemantica
    bam_id:     str
    confiable:  bool          # score >= CONFIDENCE_THRESH
    votos:      int = 0


@dataclass
class ResultadoCiclo:
    """Lo que el OPERATOR devuelve en cada iteración."""
    palabra:        Optional[str]     # None si no hay respuesta confiable
    score_agregado: float
    respuestas:     list[RespuestaBAM]
    firma_nueva:    FirmaSemantica
    convergio:      bool


# ══════════════════════════════════════════════════════════════════════════════
#  Funciones de codificación / decodificación
# ══════════════════════════════════════════════════════════════════════════════

def image_to_binary(img_array: np.ndarray) -> np.ndarray:
    """
    Convierte imagen (n×n) → vector BINARIO float32.
    Negro=0 (dispersión), Blanco=1.
    """
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(float)
    if gray.max() <= 1.0:
        gray = gray * 255.0
    return (gray >= 128).astype(np.float32).flatten()


def binary_to_image(height: int, width: int, vec: np.ndarray) -> np.ndarray:
    binary = (vec > 0).reshape(height, width).astype(np.float32)
    return (binary * 255).astype(np.uint8)


def id_to_bipolar(label_id: int) -> np.ndarray:
    bits = [1 if (label_id >> b) & 1 else -1 for b in range(N_LABEL - 1, -1, -1)]
    return np.array(bits, dtype=np.float32)


def bipolar_to_id(vec: np.ndarray) -> int:
    binary = ((np.sign(vec) + 1) / 2).astype(int)
    return int(''.join(binary.astype(str)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Clase BAM v3
# ══════════════════════════════════════════════════════════════════════════════

class BAM:
    """
    Memoria Asociativa Bidireccional con soporte de firma semántica.

    Cada patrón almacena su firma de párrafo, permitiendo al OPERATOR
    filtrar y ponderar recuperaciones por contexto semántico.
    """

    def __init__(self, bam_id: str, total_signs: int, sign_size_px: int):
        self.bam_id      = bam_id
        self.IMG_WIDTH   = sign_size_px * total_signs
        self.IMG_HEIGHT  = sign_size_px
        self.N_PIXELS    = self.IMG_WIDTH * self.IMG_HEIGHT

        self.total_signs  = total_signs
        self.sign_size_px = sign_size_px

        self._W_lil: lil_matrix      = lil_matrix((self.N_PIXELS, N_LABEL), dtype=np.float32)
        self._W_csr: csr_matrix|None = None
        self._dirty: bool            = True

        self.patterns:  list[dict]      = []
        self.label_map: dict[int, str]  = {}

        print(
            f"✅ BAM [{bam_id}] inicializada  |  "
            f"Capa A: {self.N_PIXELS} neuronas  |  "
            f"Capa B: {N_LABEL} neuronas"
        )

    # ------------------------------------------------------------------
    #  W property
    # ------------------------------------------------------------------
    @property
    def W(self) -> csr_matrix:
        if self._dirty:
            self._W_csr = self._W_lil.tocsr()
            self._dirty = False
        return self._W_csr

    # ------------------------------------------------------------------
    #  MEJORA 4: x_diff corregido para prefijos compartidos
    # ------------------------------------------------------------------
    def _calcular_x_diff(self, x_new: np.ndarray) -> np.ndarray:
        """
        Calcula la contribución única del nuevo patrón a W.

        CORRECCIÓN v3:
        En lugar de usar el acumulado máximo de TODOS los patrones,
        ponderamos cada patrón previo por su similitud coseno con x_new.
        
        Patrones muy similares (prefijos) aportan más al acumulado.
        Patrones diferentes aportan menos → sus píxeles no bloquean a x_new.

        Esto evita que "el banco cierra en la tarde" pierda los píxeles
        de "banco" solo porque "el banco abre en la mañana" los usó antes.
        """
        if not self.patterns:
            return x_new

        x_acum = np.zeros(self.N_PIXELS, dtype=np.float32)

        for p in self.patterns:
            norm = np.linalg.norm(x_new) * np.linalg.norm(p['x'])
            if norm < 1e-9:
                continue
            similitud = float(np.dot(x_new, p['x']) / norm)
            # solo patrones muy similares bloquean píxeles (similitud > 0.5)
            if similitud > 0.5:
                x_acum = np.maximum(x_acum, p['x'] * similitud)

        x_diff = x_new * (1.0 - np.clip(x_acum, 0.0, 1.0))
        return x_diff

    # ------------------------------------------------------------------
    #  Aprendizaje incremental
    # ------------------------------------------------------------------
    def learn_incremental(
        self,
        image:    np.ndarray,
        label_str: str,
        firma:    FirmaSemantica,
    ) -> None:
        label_id = len(self.patterns)
        self.label_map[label_id] = label_str

        x_new  = image_to_binary(image)
        x_diff = self._calcular_x_diff(x_new)      # MEJORA 4

        if x_diff.sum() == 0:
            print(f"⚠️  '{label_str}' no aporta píxeles únicos — patrón ignorado")
            return

        y = id_to_bipolar(label_id)
        white_pixels = np.nonzero(x_diff)[0]
        self._W_lil[white_pixels, :] += y[np.newaxis, :]
        self._dirty = True

        self.patterns.append({
            'x':      x_new,
            'x_diff': x_diff,
            'y':      y,
            'id':     label_id,
            'label':  label_str,
            'firma':  firma,
            'n_white_new': int(x_diff.sum()),
        })

        print(
            f"[{self.bam_id}][{label_id}] '{label_str}'  |  "
            f"Píxeles nuevos: {int(x_diff.sum())}  |  "
            f"Firma: {firma}"
        )

    # ------------------------------------------------------------------
    #  MEJORA 5: Recall iterativo con energía de Lyapunov
    # ------------------------------------------------------------------
    def recall_label(
        self,
        image: np.ndarray,
        firma_query: Optional[FirmaSemantica] = None,
    ) -> tuple[str, int, np.ndarray, float]:
        """
        Recupera el label asociado a una imagen.

        v3: itera hasta convergencia verificando la energía de Lyapunov.
        E = -x^T W y   (debe decrecer o estabilizarse)

        Retorna: (label_str, label_id, y_final, energia_final)
        """
        x = image_to_binary(image)
        W = self.W

        # aplicar máscara de firma si se provee
        if firma_query is not None:
            x = self._aplicar_mascara_firma(x, firma_query)

        y = np.sign(W.T @ x)
        y[y == 0] = 1.0

        energia_prev = -float(x @ (W @ y))

        for iteration in range(MAX_ITER):
            # paso A→B
            y_new = np.sign(W.T @ x)
            y_new[y_new == 0] = 1.0

            # paso B→A
            x_new = np.sign(W @ y_new)
            x_new[x_new == 0] = 0.0   # binario: negro=0

            energia = -float(x_new @ (W @ y_new))

            # convergencia: estado no cambia o energía estabilizada
            if np.array_equal(y_new, y) and np.array_equal(x_new, x):
                break
            if abs(energia - energia_prev) < 1e-6:
                break

            y           = y_new
            x           = x_new
            energia_prev = energia

        label_id  = bipolar_to_id(y)
        label_str = self.label_map.get(label_id, f"<ID {label_id} desconocido>")
        return label_str, label_id, y, energia_prev

    # ------------------------------------------------------------------
    #  Recall ranking con filtro de firma
    # ------------------------------------------------------------------
    def recall_ranking(
        self,
        image:       np.ndarray,
        firma_query: Optional[FirmaSemantica] = None,
        top_k:       int = 5,
    ) -> list[RespuestaBAM]:
        """
        Ranking de patrones por similitud coseno.

        v3: filtra patrones por overlap de firma antes de calcular scores.
        Si firma_query es None, evalúa todos los patrones.
        """
        x = image_to_binary(image)

        ranking = []
        for p in self.patterns:
            # filtro semántico graduado
            peso_firma = 1.0
            if firma_query is not None:
                peso_firma = firma_query.overlap(p['firma'])
                if peso_firma < ROUTING_THRESH:
                    continue   # patrón fuera de contexto semántico

            score = self._similitud_coseno(x, p['x']) * peso_firma
            votos = int(np.dot(x, p['x']))

            ranking.append(RespuestaBAM(
                label     = p['label'],
                score     = score,
                firma     = p['firma'],
                bam_id    = self.bam_id,
                confiable = score >= CONFIDENCE_THRESH,
                votos     = votos,
            ))

        ranking.sort(key=lambda r: r.score, reverse=True)
        return ranking[:top_k]

    # ------------------------------------------------------------------
    #  Auxiliares
    # ------------------------------------------------------------------
    def _similitud_coseno(self, x1: np.ndarray, x2: np.ndarray) -> float:
        norm = np.linalg.norm(x1) * np.linalg.norm(x2)
        return float(np.dot(x1, x2) / (norm + 1e-9))

    def _aplicar_mascara_firma(
        self,
        x:           np.ndarray,
        firma_query: FirmaSemantica,
    ) -> np.ndarray:
        """
        Atenúa píxeles de patrones con firma incompatible.
        Permite que el recall iterativo converja al mínimo correcto.
        """
        mascara = np.zeros(self.N_PIXELS, dtype=np.float32)
        for p in self.patterns:
            overlap = firma_query.overlap(p['firma'])
            if overlap > ROUTING_THRESH:
                mascara = np.maximum(mascara, p['x'] * overlap)
        return x * np.clip(mascara, 0.0, 1.0) + x * 0.1  # base mínima

    def flush(self):
        _ = self.W
        del self._W_lil
        for p in self.patterns:
            del p['x_diff']
            del p['y']
            del p['n_white_new']


# ══════════════════════════════════════════════════════════════════════════════
#  OPERATOR v3
# ══════════════════════════════════════════════════════════════════════════════

class Operator:
    """
    Coordina múltiples BAMs para generación autoregresiva.

    Mejoras v3:
      1. Enrutamiento graduado por relevancia de firma
      2. Agregación formal con resolución de contradicciones
      3. Umbral de confianza global
      4. Actualización de firma entre ciclos (decay + fusión)
    """

    def __init__(self, bams: list[BAM]):
        self.bams = {b.bam_id: b for b in bams}
        print(f"✅ OPERATOR inicializado con {len(bams)} BAMs: {list(self.bams.keys())}")

    # ------------------------------------------------------------------
    #  MEJORA 6: Enrutamiento graduado
    # ------------------------------------------------------------------
    def _enrutar(
        self,
        firma_prompt: FirmaSemantica,
    ) -> list[tuple[BAM, float]]:
        """
        Selecciona BAMs relevantes con peso de relevancia.

        Devuelve lista ordenada de (BAM, relevancia) con relevancia > ROUTING_THRESH.
        """
        resultado = []
        for bam in self.bams.values():
            # firma de la BAM = unión de todas sus firmas de patrones
            firma_bam = self._firma_agregada_bam(bam)
            relevancia = firma_prompt.overlap(firma_bam)
            if relevancia >= ROUTING_THRESH:
                resultado.append((bam, relevancia))

        resultado.sort(key=lambda t: t[1], reverse=True)
        return resultado

    def _firma_agregada_bam(self, bam: BAM) -> FirmaSemantica:
        """Firma de la BAM = máximo de cada bit en todos sus patrones."""
        agregada: dict[str, float] = {}
        for p in bam.patterns:
            for concepto, valor in p['firma'].bits.items():
                agregada[concepto] = max(agregada.get(concepto, 0.0), valor)
        return FirmaSemantica(bits=agregada)

    # ------------------------------------------------------------------
    #  MEJORA 2: Agregación formal con resolución de contradicciones
    # ------------------------------------------------------------------
    def _agregar_respuestas(
        self,
        respuestas_por_bam: dict[str, list[RespuestaBAM]],
        relevancia_bam:     dict[str, float],
    ) -> dict[str, float]:
        """
        Agrega scores de múltiples BAMs para cada candidato de palabra.

        Algoritmo:
          score_total(palabra) = Σ_BAM (score_BAM(palabra) × relevancia_BAM)
                                 - penalización_contradiccion

        Contradicción: dos BAMs con alta confianza proponen palabras distintas
        con scores muy similares → se penalizan ambas.
        """
        acumulado: dict[str, float] = {}
        votos_acumulado: dict[str, int] = {}

        for bam_id, respuestas in respuestas_por_bam.items():
            rel = relevancia_bam.get(bam_id, 1.0)
            for r in respuestas:
                if not r.confiable:
                    continue
                acumulado[r.label]        = acumulado.get(r.label, 0.0) + r.score * rel
                votos_acumulado[r.label]  = votos_acumulado.get(r.label, 0) + r.votos

        if not acumulado:
            return {}

        # detectar y penalizar contradicciones
        scores_ordenados = sorted(acumulado.values(), reverse=True)
        if len(scores_ordenados) >= 2:
            top1, top2 = scores_ordenados[0], scores_ordenados[1]
            # si los dos mejores están muy cerca → contradicción
            if top1 > 0 and (top2 / top1) > 0.85:
                # penalizar ambos candidatos principales
                candidatos_top = [k for k, v in acumulado.items()
                                  if v >= top2 * 0.85]
                for c in candidatos_top:
                    acumulado[c] *= 0.7   # reducción por contradicción
                print(f"⚠️  Contradicción detectada entre: {candidatos_top} — scores penalizados")

        return acumulado

    # ------------------------------------------------------------------
    #  MEJORA 1: Actualización de firma entre ciclos
    # ------------------------------------------------------------------
    def _actualizar_firma(
        self,
        firma_actual:   FirmaSemantica,
        palabra_nueva:  str,
        firma_ganadora: FirmaSemantica,
        peso_nueva:     float = 0.3,
    ) -> FirmaSemantica:
        """
        Actualiza la firma del ciclo siguiente.

        Estrategia:
          1. Decaer firma actual (conceptos distantes pierden peso)
          2. Fusionar con firma del patrón ganador (refuerza contexto relevante)
          3. El resultado refleja el foco semántico acumulado

        Esto evita el colapso a un único contexto — el decaimiento
        mantiene abierta la posibilidad de reactivar conceptos dormidos.
        """
        firma_decaida = firma_actual.decaer(FIRMA_DECAY)
        firma_nueva   = firma_decaida.fusionar(firma_ganadora, peso=peso_nueva)
        return firma_nueva

    # ------------------------------------------------------------------
    #  MEJORA 3: Umbral de confianza global
    # ------------------------------------------------------------------
    def _tiene_confianza(self, scores: dict[str, float]) -> bool:
        if not scores:
            return False
        return max(scores.values()) >= CONFIDENCE_THRESH

    # ------------------------------------------------------------------
    #  Ciclo principal de generación
    # ------------------------------------------------------------------
    def ciclo(
        self,
        imagen_query: np.ndarray,
        firma_prompt: FirmaSemantica,
    ) -> ResultadoCiclo:
        """
        Ejecuta un ciclo de generación:
          1. Enruta a BAMs relevantes (graduado)
          2. Cada BAM devuelve su ranking filtrado por firma
          3. OPERATOR agrega y resuelve contradicciones
          4. Verifica umbral de confianza
          5. Actualiza firma para el siguiente ciclo
        """
        # --- paso 1: enrutamiento graduado ---
        bams_relevantes = self._enrutar(firma_prompt)

        if not bams_relevantes:
            print("❌ Ninguna BAM relevante para la firma del prompt")
            return ResultadoCiclo(
                palabra        = None,
                score_agregado = 0.0,
                respuestas     = [],
                firma_nueva    = firma_prompt.decaer(),
                convergio      = False,
            )

        # --- paso 2: consultar cada BAM ---
        respuestas_por_bam: dict[str, list[RespuestaBAM]] = {}
        relevancia_bam:     dict[str, float]              = {}
        todas_respuestas:   list[RespuestaBAM]            = []

        for bam, relevancia in bams_relevantes:
            ranking = bam.recall_ranking(
                image       = imagen_query,
                firma_query = firma_prompt,
                top_k       = 5,
            )
            respuestas_por_bam[bam.bam_id] = ranking
            relevancia_bam[bam.bam_id]     = relevancia
            todas_respuestas.extend(ranking)

            print(
                f"  [{bam.bam_id}] relevancia={relevancia:.2f}  |  "
                f"top={ranking[0].label if ranking else 'ninguno'}"
            )

        # --- paso 3: agregación con resolución de contradicciones ---
        scores_agregados = self._agregar_respuestas(respuestas_por_bam, relevancia_bam)

        # --- paso 4: umbral de confianza ---
        if not self._tiene_confianza(scores_agregados):
            print("❌ Score insuficiente — el sistema no tiene respuesta confiable")
            return ResultadoCiclo(
                palabra        = None,
                score_agregado = max(scores_agregados.values(), default=0.0),
                respuestas     = todas_respuestas,
                firma_nueva    = firma_prompt.decaer(),
                convergio      = False,
            )

        # --- paso 5: elegir ganador ---
        palabra_ganadora = max(scores_agregados, key=scores_agregados.__getitem__)
        score_ganador    = scores_agregados[palabra_ganadora]

        # buscar firma del patrón ganador para actualizar contexto
        firma_ganadora = firma_prompt  # fallback
        for r in todas_respuestas:
            if r.label == palabra_ganadora:
                firma_ganadora = r.firma
                break

        # --- paso 6: actualizar firma ---
        firma_nueva = self._actualizar_firma(
            firma_actual   = firma_prompt,
            palabra_nueva  = palabra_ganadora,
            firma_ganadora = firma_ganadora,
        )

        print(
            f"✅ Ciclo completo  |  "
            f"palabra='{palabra_ganadora}'  |  "
            f"score={score_ganador:.3f}  |  "
            f"firma={firma_nueva}"
        )

        return ResultadoCiclo(
            palabra        = palabra_ganadora,
            score_agregado = score_ganador,
            respuestas     = todas_respuestas,
            firma_nueva    = firma_nueva,
            convergio      = True,
        )

    # ------------------------------------------------------------------
    #  Generación completa (múltiples ciclos)
    # ------------------------------------------------------------------
    def generar(
        self,
        imagen_inicial: np.ndarray,
        firma_inicial:  FirmaSemantica,
        max_palabras:   int = 20,
        verbose:        bool = True,
    ) -> list[str]:
        """
        Genera una secuencia de palabras de forma autoregresiva.

        En cada ciclo:
          - La firma se actualiza con decay + fusión del contexto ganador
          - Si el ciclo no converge, la generación se detiene
        """
        palabras  = []
        firma_t   = firma_inicial
        imagen_t  = imagen_inicial

        for t in range(max_palabras):
            if verbose:
                print(f"\n── Ciclo t={t} ──────────────────────────────")
                print(f"   Firma: {firma_t}")

            resultado = self.ciclo(imagen_t, firma_t)

            if not resultado.convergio or resultado.palabra is None:
                print(f"   Generación detenida en t={t}")
                break

            palabras.append(resultado.palabra)
            firma_t = resultado.firma_nueva

            # actualizar imagen_t concatenando la nueva palabra
            # (en implementación real: renderizar texto acumulado)
            # aquí lo dejamos igual para demostración
            # imagen_t = render(palabras)

        return palabras


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Ejemplo de uso ──────────────────────────────────────────────

    # Crear BAMs especializadas
    bam_finance  = BAM("finance",   total_signs=5, sign_size_px=32)
    bam_geo      = BAM("geografia", total_signs=5, sign_size_px=32)

    # Firmas de párrafo
    firma_f = FirmaSemantica.desde_binario({
        "banco_finance": 1, "dinero": 1, "horario": 1
    })
    firma_g = FirmaSemantica.desde_binario({
        "banco_river": 1, "agua": 1, "naturaleza": 1
    })

    # Crear OPERATOR
    op = Operator(bams=[bam_finance, bam_geo])

    # Prompt de ejemplo
    firma_prompt = FirmaSemantica.desde_binario({
        "banco_finance": 1, "mañana": 1, "banco_river": 1, "rio": 1
    })

    print("\nFirma del prompt:", firma_prompt)

    # Enrutar
    bams_sel = op._enrutar(firma_prompt)
    print(f"\nBAMs seleccionadas: {[(b.bam_id, round(r,2)) for b,r in bams_sel]}")

    # Actualización de firma simulada
    firma_t1 = op._actualizar_firma(
        firma_actual   = firma_prompt,
        palabra_nueva  = "abre",
        firma_ganadora = firma_f,
    )
    print(f"\nFirma t+1: {firma_t1}")

    firma_t2 = op._actualizar_firma(
        firma_actual   = firma_t1,
        palabra_nueva  = "en",
        firma_ganadora = firma_f,
    )
    print(f"Firma t+2: {firma_t2}")