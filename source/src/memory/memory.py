"""
Resonant Semantic Neuron (RSN) — Implementación completa
=========================================================
Arquitectura jerárquica de 5 niveles para LLM sin Transformers:

  Nivel 1: RSN          — neurona que aprende errores de predicción
  Nivel 2: MiniColumna  — 100 RSNs + inhibición lateral (1 concepto)
  Nivel 3: Columna      — N minicolumnas (1 dominio semántico)
  Nivel 4: Region       — N columnas (1 área de conocimiento)
  Nivel 5: Sistema      — N regiones + enrutamiento global

Principios:
  - Predictive Coding   : solo propaga SORPRESA, no la señal completa
  - Inhibición lateral  : solo 2-3 RSNs activas por minicolumna (SDR)
  - Firma semántica     : modula umbral θ de disparo (contexto top-down)
  - Bidireccionalidad   : feedback modifica umbrales, no solo pesos
  - Silencio informativo: predicción correcta → no emite señal → eficiencia
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# ══════════════════════════════════════════════════════════════════════════════
#  Constantes globales
# ══════════════════════════════════════════════════════════════════════════════

N_LABEL              = 64     # bits ID en BAM subyacente
MAX_ITER_BAM         = 50     # iteraciones BAM convergencia
CONFIDENCE_THRESH    = 0.35   # score mínimo respuesta confiable
FIRMA_DECAY          = 0.85   # decaimiento firma entre ciclos
FIRMA_DECAY_MIN      = 0.10   # umbral para apagar bit de firma
ROUTING_THRESH       = 0.20   # relevancia mínima de enrutamiento

# RSN
RSN_THETA_BASE       = 0.30   # umbral de disparo base
RSN_THETA_MIN        = 0.10   # umbral mínimo (con feedback fuerte)
RSN_THETA_MAX        = 0.90   # umbral máximo (inhibición fuerte)
RSN_LEARNING_RATE    = 0.05   # tasa de actualización de predicción
RSN_SURPRISE_THRESH  = 0.05   # error mínimo para considerar "sorpresa"

# MiniColumna
MINI_MAX_ACTIVE      = 3      # máx RSNs activas (inhibición lateral WTA)
MINI_INHIBIT_FACTOR  = 0.40   # cuánto sube θ de vecinos al ganar

# Columna
COL_MAX_MINI_ACTIVE  = 10     # máx minicolumnas activas por columna

# Region
REG_MAX_COL_ACTIVE   = 20     # máx columnas activas por región

# Sistema
SYS_FEEDBACK_CYCLES  = 3      # ciclos de feedback top-down por predicción


# ══════════════════════════════════════════════════════════════════════════════
#  FirmaSemantica  (igual que BAM v3, centralizada aquí)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FirmaSemantica:
    """
    Firma semántica ponderada. Cada concepto tiene peso [0, 1].
    Permite decaimiento gradual, fusión, y similitud continua.
    """
    bits: dict[str, float] = field(default_factory=dict)

    @classmethod
    def desde_binario(cls, bitmap: dict[str, int]) -> FirmaSemantica:
        return cls(bits={k: float(v) for k, v in bitmap.items()})

    @classmethod
    def vacia(cls) -> FirmaSemantica:
        return cls(bits={})

    def overlap(self, otra: FirmaSemantica) -> float:
        activos = {k for k, v in self.bits.items() if v > FIRMA_DECAY_MIN}
        if not activos:
            return 0.0
        return sum(min(self.bits[k], otra.bits.get(k, 0.0)) for k in activos) / len(activos)

    def decaer(self, decay: float = FIRMA_DECAY) -> FirmaSemantica:
        return FirmaSemantica(bits={
            k: v * decay for k, v in self.bits.items()
            if v * decay > FIRMA_DECAY_MIN
        })

    def fusionar(self, otra: FirmaSemantica, peso: float = 1.0) -> FirmaSemantica:
        r = dict(self.bits)
        for k, v in otra.bits.items():
            r[k] = r.get(k, 0.0) + v * peso
        mx = max(r.values(), default=1.0)
        if mx == 0.0:
            return FirmaSemantica(bits={})
        return FirmaSemantica(bits={k: v / mx for k, v in r.items()})

    def activos(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.bits.items() if v > FIRMA_DECAY_MIN}

    def __repr__(self) -> str:
        return f"Firma({self.activos()})"

    def __bool__(self) -> bool:
        return bool(self.activos())


# ══════════════════════════════════════════════════════════════════════════════
#  Estructuras de resultado
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpikeRSN:
    """Señal emitida por una RSN cuando la sorpresa supera el umbral."""
    rsn_id:     int
    sorpresa:   float          # magnitud del error de predicción
    prediccion: str            # qué predijo la RSN
    real:       str            # qué recibió
    firma:      FirmaSemantica
    peso:       float          # confianza acumulada de la RSN


@dataclass
class ActivacionMini:
    """Resultado de una MiniColumna tras procesar una entrada."""
    mini_id:       str
    concepto:      str
    spikes:        list[SpikeRSN]
    ganadores:     list[int]          # índices RSN que dispararon (WTA)
    sorpresa_max:  float
    prediccion:    str                # predicción consenso de la minicolumna
    confianza:     float
    firma:         FirmaSemantica


@dataclass
class ActivacionColumna:
    """Resultado de una Columna semántica."""
    col_id:        str
    activaciones:  list[ActivacionMini]   # solo minicolumnas activas
    prediccion:    str
    confianza:     float
    sorpresa_media: float
    firma_salida:  FirmaSemantica


@dataclass
class ActivacionRegion:
    """Resultado de una Region."""
    region_id:     str
    activaciones:  list[ActivacionColumna]
    prediccion:    str
    confianza:     float
    firma_salida:  FirmaSemantica


@dataclass
class ResultadoSistema:
    """Resultado final del Sistema para un ciclo de generación."""
    palabra:          Optional[str]
    confianza:        float
    sorpresa_global:  float
    activaciones:     list[ActivacionRegion]
    firma_nueva:      FirmaSemantica
    convergio:        bool
    tiempo_ms:        float


# ══════════════════════════════════════════════════════════════════════════════
#  NIVEL 1: RSN — Resonant Semantic Neuron
# ══════════════════════════════════════════════════════════════════════════════

class RSN:
    """
    Neurona fundamental del sistema. Aprende errores de predicción.

    Estado interno:
      _prediccion : qué espera ver en el siguiente paso
      _peso       : confianza acumulada (sube si predice bien)
      _theta      : umbral de disparo (modulado por feedback y competencia)
      _firma      : contexto semántico en que esta RSN se especializa

    Ciclo de vida por token:
      1. recibe (entrada, firma_contexto)
      2. calcula sorpresa = ||entrada - _prediccion||
      3. si sorpresa > theta → emite SpikeRSN
      4. actualiza _prediccion con learning_rate
      5. actualiza _peso y _theta
    """

    def __init__(self, rsn_id: int, firma_inicial: FirmaSemantica):
        self.rsn_id      = rsn_id
        self._prediccion = ""
        self._peso       = 0.5
        self._theta      = RSN_THETA_BASE
        self._firma      = firma_inicial
        self._n_spikes   = 0
        self._n_correctas = 0

    # ------------------------------------------------------------------
    #  Procesar entrada
    # ------------------------------------------------------------------
    def procesar(
        self,
        entrada:        str,
        firma_contexto: FirmaSemantica,
        feedback_theta: float = 0.0,
    ) -> Optional[SpikeRSN]:
        """
        Procesa una entrada y opcionalmente emite un spike.

        feedback_theta: ajuste de umbral desde capas superiores
                        negativo → más fácil disparar (contexto refuerza)
                        positivo → más difícil disparar (inhibición)
        """
        # modular umbral con feedback top-down y firma
        compatibilidad = self._firma.overlap(firma_contexto)
        theta_efectivo = self._theta + feedback_theta - (compatibilidad * 0.15)
        theta_efectivo = float(np.clip(theta_efectivo, RSN_THETA_MIN, RSN_THETA_MAX))

        # calcular sorpresa
        sorpresa = self._calcular_sorpresa(entrada)

        spike = None
        if sorpresa > theta_efectivo:
            spike = SpikeRSN(
                rsn_id     = self.rsn_id,
                sorpresa   = sorpresa,
                prediccion = self._prediccion,
                real       = entrada,
                firma      = self._firma,
                peso       = self._peso,
            )
            self._n_spikes += 1

        # aprender: actualizar predicción
        self._actualizar_prediccion(entrada, sorpresa)

        # actualizar confianza
        if sorpresa < RSN_SURPRISE_THRESH:
            self._n_correctas += 1
            self._peso = min(1.0, self._peso + RSN_LEARNING_RATE * 0.5)
        else:
            self._peso = max(0.0, self._peso - RSN_LEARNING_RATE * sorpresa)

        return spike

    # ------------------------------------------------------------------
    #  Actualizar umbral desde MiniColumna (inhibición lateral WTA)
    # ------------------------------------------------------------------
    def inhibir(self, factor: float = MINI_INHIBIT_FACTOR) -> None:
        """Sube θ — esta RSN perdió la competencia WTA."""
        self._theta = min(RSN_THETA_MAX, self._theta + factor * 0.3)

    def reforzar(self) -> None:
        """Baja θ — esta RSN ganó la competencia WTA."""
        self._theta = max(RSN_THETA_MIN, self._theta - 0.05)
        self._peso  = min(1.0, self._peso + RSN_LEARNING_RATE)

    # ------------------------------------------------------------------
    #  Actualizar firma semántica (aprendizaje de contexto)
    # ------------------------------------------------------------------
    def actualizar_firma(self, firma_nueva: FirmaSemantica, tasa: float = 0.1) -> None:
        """Incorpora gradualmente el contexto observado en la firma de la RSN."""
        self._firma = self._firma.fusionar(firma_nueva, peso=tasa)

    # ------------------------------------------------------------------
    #  Auxiliares privados
    # ------------------------------------------------------------------
    def _calcular_sorpresa(self, entrada: str) -> float:
        """
        Sorpresa como distancia Jaccard entre tokens de predicción y entrada.
        Si la RSN no tiene predicción previa, sorpresa = 1.0.
        """
        if not self._prediccion:
            return 1.0
        tokens_pred  = set(self._prediccion.lower().split())
        tokens_real  = set(entrada.lower().split())
        if not tokens_pred and not tokens_real:
            return 0.0
        union        = tokens_pred | tokens_real
        interseccion = tokens_pred & tokens_real
        return 1.0 - len(interseccion) / len(union)

    def _actualizar_prediccion(self, entrada: str, sorpresa: float) -> None:
        """
        Actualiza la predicción interna mezclando lo observado con lo esperado.
        Alta sorpresa → aprender más del nuevo input.
        """
        if not self._prediccion:
            self._prediccion = entrada
            return
        tasa = RSN_LEARNING_RATE + sorpresa * 0.2
        # mezcla simplificada: si sorpresa alta, adoptar entrada directamente
        if sorpresa > 0.7:
            self._prediccion = entrada
        elif sorpresa > 0.3:
            # mezcla por tokens
            tokens_pred = self._prediccion.split()
            tokens_real = entrada.split()
            n = max(len(tokens_pred), len(tokens_real))
            mezclados   = []
            for i in range(n):
                if i < len(tokens_real):
                    mezclados.append(tokens_real[i])
                elif i < len(tokens_pred):
                    mezclados.append(tokens_pred[i])
            self._prediccion = " ".join(mezclados[:len(tokens_real)])

    @property
    def precision(self) -> float:
        total = self._n_spikes + self._n_correctas
        return self._n_correctas / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (f"RSN(id={self.rsn_id} θ={self._theta:.2f} "
                f"peso={self._peso:.2f} pred='{self._prediccion}')")


# ══════════════════════════════════════════════════════════════════════════════
#  NIVEL 2: MiniColumna — 100 RSNs + inhibición lateral WTA
# ══════════════════════════════════════════════════════════════════════════════

class MiniColumna:
    """
    Agrupa N RSNs especializadas en el mismo concepto semántico.

    Implementa Winner-Take-All (WTA):
      - Todas las RSNs procesan la entrada
      - Solo las MINI_MAX_ACTIVE con mayor sorpresa emiten spike
      - Las perdedoras son inhibidas (θ sube)
      - Las ganadoras son reforzadas (θ baja)

    Esto produce representaciones esparsa automáticamente.
    """

    def __init__(
        self,
        mini_id:   str,
        concepto:  str,
        n_rsn:     int              = 20,
        firma_base: FirmaSemantica | None = None,
    ):
        self.mini_id  = mini_id
        self.concepto = concepto
        firma         = firma_base or FirmaSemantica.vacia()
        self.rsns     = [RSN(i, firma) for i in range(n_rsn)]
        self._historial: list[str] = []   # últimas predicciones correctas

    # ------------------------------------------------------------------
    #  Procesar entrada (feedforward)
    # ------------------------------------------------------------------
    def procesar(
        self,
        entrada:        str,
        firma_contexto: FirmaSemantica,
        feedback_theta: float = 0.0,
    ) -> ActivacionMini:
        """
        Ejecuta ciclo WTA sobre todas las RSNs.
        Devuelve ActivacionMini con los ganadores y la predicción consenso.
        """
        # paso 1: todas las RSNs procesan
        candidatos: list[tuple[SpikeRSN, RSN]] = []
        for rsn in self.rsns:
            spike = rsn.procesar(entrada, firma_contexto, feedback_theta)
            if spike is not None:
                candidatos.append((spike, rsn))

        # paso 2: WTA — ordenar por sorpresa, tomar top MINI_MAX_ACTIVE
        candidatos.sort(key=lambda t: t[0].sorpresa, reverse=True)
        ganadores_raw  = candidatos[:MINI_MAX_ACTIVE]
        perdedores_raw = candidatos[MINI_MAX_ACTIVE:]

        # paso 3: inhibir perdedores, reforzar ganadores
        ganadores_ids = []
        spikes        = []
        for spike, rsn in ganadores_raw:
            rsn.reforzar()
            ganadores_ids.append(rsn.rsn_id)
            spikes.append(spike)
            # actualizar firma de la RSN con el contexto observado
            rsn.actualizar_firma(firma_contexto, tasa=0.05)

        for spike, rsn in perdedores_raw:
            rsn.inhibir()

        # paso 4: predicción consenso = predicción del ganador con mayor peso
        prediccion = ""
        confianza  = 0.0
        if spikes:
            mejor = max(spikes, key=lambda s: s.peso)
            prediccion = mejor.prediccion
            confianza  = mejor.peso
        elif self._historial:
            prediccion = self._historial[-1]
            confianza  = 0.15

        # registrar si predijo bien
        if prediccion == entrada:
            self._historial.append(entrada)
            if len(self._historial) > 100:
                self._historial.pop(0)

        sorpresa_max = max((s.sorpresa for s in spikes), default=0.0)

        # firma de salida: fusión de firmas de los ganadores
        firma_sal = FirmaSemantica.vacia()
        for spike in spikes:
            firma_sal = firma_sal.fusionar(spike.firma, peso=spike.peso)

        return ActivacionMini(
            mini_id      = self.mini_id,
            concepto     = self.concepto,
            spikes       = spikes,
            ganadores    = ganadores_ids,
            sorpresa_max = sorpresa_max,
            prediccion   = prediccion,
            confianza    = confianza,
            firma        = firma_sal,
        )

    # ------------------------------------------------------------------
    #  Recibir feedback top-down
    # ------------------------------------------------------------------
    def aplicar_feedback(
        self,
        prediccion_superior: str,
        firma_superior:      FirmaSemantica,
    ) -> None:
        """
        Ajusta umbrales según la predicción de la capa superior.
        RSNs cuya predicción coincide con la capa superior bajan θ.
        RSNs que contradicen la capa superior suben θ.
        """
        compatibilidad = firma_superior.overlap(
            FirmaSemantica.desde_binario({self.concepto: 1})
        )
        for rsn in self.rsns:
            if rsn._prediccion == prediccion_superior:
                rsn._theta = max(RSN_THETA_MIN, rsn._theta - 0.05 * compatibilidad)
            else:
                rsn._theta = min(RSN_THETA_MAX, rsn._theta + 0.02 * compatibilidad)

    @property
    def firma_agregada(self) -> FirmaSemantica:
        """Firma máxima de todas las RSNs — representa el dominio de esta minicolumna."""
        agg: dict[str, float] = {}
        for rsn in self.rsns:
            for k, v in rsn._firma.bits.items():
                agg[k] = max(agg.get(k, 0.0), v)
        return FirmaSemantica(bits=agg)

    def __repr__(self) -> str:
        activas = sum(1 for r in self.rsns if r._theta < RSN_THETA_BASE)
        return f"MiniCol({self.mini_id} '{self.concepto}' rsns={len(self.rsns)} activas~{activas})"


# ══════════════════════════════════════════════════════════════════════════════
#  NIVEL 3: ColumnaSemantica — N MiniColumnas, 1 dominio
# ══════════════════════════════════════════════════════════════════════════════

class ColumnaSemantica:
    """
    Agrupa MiniColumnas de un mismo dominio semántico.

    Responsabilidades:
      - Enrutar la entrada solo a minicolumnas relevantes por firma
      - Agregar predicciones por votación ponderada por confianza
      - Aplicar segundo nivel de inhibición (solo COL_MAX_MINI_ACTIVE activas)
      - Propagar feedback hacia abajo
    """

    def __init__(self, col_id: str, dominio: str):
        self.col_id    = col_id
        self.dominio   = dominio
        self.minis:  dict[str, MiniColumna] = {}

    # ------------------------------------------------------------------
    #  Gestión de minicolumnas
    # ------------------------------------------------------------------
    def agregar_mini(self, mini: MiniColumna) -> None:
        self.minis[mini.mini_id] = mini

    def crear_mini(
        self,
        concepto:   str,
        n_rsn:      int = 20,
        firma_base: FirmaSemantica | None = None,
    ) -> MiniColumna:
        mini_id = f"{self.col_id}:{concepto}"
        mini    = MiniColumna(mini_id, concepto, n_rsn, firma_base)
        self.minis[mini_id] = mini
        return mini

    # ------------------------------------------------------------------
    #  Procesar entrada (feedforward)
    # ------------------------------------------------------------------
    def procesar(
        self,
        entrada:        str,
        firma_contexto: FirmaSemantica,
        feedback_theta: float = 0.0,
    ) -> ActivacionColumna:
        """
        Procesa la entrada en todas las minicolumnas relevantes.
        Aplica inhibición lateral entre minicolumnas (COL_MAX_MINI_ACTIVE).
        """
        activaciones_raw: list[ActivacionMini] = []

        for mini in self.minis.values():
            relevancia = firma_contexto.overlap(mini.firma_agregada)
            if relevancia < ROUTING_THRESH and firma_contexto:
                continue
            act = mini.procesar(entrada, firma_contexto, feedback_theta)
            activaciones_raw.append(act)

        # inhibición lateral entre minicolumnas: solo top-K por sorpresa
        activaciones_raw.sort(key=lambda a: a.sorpresa_max, reverse=True)
        activas   = activaciones_raw[:COL_MAX_MINI_ACTIVE]

        # predicción por votación ponderada
        votos: dict[str, float] = {}
        for act in activas:
            if act.prediccion:
                votos[act.prediccion] = votos.get(act.prediccion, 0.0) + act.confianza

        prediccion = max(votos, key=votos.__getitem__) if votos else ""
        confianza  = (votos[prediccion] / sum(votos.values())) if votos else 0.0

        sorpresa_media = (
            sum(a.sorpresa_max for a in activas) / len(activas)
            if activas else 0.0
        )

        # firma de salida
        firma_sal = FirmaSemantica.vacia()
        for act in activas:
            firma_sal = firma_sal.fusionar(act.firma, peso=act.confianza)

        return ActivacionColumna(
            col_id         = self.col_id,
            activaciones   = activas,
            prediccion     = prediccion,
            confianza      = confianza,
            sorpresa_media = sorpresa_media,
            firma_salida   = firma_sal,
        )

    # ------------------------------------------------------------------
    #  Feedback top-down
    # ------------------------------------------------------------------
    def aplicar_feedback(
        self,
        prediccion_superior: str,
        firma_superior:      FirmaSemantica,
    ) -> None:
        """Propaga feedback a todas las minicolumnas relevantes."""
        for mini in self.minis.values():
            relevancia = firma_superior.overlap(mini.firma_agregada)
            if relevancia >= ROUTING_THRESH:
                mini.aplicar_feedback(prediccion_superior, firma_superior)

    @property
    def firma_agregada(self) -> FirmaSemantica:
        agg: dict[str, float] = {}
        for mini in self.minis.values():
            for k, v in mini.firma_agregada.bits.items():
                agg[k] = max(agg.get(k, 0.0), v)
        return FirmaSemantica(bits=agg)

    def __repr__(self) -> str:
        return f"Columna({self.col_id} '{self.dominio}' minis={len(self.minis)})"


# ══════════════════════════════════════════════════════════════════════════════
#  NIVEL 4: Region — N Columnas, 1 área de conocimiento
# ══════════════════════════════════════════════════════════════════════════════

class Region:
    """
    Agrupa columnas de un área de conocimiento.

    Añade:
      - Memoria episódica de secuencias recientes (contexto de largo plazo)
      - Predicción de orden superior (sobre predicciones de columnas)
      - Ciclos de feedback interno antes de emitir resultado
    """

    def __init__(self, region_id: str, area: str):
        self.region_id = region_id
        self.area      = area
        self.columnas: dict[str, ColumnaSemantica] = {}
        self._memoria_episodica: list[str] = []   # últimas N palabras vistas
        self._max_episodica = 50

    # ------------------------------------------------------------------
    #  Gestión de columnas
    # ------------------------------------------------------------------
    def agregar_columna(self, col: ColumnaSemantica) -> None:
        self.columnas[col.col_id] = col

    def crear_columna(self, dominio: str) -> ColumnaSemantica:
        col_id = f"{self.region_id}:{dominio}"
        col    = ColumnaSemantica(col_id, dominio)
        self.columnas[col_id] = col
        return col

    # ------------------------------------------------------------------
    #  Procesar (feedforward + feedback interno)
    # ------------------------------------------------------------------
    def procesar(
        self,
        entrada:        str,
        firma_contexto: FirmaSemantica,
    ) -> ActivacionRegion:
        """
        Procesa en 2 fases:
          Fase 1 — feedforward: todas las columnas relevantes procesan
          Fase 2 — feedback interno: la predicción ganadora baja a modular θ
        """
        # ── Fase 1: feedforward ──────────────────────────────────────
        activaciones_col: list[ActivacionColumna] = []

        for col in self.columnas.values():
            relevancia = firma_contexto.overlap(col.firma_agregada)
            if relevancia < ROUTING_THRESH and firma_contexto:
                continue
            act = col.procesar(entrada, firma_contexto)
            activaciones_col.append(act)

        # inhibición: solo top REG_MAX_COL_ACTIVE columnas activas
        activaciones_col.sort(key=lambda a: a.sorpresa_media, reverse=True)
        activas = activaciones_col[:REG_MAX_COL_ACTIVE]

        # votación de predicción regional
        votos: dict[str, float] = {}
        for act in activas:
            if act.prediccion:
                votos[act.prediccion] = (
                    votos.get(act.prediccion, 0.0) + act.confianza
                )

        prediccion = max(votos, key=votos.__getitem__) if votos else ""
        confianza  = (votos[prediccion] / sum(votos.values())) if votos else 0.0

        # ── Fase 2: feedback interno ─────────────────────────────────
        if prediccion:
            firma_fb = firma_contexto.fusionar(
                FirmaSemantica.desde_binario({self.area: 1}), peso=0.3
            )
            for col in self.columnas.values():
                col.aplicar_feedback(prediccion, firma_fb)

        # actualizar memoria episódica
        if entrada:
            self._memoria_episodica.append(entrada)
            if len(self._memoria_episodica) > self._max_episodica:
                self._memoria_episodica.pop(0)

        firma_sal = FirmaSemantica.vacia()
        for act in activas:
            firma_sal = firma_sal.fusionar(act.firma_salida, peso=act.confianza)

        return ActivacionRegion(
            region_id    = self.region_id,
            activaciones = activas,
            prediccion   = prediccion,
            confianza    = confianza,
            firma_salida = firma_sal,
        )

    @property
    def firma_agregada(self) -> FirmaSemantica:
        agg: dict[str, float] = {}
        for col in self.columnas.values():
            for k, v in col.firma_agregada.bits.items():
                agg[k] = max(agg.get(k, 0.0), v)
        return FirmaSemantica(bits=agg)

    @property
    def contexto_episodico(self) -> str:
        return " ".join(self._memoria_episodica[-10:])

    def __repr__(self) -> str:
        return f"Region({self.region_id} '{self.area}' cols={len(self.columnas)})"


# ══════════════════════════════════════════════════════════════════════════════
#  NIVEL 5: Sistema RSN — enrutamiento global y generación autoregresiva
# ══════════════════════════════════════════════════════════════════════════════

class SistemaRSN:
    """
    Sistema completo de N regiones especializadas.

    Ciclo de generación:
      1. Enrutamiento: firma_prompt → regiones relevantes (graduado)
      2. Feedforward:  cada región procesa la entrada
      3. Agregación:   votación ponderada entre regiones
      4. Contradicción: penaliza candidatos casi empatados
      5. Confianza:    umbral global antes de emitir palabra
      6. Feedback:     predicción ganadora baja a todas las regiones activas
      7. Firma:        actualización con decay + fusión del ganador
    """

    def __init__(self, sistema_id: str = "RSN-LLM"):
        self.sistema_id = sistema_id
        self.regiones: dict[str, Region] = {}
        self._ciclos_totales   = 0
        self._palabras_emitidas = 0
        self._silencios         = 0

    # ------------------------------------------------------------------
    #  Construcción del sistema
    # ------------------------------------------------------------------
    def agregar_region(self, region: Region) -> None:
        self.regiones[region.region_id] = region

    def crear_region(self, area: str) -> Region:
        region = Region(area, area)
        self.regiones[area] = region
        return region

    # ------------------------------------------------------------------
    #  Enrutamiento graduado
    # ------------------------------------------------------------------
    def _enrutar(
        self,
        firma_prompt: FirmaSemantica,
    ) -> list[tuple[Region, float]]:
        """
        Selecciona regiones por overlap de firma.
        Devuelve lista (Region, relevancia) ordenada por relevancia desc.
        Si la firma está vacía, activa todas las regiones.
        """
        resultado = []
        for region in self.regiones.values():
            if not firma_prompt:
                resultado.append((region, 1.0))
                continue
            rel = firma_prompt.overlap(region.firma_agregada)
            if rel >= ROUTING_THRESH:
                resultado.append((region, rel))

        resultado.sort(key=lambda t: t[1], reverse=True)
        return resultado

    # ------------------------------------------------------------------
    #  Resolución de contradicciones
    # ------------------------------------------------------------------
    def _resolver_contradicciones(
        self,
        votos: dict[str, float],
    ) -> dict[str, float]:
        """
        Penaliza candidatos casi empatados (ratio > 0.85).
        Devuelve votos ajustados.
        """
        if len(votos) < 2:
            return votos

        ordenados = sorted(votos.values(), reverse=True)
        top1, top2 = ordenados[0], ordenados[1]

        if top1 > 0 and (top2 / top1) > 0.85:
            conflictivos = [k for k, v in votos.items() if v >= top2 * 0.85]
            ajustados    = dict(votos)
            for c in conflictivos:
                ajustados[c] *= 0.70
            print(f"  ⚠️  Contradicción: {conflictivos} → penalizados")
            return ajustados

        return votos

    # ------------------------------------------------------------------
    #  Actualización de firma entre ciclos
    # ------------------------------------------------------------------
    def _actualizar_firma(
        self,
        firma_actual:   FirmaSemantica,
        firma_ganadora: FirmaSemantica,
        peso_nueva:     float = 0.30,
    ) -> FirmaSemantica:
        return firma_actual.decaer(FIRMA_DECAY).fusionar(firma_ganadora, peso=peso_nueva)

    # ------------------------------------------------------------------
    #  Ciclo de generación
    # ------------------------------------------------------------------
    def ciclo(
        self,
        entrada:      str,
        firma_prompt: FirmaSemantica,
        verbose:      bool = False,
    ) -> ResultadoSistema:
        """
        Ejecuta un ciclo completo de generación:
          entrada    = última palabra observada (o prompt inicial)
          firma_prompt = contexto semántico acumulado

        Devuelve ResultadoSistema con la siguiente palabra predicha.
        """
        t0 = time.perf_counter()
        self._ciclos_totales += 1

        # ── 1. Enrutamiento ──────────────────────────────────────────
        regiones_activas = self._enrutar(firma_prompt)

        if not regiones_activas:
            self._silencios += 1
            return ResultadoSistema(
                palabra          = None,
                confianza        = 0.0,
                sorpresa_global  = 0.0,
                activaciones     = [],
                firma_nueva      = firma_prompt.decaer(),
                convergio        = False,
                tiempo_ms        = (time.perf_counter() - t0) * 1000,
            )

        # ── 2. Feedforward ───────────────────────────────────────────
        activaciones_reg: list[ActivacionRegion] = []
        rel_por_region:   dict[str, float]       = {}

        for region, relevancia in regiones_activas:
            act = region.procesar(entrada, firma_prompt)
            activaciones_reg.append(act)
            rel_por_region[region.region_id] = relevancia
            if verbose:
                print(f"  [{region.region_id}] rel={relevancia:.2f} "
                      f"pred='{act.prediccion}' conf={act.confianza:.2f}")

        # ── 3. Votación global ───────────────────────────────────────
        votos: dict[str, float] = {}
        for act in activaciones_reg:
            if not act.prediccion:
                continue
            rel = rel_por_region.get(act.region_id, 1.0)
            votos[act.prediccion] = (
                votos.get(act.prediccion, 0.0) + act.confianza * rel
            )

        # ── 4. Resolver contradicciones ──────────────────────────────
        votos = self._resolver_contradicciones(votos)

        # ── 5. Umbral de confianza ───────────────────────────────────
        if not votos or max(votos.values()) < CONFIDENCE_THRESH:
            self._silencios += 1
            return ResultadoSistema(
                palabra          = None,
                confianza        = max(votos.values(), default=0.0),
                sorpresa_global  = self._sorpresa_global(activaciones_reg),
                activaciones     = activaciones_reg,
                firma_nueva      = firma_prompt.decaer(),
                convergio        = False,
                tiempo_ms        = (time.perf_counter() - t0) * 1000,
            )

        # ── 6. Ganador ───────────────────────────────────────────────
        palabra    = max(votos, key=votos.__getitem__)
        confianza  = votos[palabra] / sum(votos.values())

        # firma del ganador (de la región que más lo apoyó)
        firma_gan  = FirmaSemantica.vacia()
        for act in activaciones_reg:
            if act.prediccion == palabra:
                firma_gan = firma_gan.fusionar(act.firma_salida, peso=act.confianza)

        # ── 7. Feedback global ───────────────────────────────────────
        for region, _ in regiones_activas:
            for col in region.columnas.values():
                col.aplicar_feedback(palabra, firma_gan)

        # ── 8. Actualizar firma ──────────────────────────────────────
        firma_nueva = self._actualizar_firma(firma_prompt, firma_gan)

        self._palabras_emitidas += 1

        return ResultadoSistema(
            palabra          = palabra,
            confianza        = confianza,
            sorpresa_global  = self._sorpresa_global(activaciones_reg),
            activaciones     = activaciones_reg,
            firma_nueva      = firma_nueva,
            convergio        = True,
            tiempo_ms        = (time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    #  Generación autoregresiva
    # ------------------------------------------------------------------
    def generar(
        self,
        entrada_inicial: str,
        firma_inicial:   FirmaSemantica,
        max_palabras:    int  = 20,
        verbose:         bool = True,
    ) -> list[str]:
        """
        Genera una secuencia de palabras de forma autoregresiva.

        En cada ciclo t:
          - entrada_t  = palabra generada en t-1
          - firma_t    = firma actualizada con decay + ganador
          - si no converge → detener
        """
        palabras = []
        entrada  = entrada_inicial
        firma_t  = firma_inicial

        for t in range(max_palabras):
            if verbose:
                print(f"\n── t={t} entrada='{entrada}' ──────────────────")
                print(f"   {firma_t}")

            resultado = self.ciclo(entrada, firma_t, verbose=verbose)

            if not resultado.convergio or resultado.palabra is None:
                if verbose:
                    print(f"   → sin predicción confiable (conf={resultado.confianza:.2f})")
                break

            palabras.append(resultado.palabra)
            entrada = resultado.palabra
            firma_t = resultado.firma_nueva

            if verbose:
                print(f"   → '{resultado.palabra}' "
                      f"(conf={resultado.confianza:.2f} "
                      f"sorpresa={resultado.sorpresa_global:.2f} "
                      f"t={resultado.tiempo_ms:.1f}ms)")

        return palabras

    # ------------------------------------------------------------------
    #  Aprendizaje de corpus
    # ------------------------------------------------------------------
    def aprender_parrafo(
        self,
        tokens:  list[str],
        firma:   FirmaSemantica,
        verbose: bool = False,
    ) -> None:
        """
        Alimenta un párrafo token por token a todas las regiones relevantes.
        Las RSNs aprenden las transiciones observadas.

        tokens: lista de palabras del párrafo
        firma:  firma semántica del párrafo completo
        """
        regiones = self._enrutar(firma)
        if not regiones and self.regiones:
            # si no hay match de firma, usar todas las regiones
            regiones = [(r, 1.0) for r in self.regiones.values()]

        for i, token in enumerate(tokens[:-1]):
            siguiente = tokens[i + 1]
            for region, _ in regiones:
                # enseñar: entrada=token actual, las RSNs aprenden a predecir siguiente
                region.procesar(token, firma)
            if verbose and i % 10 == 0:
                print(f"  aprendiendo token {i}/{len(tokens)}: '{token}' → '{siguiente}'")

    # ------------------------------------------------------------------
    #  Estadísticas
    # ------------------------------------------------------------------
    def _sorpresa_global(self, acts: list[ActivacionRegion]) -> float:
        if not acts:
            return 0.0
        vals = [a.confianza for a in acts if a.confianza > 0]
        return 1.0 - (sum(vals) / len(vals)) if vals else 1.0

    def estadisticas(self) -> dict:
        total_rsn = sum(
            len(mini.rsns)
            for r in self.regiones.values()
            for col in r.columnas.values()
            for mini in col.minis.values()
        )
        total_mini = sum(
            len(col.minis)
            for r in self.regiones.values()
            for col in r.columnas.values()
        )
        total_col = sum(len(r.columnas) for r in self.regiones.values())
        return {
            "regiones":          len(self.regiones),
            "columnas":          total_col,
            "minicolumnas":      total_mini,
            "rsns":              total_rsn,
            "ciclos_totales":    self._ciclos_totales,
            "palabras_emitidas": self._palabras_emitidas,
            "silencios":         self._silencios,
            "tasa_emision":      (
                self._palabras_emitidas / self._ciclos_totales
                if self._ciclos_totales > 0 else 0.0
            ),
        }

    def __repr__(self) -> str:
        stats = self.estadisticas()
        return (
            f"SistemaRSN('{self.sistema_id}' "
            f"regiones={stats['regiones']} "
            f"cols={stats['columnas']} "
            f"minis={stats['minicolumnas']} "
            f"rsns={stats['rsns']})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades de impresión
# ══════════════════════════════════════════════════════════════════════════════

def print_resultado(r: ResultadoSistema, titulo: str = "Resultado") -> None:
    estado = "✅" if r.convergio else "❌"
    print(f"\n{estado} {titulo}")
    print(f"  palabra      : {r.palabra!r}")
    print(f"  confianza    : {r.confianza:.4f}")
    print(f"  sorpresa     : {r.sorpresa_global:.4f}")
    print(f"  convergio    : {r.convergio}")
    print(f"  tiempo       : {r.tiempo_ms:.2f}ms")
    print(f"  firma_nueva  : {r.firma_nueva}")
    if r.activaciones:
        print(f"  regiones activas:")
        for act in r.activaciones:
            print(f"    [{act.region_id}] pred='{act.prediccion}' "
                  f"conf={act.confianza:.2f} cols={len(act.activaciones)}")


def print_estadisticas(sistema: SistemaRSN) -> None:
    s = sistema.estadisticas()
    print(f"\n{'─'*50}")
    print(f"  Sistema        : {sistema.sistema_id}")
    print(f"  Regiones       : {s['regiones']}")
    print(f"  Columnas       : {s['columnas']}")
    print(f"  Minicolumnas   : {s['minicolumnas']}")
    print(f"  RSNs           : {s['rsns']}")
    print(f"  Ciclos totales : {s['ciclos_totales']}")
    print(f"  Palabras emit. : {s['palabras_emitidas']}")
    print(f"  Silencios      : {s['silencios']}")
    print(f"  Tasa emisión   : {s['tasa_emision']:.2%}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════════════════════
#  Builder — construir sistema desde configuración
# ══════════════════════════════════════════════════════════════════════════════

class BuilderRSN:
    """
    Construye un SistemaRSN desde una configuración declarativa.

    Ejemplo:
        config = {
            "regiones": {
                "lenguaje": {
                    "dominio_financiero": ["banco", "fondo", "interes"],
                    "dominio_geografico": ["rio",   "orilla", "lago"],
                },
                "conocimiento": {
                    "dominio_temporal":   ["mañana", "hoy", "antes"],
                }
            }
        }
        sistema = BuilderRSN.desde_config(config, n_rsn=10)
    """

    @staticmethod
    def desde_config(
        config:  dict,
        n_rsn:   int = 10,
        verbose: bool = True,
    ) -> SistemaRSN:
        sistema = SistemaRSN()

        for area, columnas_dict in config.get("regiones", {}).items():
            region = sistema.crear_region(area)

            for dominio, conceptos in columnas_dict.items():
                col = region.crear_columna(dominio)

                for concepto in conceptos:
                    firma_base = FirmaSemantica.desde_binario({concepto: 1, dominio: 1})
                    col.crear_mini(concepto, n_rsn=n_rsn, firma_base=firma_base)

        if verbose:
            print(f"✅ Sistema construido: {sistema}")

        return sistema


# ══════════════════════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  RSN v1 — Demo de construcción y ciclo básico")
    print("=" * 60)

    # ── 1. Construir sistema ─────────────────────────────────────────
    config = {
        "regiones": {
            "lenguaje": {
                "financiero": ["banco",  "fondo",   "interes", "deposito"],
                "geografico": ["rio",    "orilla",  "lago",    "corriente"],
                "temporal":   ["mañana", "tarde",   "hoy",     "antes"],
            },
            "conocimiento": {
                "acciones":   ["abre",   "cierra",  "sube",    "baja"],
                "entidades":  ["banco",  "persona", "empresa", "lugar"],
            },
        }
    }

    sistema = BuilderRSN.desde_config(config, n_rsn=10, verbose=True)
    print_estadisticas(sistema)

    # ── 2. Aprender un párrafo ───────────────────────────────────────
    print("\n── Aprendizaje ─────────────────────────────────────")

    corpus_financiero = [
        "el banco abre en la mañana",
        "el banco cierra en la tarde",
        "el fondo deposita interes cada mes",
    ]
    corpus_geografico = [
        "el banco del rio crece en invierno",
        "la orilla del lago es tranquila",
        "la corriente del rio es fuerte",
    ]

    firma_fin = FirmaSemantica.desde_binario({"banco_finance": 1, "interes": 1})
    firma_geo = FirmaSemantica.desde_binario({"banco_river":   1, "agua":    1})

    for texto in corpus_financiero:
        tokens = texto.split()
        sistema.aprender_parrafo(tokens, firma_fin, verbose=False)
        print(f"  ✓ aprendido: '{texto}'")

    for texto in corpus_geografico:
        tokens = texto.split()
        sistema.aprender_parrafo(tokens, firma_geo, verbose=False)
        print(f"  ✓ aprendido: '{texto}'")

    # ── 3. Ciclo de predicción ───────────────────────────────────────
    print("\n── Predicción ──────────────────────────────────────")

    # caso 1: contexto financiero
    firma_prompt = FirmaSemantica.desde_binario({
        "banco_finance": 1, "mañana": 1
    })
    resultado = sistema.ciclo("banco", firma_prompt, verbose=True)
    print_resultado(resultado, "banco [contexto financiero]")

    # caso 2: contexto geográfico
    firma_prompt2 = FirmaSemantica.desde_binario({
        "banco_river": 1, "agua": 1
    })
    resultado2 = sistema.ciclo("banco", firma_prompt2, verbose=True)
    print_resultado(resultado2, "banco [contexto geográfico]")

    # ── 4. Generación autoregresiva ──────────────────────────────────
    print("\n── Generación autoregresiva ────────────────────────")
    secuencia = sistema.generar(
        entrada_inicial = "el",
        firma_inicial   = firma_fin,
        max_palabras    = 6,
        verbose         = True,
    )
    print(f"\n  Secuencia generada: el {' '.join(secuencia)}")

    # ── 5. Estadísticas finales ──────────────────────────────────────
    print_estadisticas(sistema)