# ── espacio_semantico.py ──────────────────────────────────────────

import numpy as np
import pickle
from pathlib import Path

LABEL_DIM = 16 * 22   # 352

class EspacioSemantico:
    """
    Singleton — una sola instancia para todo el sistema.

    Responsabilidades:
      1. Custodiar el diccionario de primitivos
      2. Calcular y cachear firmas de conceptos
      3. Mantener la matriz M (C × N)
      4. Proveer coordenadas a cualquier BAN
      5. Persistir y restaurar el estado completo
    """

    _instancia: "EspacioSemantico | None" = None
    _ruta_default = Path("models/espacio_semantico.pkl")

    # ── singleton ────────────────────────────────────────────────
    @classmethod
    def instancia(cls) -> "EspacioSemantico":
        """
        Retorna la única instancia del espacio semántico.
        La crea si no existe. La carga de disco si hay archivo.
        """
        if cls._instancia is None:
            if cls._ruta_default.exists():
                cls._instancia = cls.load(cls._ruta_default)
                print(f"  🌐 EspacioSemantico cargado desde disco")
            else:
                cls._instancia = cls()
                print(f"  🌐 EspacioSemantico nuevo creado")
        return cls._instancia

    @classmethod
    def reset(cls):
        """Solo para tests — reinicia el singleton."""
        cls._instancia = None

    # ── constructor ──────────────────────────────────────────────
    def __init__(self):
        # diccionario: índice → [primitivos]
        self.diccionario  : dict[str, list[str]]  = {}

        # primitivos conocidos con índice fijo
        self.primitivos   : list[str]             = []
        self.idx_prim     : dict[str, int]        = {}

        # firmas de primitivos — deterministas por hash
        self._firmas_prim : dict[str, np.ndarray] = {}

        # firmas de conceptos — calculadas y cacheadas
        self._firmas_conc : dict[str, np.ndarray] = {}

        # matriz semántica M (C × N) — se reconstruye al agregar
        self._M           : np.ndarray | None     = None
        self._M_valida    : bool                  = False

    # ── definir concepto ─────────────────────────────────────────
    def definir(self, concepto: str,
                definicion: list[str]) -> "EspacioSemantico":
        """
        concepto   : puede ser compuesto  "banco_financiero"
        definicion : solo primitivos      ["edificio", "dinero", "deposito"]

        Valida que la definición no contenga conceptos compuestos.
        """
        # validar — solo primitivos en la definición
        for rasgo in definicion:
            if rasgo in self.diccionario:
                raise ValueError(
                    f"  ❌ '{rasgo}' es un concepto compuesto.\n"
                    f"     La definición de '{concepto}' solo puede "
                    f"contener primitivos."
                )

        self.diccionario[concepto] = definicion

        # registrar primitivos nuevos
        for p in definicion:
            if p not in self.idx_prim:
                self.primitivos.append(p)
                self.idx_prim[p] = len(self.primitivos) - 1
                self._firmas_prim[p] = self._firma_primitivo(p)

        # invalidar cache
        self._firmas_conc.pop(concepto, None)
        self._M_valida = False

        return self

    # ── firma de primitivo — determinista por hash ───────────────
    def _firma_primitivo(self, primitivo: str) -> np.ndarray:
        """
        Firma bipolar {-1,+1} reproducible desde el nombre.
        Mismo nombre → misma firma siempre.
        """
        semilla = hash(primitivo) % (2**32)
        rng     = np.random.default_rng(seed=semilla)
        return rng.choice([-1.0, 1.0], size=LABEL_DIM).astype(np.float32)

    # ── firma de concepto — suma ponderada de primitivos ─────────
    def firma(self, concepto: str) -> np.ndarray:
        """
        Retorna la firma bipolar del concepto.
        Cacheada — se calcula solo la primera vez.

        w_l = 1/l — primer primitivo pesa más (categoría padre)
        """
        if concepto in self._firmas_conc:
            return self._firmas_conc[concepto]

        if concepto not in self.diccionario:
            raise KeyError(
                f"  ❌ Concepto '{concepto}' no definido.\n"
                f"     Conceptos disponibles: {list(self.diccionario.keys())}"
            )

        definicion    = self.diccionario[concepto]
        acumulada     = np.zeros(LABEL_DIM, dtype=np.float32)

        for l, primitivo in enumerate(definicion, start=1):
            peso       = 1.0 / l
            acumulada += peso * self._firmas_prim[primitivo]

        firma = np.where(acumulada >= 0, 1.0, -1.0).astype(np.float32)
        self._firmas_conc[concepto] = firma
        return firma

    # ── coordenada binaria en espacio de primitivos ──────────────
    def coordenada(self, concepto: str) -> np.ndarray:
        """
        Vector binario {0,1}^N donde N = len(primitivos).
        Fila de la matriz M para este concepto.
        """
        N      = len(self.primitivos)
        coord  = np.zeros(N, dtype=np.float32)
        for p in self.diccionario[concepto]:
            coord[self.idx_prim[p]] = 1.0
        return coord

    # ── matriz M (C × N) ─────────────────────────────────────────
    @property
    def M(self) -> np.ndarray:
        """
        Matriz semántica completa.
        Se reconstruye solo cuando el diccionario cambia.
        """
        if not self._M_valida or self._M is None:
            C          = len(self.diccionario)
            N          = len(self.primitivos)
            self._M    = np.zeros((C, N), dtype=np.float32)
            for i, concepto in enumerate(self.diccionario):
                self._M[i] = self.coordenada(concepto)
            self._M_valida = True
        return self._M

    # ── proyección 3D via SVD ─────────────────────────────────────
    def proyectar_3d(self) -> tuple[np.ndarray, float]:
        """
        Retorna (M_3d, varianza_explicada).
        M_3d shape: (C, 3)
        """
        U, S, _ = np.linalg.svd(self.M, full_matrices=False)
        M_3d    = U[:, :3] * S[:3]
        eta     = float((S[:3]**2).sum() / (S**2).sum())
        return M_3d, eta

    # ── distancia semántica ───────────────────────────────────────
    def distancia(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.coordenada(a) - self.coordenada(b)))

    # ── inferencia por álgebra de conjuntos ──────────────────────
    def seccion(self, primitivo: str) -> list[str]:
        """Todos los conceptos que contienen este primitivo."""
        return [c for c, d in self.diccionario.items()
                if primitivo in d]

    def interseccion(self, primitivos: list[str]) -> list[str]:
        """Conceptos que contienen TODOS los primitivos dados."""
        return [c for c, d in self.diccionario.items()
                if all(p in d for p in primitivos)]

    def excluir(self, primitivos_req: list[str],
                primitivos_excl: list[str]) -> list[str]:
        """
        Conceptos que tienen primitivos_req
        pero NO tienen primitivos_excl.
        """
        candidatos = self.interseccion(primitivos_req)
        return [c for c in candidatos
                if not any(p in self.diccionario[c]
                           for p in primitivos_excl)]

    # ── persistencia ─────────────────────────────────────────────
    def save(self, path: str | Path | None = None) -> "EspacioSemantico":
        path = Path(path or self._ruta_default)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  💾 EspacioSemantico guardado → {path}")
        print(f"     conceptos  : {len(self.diccionario)}")
        print(f"     primitivos : {len(self.primitivos)}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "EspacioSemantico":
        with open(path, "rb") as f:
            instancia = pickle.load(f)
        cls._instancia = instancia
        return instancia

    # ── summary ──────────────────────────────────────────────────
    def summary(self):
        M_3d, eta = self.proyectar_3d()
        conceptos = list(self.diccionario.keys())

        print(f"\n{'═'*60}")
        print(f"  ESPACIO SEMÁNTICO GLOBAL")
        print(f"{'─'*60}")
        print(f"  Conceptos  : {len(self.diccionario)}")
        print(f"  Primitivos : {len(self.primitivos)}")
        print(f"  Matriz M   : {self.M.shape}")
        print(f"  Varianza 3D: {eta:.1%}")
        print(f"{'─'*60}")
        print(f"  {'CONCEPTO':<25} {'DEFINICIÓN'}")
        for c, d in self.diccionario.items():
            print(f"  {c:<25} {d}")
        print(f"{'─'*60}")
        print(f"  COORDENADAS 3D:")
        for i, c in enumerate(conceptos):
            x, y, z = M_3d[i]
            print(f"  {c:<25} ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        print(f"{'═'*60}\n")
        
        
        
        
        
class BAN:
    def __init__(self):
        # ── existentes ───────────────────────────────────────────
        self.labels     : list[str]             = []
        self.label_vecs : dict[str, np.ndarray] = {}
        self._A_rows    : list                  = []
        self._B_rows    : list                  = []
        self.W_fwd      : np.ndarray | None     = None
        self.W_back     : np.ndarray | None     = None
        self._fitted    : bool                  = False

        # ── referencia al espacio semántico global ───────────────
        # NO es una copia — es una referencia al singleton
        self._espacio   : EspacioSemantico = EspacioSemantico.instancia()

    def train_from_(self, filename: str, label: str, ...):
        """
        Usa el espacio semántico para obtener el vector del label.
        Si el label está en el diccionario usa su firma semántica.
        Si no, usa codificación posicional clásica.
        """
        if label in self._espacio.diccionario:
            # firma desde el espacio semántico global ✅
            vec_B = self._espacio.firma(label)
        else:
            # fallback — codificación posicional
            vec_B = _encode_label(len(self.labels))

        if label not in self.label_vecs:
            self.labels.append(label)
            self.label_vecs[label] = vec_B

        # resto igual que antes...
        vec_A = _preprocess(filename)
        self._A_rows.append(sp.csr_matrix(vec_A.reshape(1, -1)))
        self._B_rows.append(vec_B)
        self._fit()        
        
        
        
        
        
        
        