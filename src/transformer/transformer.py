

"""
╔══════════════════════════════════════════════════════════════════════╗
║        TRANSFORMER  —  PREDICTOR DE SIGUIENTE PALABRA               ║
║                                                                      ║
║  Arquitectura Decoder-Only (estilo GPT):                             ║
║                                                                      ║
║    Token IDs                                                         ║
║       ↓                                                              ║
║    [Embedding]  +  [Positional Encoding]                            ║
║       ↓                                                              ║
║    ┌─ TransformerBlock ─────────────────────────────┐               ║
║    │  MultiHead Self-Attention (máscara causal)     │               ║
║    │  → residual + LayerNorm                        │               ║
║    │  FeedForward  ReLU/GELU                        │               ║
║    │  → residual + LayerNorm                        │               ║
║    └────────────────────────────────────────────────┘               ║
║       ↓                                                              ║
║    [Linear head]  →  Softmax  →  P(siguiente palabra)              ║
║                                                                      ║
║  Clave: Atención CAUSAL  → cada token atiende a TODA la            ║
║         secuencia anterior simultáneamente, sin ventana fija.       ║
║         "banco" ≠ "banco del río" gracias al contexto completo.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, re, math, random, pickle
from collections import Counter
import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  DETECCIÓN DE BACKEND
# ──────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    BACKEND = "pytorch"
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅  Backend: PyTorch  |  Dispositivo: {DEVICE}")
except ImportError:
    BACKEND = "numpy"
    print("⚠️  PyTorch no encontrado")
    print("    → Usando Transformer NumPy puro con backprop analítico completo")


# ══════════════════════════════════════════════════════════════════════
#  TOKENIZADOR
# ══════════════════════════════════════════════════════════════════════
class WordTokenizer:
    SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self.word2idx: dict = {}
        self.idx2word: dict = {}

    @staticmethod
    def tokenize(text: str) -> list:
        import unicodedata
        # ── 1. Dividir CamelCase ANTES de lowercase ──────────────
        #    "otherYou"  →  "other You"
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

        # ── 2. Normalizar unicode ─────────────────────────────────
        text = unicodedata.normalize("NFKD", text)
        # Apóstrofes curvos → recto ASCII
        text = re.sub(r"[\u2018\u2019\u201a\u201b\u02bc\u02b9]", "'", text)
        # Comillas tipográficas → rectas
        text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
        # Guiones especiales (em-dash, en-dash) → guión normal
        text = re.sub(r"[\u2013\u2014\u2012\u2010]", "-", text)

        text = text.lower()

        # ── 3. Separar puntuación y guiones ──────────────────────
        text = re.sub(r"([.,!?;:\"'¿¡()\[\]{}/\\])", r" \1 ", text)
        text = re.sub(r"-", " - ", text)

        # ── 4. Separar dígitos pegados a letras ──────────────────
        #    "banco2024" → "banco 2024"
        text = re.sub(r"([a-záéíóúüñ])([0-9])", r"\1 \2", text)
        text = re.sub(r"([0-9])([a-záéíóúüñ])", r"\1 \2", text)

        # ── 5. Tokenizar y filtrar ────────────────────────────────
        tokens = [t.strip() for t in text.split() if t.strip()]
        # Descartar tokens de control o vacíos; limitar longitud máxima
        tokens = [t for t in tokens if 1 <= len(t) <= 30]
        return tokens

    @staticmethod
    def _is_suspicious(token: str) -> bool:
        """Detecta tokens que parecen ser palabras pegadas sin espacio."""
        # Umbral: más de 20 caracteres alfabéticos seguidos sin ser una URL
        if len(token) > 20 and re.match(r"^[a-záéíóúüñ]+$", token):
            return True
        return False

    def build(self, corpus: list) -> None:
        counter: Counter = Counter()
        for s in corpus:
            counter.update(self.tokenize(s))
        self.word2idx, self.idx2word = {}, {}
        for i, sp in enumerate(self.SPECIALS):
            self.word2idx[sp] = i
            self.idx2word[i]  = sp
        idx = len(self.SPECIALS)
        suspicious = []
        for word, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if self._is_suspicious(word):
                suspicious.append(word)
                continue          # no añadir tokens sospechosos al vocab
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
            idx += 1
        print(f"📚  Vocabulario: {len(self.word2idx)} tokens únicos")
        if suspicious:
            print(f"⚠️   Tokens ignorados por ser posibles palabras pegadas:")
            for s in suspicious:
                print(f"      '{s}'  →  ¿quisiste decir: '{' '.join(s)}' ?")
            print("    Tip: asegúrate de separar las palabras con espacios.")

    def encode(self, text: str) -> list:
        unk = self.word2idx["<UNK>"]
        return [self.word2idx.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids: list) -> list:
        return [self.idx2word.get(i, "<UNK>") for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx["<PAD>"]


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER NumPy — backprop analítico completo, optimizador Adam
# ══════════════════════════════════════════════════════════════════════
class NumpyTransformer:
    """
    Transformer decoder-only de 1 capa implementado completamente
    en NumPy puro con backpropagation analítica por todas las capas.

    Parámetros entrenables:
        E               Embedding matricial               (V × D)
        Wq, Wk, Wv, Wo  Proyecciones de atención          (D × D)
        g1, b1n         LayerNorm 1 (escala, bias)        (D,)
        W1, b1          FFN capa entrada                  (D × Dff)
        W2, b2          FFN capa salida                   (Dff × D)
        g2, b2n         LayerNorm 2                       (D,)
        Wh              Cabeza de lenguaje (LM head)      (D × V)
    """

    def __init__(self, vocab_size: int, d_model: int = 64,
                 d_ff: int = 128, ctx: int = 24):
        self.V, self.D, self.Dff, self.ctx = vocab_size, d_model, d_ff, ctx
        r = 0.05

        # Embeddings + proyecciones atención
        self.E   = np.random.randn(vocab_size, d_model) * r
        self.Wq  = np.eye(d_model) + np.random.randn(d_model, d_model) * r * 0.1
        self.Wk  = np.eye(d_model) + np.random.randn(d_model, d_model) * r * 0.1
        self.Wv  = np.eye(d_model) + np.random.randn(d_model, d_model) * r * 0.1
        self.Wo  = np.eye(d_model) + np.random.randn(d_model, d_model) * r * 0.1
        # LayerNorm 1
        self.g1  = np.ones(d_model)
        self.b1n = np.zeros(d_model)
        # FFN
        self.W1  = np.random.randn(d_model, d_ff) * r
        self.b1  = np.zeros(d_ff)
        self.W2  = np.random.randn(d_ff, d_model) * r
        self.b2  = np.zeros(d_model)
        # LayerNorm 2
        self.g2  = np.ones(d_model)
        self.b2n = np.zeros(d_model)
        # LM Head (peso compartido con embedding = weight tying)
        self.Wh  = self.E.T.copy()   # (D × V)

        # Adam
        self._step = 0
        self._m    = {p: np.zeros_like(getattr(self, p)) for p in self._params()}
        self._v    = {p: np.zeros_like(getattr(self, p)) for p in self._params()}

    def _params(self):
        return ["E","Wq","Wk","Wv","Wo","g1","b1n","W1","b1","W2","b2","g2","b2n","Wh"]

    # ── Funciones elementales ─────────────────────────────
    @staticmethod
    def softmax(x, axis=-1):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

    @staticmethod
    def layer_norm(x, g, b, eps=1e-6):
        mu  = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + eps)
        return g * xhat + b, xhat, np.sqrt(var + eps)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # ── Positional Encoding sinusoidal ───────────────────
    def pos_enc(self, T):
        pe  = np.zeros((T, self.D))
        pos = np.arange(T)[:, None]
        div = np.exp(np.arange(0, self.D, 2) * (-math.log(10000) / self.D))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:self.D//2])
        return pe

    # ── Forward ──────────────────────────────────────────
    def forward(self, ids: list) -> tuple:
        ids = list(ids[-self.ctx:])
        T   = len(ids)
        D, Dff = self.D, self.Dff

        # 1. Embedding + pos encoding
        X  = self.E[ids] + self.pos_enc(T)             # (T, D)

        # 2. Proyecciones Q K V
        Q  = X @ self.Wq                               # (T, D)
        K  = X @ self.Wk
        V_ = X @ self.Wv

        # 3. Scaled dot-product attention (causal)
        scale  = math.sqrt(D)
        scores = (Q @ K.T) / scale                     # (T, T)
        mask   = np.triu(np.full((T, T), -1e9), 1)
        scores = scores + mask
        A      = self.softmax(scores, axis=-1)         # (T, T)
        Cv     = A @ V_                                # (T, D)
        attn_out = Cv @ self.Wo                        # (T, D)

        # 4. Residual + LayerNorm 1
        H1_pre = X + attn_out
        H1, xhat1, std1 = self.layer_norm(H1_pre, self.g1, self.b1n)

        # 5. Feed-Forward: ReLU
        ff_in  = H1 @ self.W1 + self.b1               # (T, Dff)
        ff_act = self.relu(ff_in)                      # (T, Dff)
        ff_out = ff_act @ self.W2 + self.b2            # (T, D)

        # 6. Residual + LayerNorm 2
        H2_pre = H1 + ff_out
        H2, xhat2, std2 = self.layer_norm(H2_pre, self.g2, self.b2n)

        # 7. LM head: solo el último token real
        last   = H2[-1]                                # (D,)
        logits = last @ self.Wh                        # (V,)
        probs  = self.softmax(logits)

        cache = dict(
            ids=ids, X=X, Q=Q, K=K, V_=V_, A=A, Cv=Cv,
            attn_out=attn_out, H1_pre=H1_pre,
            H1=H1, xhat1=xhat1, std1=std1,
            ff_in=ff_in, ff_act=ff_act, ff_out=ff_out,
            H2_pre=H2_pre, H2=H2, xhat2=xhat2, std2=std2,
            last=last, logits=logits, probs=probs, T=T,
        )
        return probs, cache

    # ── Backward ─────────────────────────────────────────
    def backward(self, cache: dict, target: int) -> dict:
        T, D, Dff = cache["T"], self.D, self.Dff
        ids = cache["ids"]
        g   = {p: np.zeros_like(getattr(self, p)) for p in self._params()}

        # dLoss/dlogits  (softmax + cross-entropy)
        dl = cache["probs"].copy()
        dl[target] -= 1.0                              # (V,)

        # Head: Wh (D × V), last (D,)
        g["Wh"] += np.outer(cache["last"], dl)
        dlast    = self.Wh @ dl                        # (D,)

        # --- LayerNorm backward helper ---
        def ln_back(dout, xhat, std, g_param):
            N    = xhat.shape[-1]
            dg   = (dout * xhat).sum(0)
            db   = dout.sum(0)
            dxhat = dout * g_param
            dx   = (dxhat - dxhat.mean(-1, keepdims=True)
                    - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
            return dx, dg, db

        # Propagamos dlast a H2
        dH2       = np.zeros((T, D))
        dH2[-1]   = dlast

        # LayerNorm 2 backward
        dH2_pre, dg2, db2n = ln_back(dH2, cache["xhat2"], cache["std2"], self.g2)
        g["g2"] += dg2; g["b2n"] += db2n

        # Residual 2
        dH1_r2   = dH2_pre.copy()
        dff_out  = dH2_pre.copy()

        # FFN backward
        g["b2"] += dff_out.sum(0)
        g["W2"] += cache["ff_act"].T @ dff_out
        dff_act  = dff_out @ self.W2.T
        dff_in   = dff_act * (cache["ff_in"] > 0)     # ReLU backward
        g["b1"] += dff_in.sum(0)
        g["W1"] += cache["H1"].T @ dff_in
        dH1_ff   = dff_in @ self.W1.T

        dH1      = dH1_r2 + dH1_ff

        # LayerNorm 1 backward
        dH1_pre, dg1, db1n = ln_back(dH1, cache["xhat1"], cache["std1"], self.g1)
        g["g1"] += dg1; g["b1n"] += db1n

        # Residual 1
        dX_r1     = dH1_pre.copy()
        dattn_out = dH1_pre.copy()

        # Atención backward
        g["Wo"] += cache["Cv"].T @ dattn_out
        dCv      = dattn_out @ self.Wo.T
        dV_      = cache["A"].T @ dCv
        dA       = dCv @ cache["V_"].T

        # Softmax backward (batched causal)
        A     = cache["A"]
        dsco  = A * (dA - (dA * A).sum(-1, keepdims=True))
        dsco /= math.sqrt(D)
        dsco[np.triu(np.ones((T,T), dtype=bool), 1)] = 0.0   # causal

        dQ = dsco  @ cache["K"]
        dK = dsco.T @ cache["Q"]

        g["Wq"] += cache["X"].T @ dQ
        g["Wk"] += cache["X"].T @ dK
        g["Wv"] += cache["X"].T @ dV_

        dX_attn  = dQ @ self.Wq.T + dK @ self.Wk.T + dV_ @ self.Wv.T
        dX       = dX_r1 + dX_attn                    # (T, D)

        # Embedding backward
        for t, idx in enumerate(ids):
            g["E"][idx] += dX[t]

        return g

    # ── Optimizador Adam ──────────────────────────────────
    def adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._step += 1
        t = self._step
        for name, grad in grads.items():
            m = b1*self._m[name] + (1-b1)*grad
            v = b2*self._v[name] + (1-b2)*grad**2
            self._m[name] = m
            self._v[name] = v
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            setattr(self, name, getattr(self, name) - lr * mh / (np.sqrt(vh) + eps))

    # ── Predicción ────────────────────────────────────────
    def predict_probs(self, ids, temperature=1.0):
        probs, _ = self.forward(ids)
        if temperature != 1.0:
            log_p = np.log(probs + 1e-9) / temperature
            e     = np.exp(log_p - log_p.max())
            probs = e / e.sum()
        return probs

    def attention_map(self, ids):
        """Pesos de atención del último token hacia cada posición anterior."""
        _, cache = self.forward(ids)
        return cache["A"][-1]   # (T,)

    def param_count(self):
        return sum(getattr(self, p).size for p in self._params())


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER PyTorch
# ══════════════════════════════════════════════════════════════════════
if BACKEND == "pytorch":

    class PositionalEncoding(nn.Module):
        def __init__(self, d, max_len=512, dropout=0.1):
            super().__init__()
            self.drop = nn.Dropout(dropout)
            pe  = torch.zeros(max_len, d)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000)/d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return self.drop(x + self.pe[:, :x.size(1)])

    class Block(nn.Module):
        def __init__(self, d, heads, dff, drop):
            super().__init__()
            self.attn  = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.ff    = nn.Sequential(nn.Linear(d, dff), nn.GELU(),
                                       nn.Dropout(drop), nn.Linear(dff, d))
            self.n1    = nn.LayerNorm(d)
            self.n2    = nn.LayerNorm(d)
            self.drop  = nn.Dropout(drop)
        def forward(self, x, causal, pad_mask):
            a, _ = self.attn(x, x, x, attn_mask=causal,
                             key_padding_mask=pad_mask, need_weights=False)
            x = self.n1(x + self.drop(a))
            x = self.n2(x + self.drop(self.ff(x)))
            return x

    class TransformerLM(nn.Module):
        def __init__(self, V, d=128, heads=4, layers=3,
                     dff=256, max_len=128, drop=0.1, pad=0):
            super().__init__()
            self.d = d; self.pad = pad
            self.embed  = nn.Embedding(V, d, padding_idx=pad)
            self.pos    = PositionalEncoding(d, max_len, drop)
            self.blocks = nn.ModuleList([Block(d, heads, dff, drop) for _ in range(layers)])
            self.norm   = nn.LayerNorm(d)
            self.head   = nn.Linear(d, V, bias=False)
            self.head.weight = self.embed.weight
            for p in self.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
        def _cmask(self, T, dev):
            return torch.triu(torch.ones(T, T, device=dev), 1).bool()
        def forward(self, x):
            B, T  = x.shape
            pad_m = (x == self.pad)
            cm    = self._cmask(T, x.device)
            h = self.pos(self.embed(x) * math.sqrt(self.d))
            for b in self.blocks:
                h = b(h, cm, pad_m)
            return self.head(self.norm(h))
        def n_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
class TransformerEngine:

    DEFAULT_CORPUS = [
        # --- 1. BANK: ÁMBITO FINANCIERO (Institución, Dinero, Crédito) ---
        "The bank was closed because the financial officer was late",
        "The central bank raised interest rates today",
        "The bank loaned money to the entrepreneur yesterday",
        "The bank manager arrived late to the meeting",
        "The financier reviewed all the bank documents",
        "The bank approved the credit for the company",
        "The financier arrived late to the bank due to traffic",
        "The bank closed at five o'clock sharp",
        "He deposited his paycheck at the local bank branch",
        "The investment bank managed the portfolio for the client",
        "Every commercial bank must follow federal regulations",
        "She opened a savings account at the city bank",
        "The bank teller counted the cash before the bank closed",
        "A mobile bank app allows users to transfer money easily",
        "The bank vault was locked with a digital security code",
        "The international bank provides loans to developing nations",
        "The bank statement showed a high balance this month",
        "The world bank released a report on global poverty",
        "I need to go to the bank to talk to a bank officer",
        "The bank robbery was prevented by the bank security guard",
        "A blood bank is not the same as a commercial bank",
        "The bank charges a fee for every international wire",
        "Many people lost trust in the bank after the bank crash",
        "The bank headquarters is located in the bank district",
        "He works as a bank analyst for a large merchant bank",
        "The bank credit card has a very high interest rate",
        "The bank mortgage allowed them to buy a new house",
        "The food bank distributes meals near the local bank",
        "The bank solvent status was confirmed by the bank auditor",
        "The bank reserves are held at the federal reserve bank",

        # --- 2. BANK: GEOGRAFÍA Y NATURALEZA (Río, Orilla, Pendiente) ---
        "The river bank has a lot of green vegetation",
        "I sat on the park bank to read my book",
        "The river and the bank are part of the natural landscape",
        "A bank of fish swam along the quiet river",
        "The river water was cold near the muddy bank",
        "The river rose and flooded the bank after the storm",
        "There is a grassy bank where we can have our picnic",
        "Trees were planted along the bank to prevent soil erosion",
        "The steep bank made it difficult to reach the water",
        "We sat on the sandy bank and watched the sunset",
        "The north bank of the river is popular for fishing",
        "A small wooden boat was tied to the river bank",
        "The river bank was covered in colorful spring flowers",
        "The current was strong near the rocky river bank",
        "He stood on the bank and jumped into the river",
        "The canal bank is a great place for a morning run",
        "The steep bank protected the house from the rising river",
        "Frogs were jumping from the bank into the river water",
        "The river bank erosion is a serious environmental problem",
        "They built a stone wall along the river bank",
        "The muddy bank of the river bank was full of tracks",
        "A steep bank of earth rose behind the small cottage",

        # --- 3. BANK: CONJUNTO O COLECCIÓN (Datos, Nubes, Máquinas) ---
        "The data bank contains thousands of medical records",
        "A thick bank of clouds moved in from the west",
        "The blood bank is calling for more donors this month",
        "The control room has a massive bank of monitors",
        "The server room houses a bank of high-speed computers",
        "The memory bank was corrupted after the power surge",
        "A bank of switches controls the lights in the building",
        "The test questions were selected from a large question bank",
        "A dense bank of fog rolled in from the harbor",
        "The seed bank preserves thousands of plant species",
        "A bank of batteries provides power to the facility",
        "The organ bank coordinates transplants across the country",
        "A bank of elevators leads to the top floor of the bank",
        "The sound bank includes hundreds of digital effects",
        "A dark bank of smoke rose from the industrial bank",
        "The project uses a bank of sensors to monitor the river",
        "A bank of telephones was installed in the bank lobby",
        "The power plant has a bank of giant transformers",

        # --- 4. BANK: ACCIÓN / VERBO (Inclinar, Confiar, Apilar) ---
        "You can bank on his honesty for this project",
        "The aircraft began to bank steeply to the left",
        "The pilot had to bank the plane to avoid the storm",
        "He learned how to bank the billiard ball into the pocket",
        "She decided to bank the fire to keep the room warm",
        "The cyclist had to bank hard to stay on the track",
        "The driver had to bank the car around the sharp curve",
        "The eagle started to bank as it searched for prey",
        "You should not bank on the weather being good tomorrow",
        "The bobsled began to bank high on the icy wall",
        "The jet will bank right before it reaches the bank",
        "He decided to bank the extra coals for the morning",
        "You can bank on the fact that the bank will be open",
        "The drone began to bank to capture the river bank",

        # --- 5. FRASES MIXTAS (Conexiones Semánticas Cruzadas) ---
        "The bank manager sat on the river bank during lunch",
        "The pilot saw a bank of clouds over the river bank",
        "The data bank was funded by the central bank",
        "The bank officer deposited the data in the data bank",
        "The financier stood on the river bank to think about the bank",
        "A bank of monitors showed the bank security footage",
        "The bank loan helped the farmer protect the river bank",
        "The bank teller saw the cloud bank from the bank window",
        "The bank vault is as solid as a stone river bank",
        "You can bank on the bank manager to approve the bank loan",
        "The bank of fish was visible from the steep river bank",
        "The blood bank is located right next to the city bank",
        "The bank district is famous for its large bank of offices",
        "The bank manager had to bank the plane during his flight lesson",
        "The river bank was as busy as the local bank branch",
        "A bank of fog made the bank building look ghostly"
    ]

    def __init__(self, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, max_len=24, dropout=0.1):
        self.tokenizer = WordTokenizer()
        self.corpus    = list(self.DEFAULT_CORPUS)
        self.model     = None
        self._hp = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        d_ff=d_ff, max_len=max_len, dropout=dropout)

    def build(self):
        self.tokenizer.build(self.corpus)
        V = self.tokenizer.vocab_size
        if BACKEND == "pytorch":
            self.model = TransformerLM(
                V=V, d=self._hp["d_model"], heads=self._hp["n_heads"],
                layers=self._hp["n_layers"], dff=self._hp["d_ff"],
                max_len=self._hp["max_len"], drop=self._hp["dropout"],
                pad=self.tokenizer.pad_idx
            ).to(DEVICE)
            print(f"🧠  TransformerLM (PyTorch): {self.model.n_params():,} parámetros")
        else:
            self.model = NumpyTransformer(V, d_model=self._hp["d_model"],
                                          d_ff=self._hp["d_ff"],
                                          ctx=self._hp["max_len"])
            print(f"🧠  NumpyTransformer (backprop completo): {self.model.param_count():,} parámetros")
            print(f"    Capas: Embed → PosEnc → MHA(causal) → FFN(ReLU) → LayerNorm × 2 → LM-Head")
            print(f"    Optimizador: Adam  |  Gradientes: analíticos por todas las capas")

    def _make_sequences(self):
        pairs  = []
        pad    = self.tokenizer.pad_idx
        maxlen = self._hp["max_len"]
        for s in self.corpus:
            ids = self.tokenizer.encode(s)
            if len(ids) < 2: continue
            for t in range(1, len(ids)):
                ctx    = ids[:t]
                target = ids[t]
                ctx = (ctx[-maxlen:] if len(ctx) >= maxlen
                       else [pad]*(maxlen-len(ctx)) + ctx)
                pairs.append((ctx, target))
        return pairs

    def train(self, epochs=100, lr=5e-3, batch_size=32):
        if self.model is None:
            self.build()
        pairs = self._make_sequences()
        print(f"\n🏋️  {epochs} épocas  |  {len(pairs)} pares  |  lr={lr}")
        print("─"*65)
        if BACKEND == "pytorch":
            self._train_pytorch(pairs, epochs, lr, batch_size)
        else:
            self._train_numpy(pairs, epochs, lr)

    def _train_pytorch(self, pairs, epochs, lr, batch_size):
        opt   = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs*(len(pairs)//batch_size+1))
        crit  = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_idx)
        self.model.train()
        for epoch in range(1, epochs+1):
            random.shuffle(pairs)
            total, nb = 0.0, 0
            for i in range(0, len(pairs), batch_size):
                b   = pairs[i:i+batch_size]
                ctx = torch.tensor([p[0] for p in b], dtype=torch.long, device=DEVICE)
                tgt = torch.tensor([p[1] for p in b], dtype=torch.long, device=DEVICE)
                opt.zero_grad()
                loss = crit(self.model(ctx)[:,-1,:], tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step()
                total += loss.item(); nb += 1
            avg = total/max(nb,1)
            if epoch % 10 == 0 or epoch in (1, epochs):
                bar = "█"*max(0, int(25*(1-avg/6))) + "░"*max(0, 25-int(25*(1-avg/6)))
                print(f"  Época {epoch:>4}/{epochs}  │  Loss: {avg:.4f}"
                      f"  │  Perp: {math.exp(min(avg,20)):6.2f}  │  {bar}")
        self.model.eval()
        print("─"*65); print("✅  Entrenamiento completado\n")

    def _train_numpy(self, pairs, epochs, lr):
        for epoch in range(1, epochs+1):
            random.shuffle(pairs)
            total = 0.0
            for ctx, target in pairs:
                probs, cache = self.model.forward(ctx)
                loss   = -math.log(float(probs[target]) + 1e-9)
                grads  = self.model.backward(cache, target)
                self.model.adam_step(grads, lr)
                total += loss
            avg = total / len(pairs)
            if epoch % 10 == 0 or epoch in (1, epochs):
                bar = "█"*max(0, int(25*(1-avg/6))) + "░"*max(0, 25-int(25*(1-avg/6)))
                print(f"  Época {epoch:>4}/{epochs}  │  Loss: {avg:.4f}"
                      f"  │  Perp: {math.exp(min(avg,20)):6.2f}  │  {bar}")
        print("─"*65); print("✅  Entrenamiento completado\n")

    def predict(self, prompt, top_k=8, temperature=1.0):
        if self.model is None:
            raise RuntimeError("Primero entrena con: train")
        ids = self.tokenizer.encode(prompt)
        if not ids: raise ValueError("Prompt vacío.")
        maxlen = self._hp["max_len"]
        ctx = ids[-maxlen:]
        ctx = [self.tokenizer.pad_idx]*(maxlen-len(ctx)) + ctx
        specials = {self.tokenizer.word2idx[s] for s in WordTokenizer.SPECIALS}

        if BACKEND == "pytorch":
            self.model.eval()
            with torch.no_grad():
                t = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                logit = self.model(t)[0,-1,:] / temperature
                probs = F.softmax(logit, dim=-1).cpu().numpy()
        else:
            probs = self.model.predict_probs(ctx, temperature)

        return sorted(
            [(self.tokenizer.idx2word[i], float(probs[i]))
             for i in range(len(probs)) if i not in specials],
            key=lambda x: -x[1]
        )[:top_k]

    def show_attention(self, prompt):
        ids    = self.tokenizer.encode(prompt)
        tokens = self.tokenizer.decode(ids)
        if not ids: return
        maxlen = self._hp["max_len"]
        ctx    = ids[-maxlen:]
        pad_n  = maxlen - len(ctx)
        ctx_p  = [self.tokenizer.pad_idx]*pad_n + ctx
        real   = tokens[-len(ctx):]

        if BACKEND == "pytorch":
            print("  (Mapa de atención detallado disponible en la versión PyTorch)")
            return

        a = self.model.attention_map(ctx_p)[-len(real):]  # solo tokens reales
        print(f"\n  🔍  Atención del último token «{real[-1]}» sobre el contexto:\n")
        print(f"  {'Token':<22} {'Peso':>8}   {'Barra visual':30}")
        print(f"  {'─'*22} {'─'*8}   {'─'*30}")
        mx = max(a) + 1e-9
        for tok, w in zip(real, a):
            bar  = "█" * int(w/mx*30)
            star = "  ← máxima atención" if w == max(a) else ""
            print(f"  {tok:<22} {w:>8.4f}   {bar:<30}{star}")
        print()

    def save(self, path="transformer_lm.pkl"):
        data = {"corpus": self.corpus, "tokenizer": self.tokenizer, "hp": self._hp}
        if BACKEND == "pytorch":
            data["state_dict"] = self.model.state_dict()
        else:
            data["weights"] = {p: getattr(self.model, p) for p in self.model._params()}
            data["adam"]    = {"m": self.model._m, "v": self.model._v,
                               "step": self.model._step}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾  Guardado: {path}")

    def load(self, path="transformer_lm.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.corpus    = data["corpus"]
        self.tokenizer = data["tokenizer"]
        self._hp       = data["hp"]
        V = self.tokenizer.vocab_size
        if BACKEND == "pytorch":
            self.model = TransformerLM(
                V=V, d=self._hp["d_model"], heads=self._hp["n_heads"],
                layers=self._hp["n_layers"], dff=self._hp["d_ff"],
                max_len=self._hp["max_len"], drop=self._hp["dropout"],
                pad=self.tokenizer.pad_idx
            ).to(DEVICE)
            self.model.load_state_dict(data["state_dict"])
            self.model.eval()
        else:
            self.model = NumpyTransformer(V, d_model=self._hp["d_model"],
                                          d_ff=self._hp["d_ff"],
                                          ctx=self._hp["max_len"])
            for p, w in data["weights"].items():
                setattr(self.model, p, w)
            self.model._m    = data["adam"]["m"]
            self.model._v    = data["adam"]["v"]
            self.model._step = data["adam"]["step"]
        print(f"📂  Cargado: {path}")


# ══════════════════════════════════════════════════════════════════════
#  INTERFAZ INTERACTIVA
# ══════════════════════════════════════════════════════════════════════
BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║    TRANSFORMER  —  PREDICTOR DE SIGUIENTE PALABRA                   ║
║    Embed + PosEnc + MHA(causal) + FFN + LM-Head                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  train [épocas] [lr]     entrenar    (ej: train 120 0.003)         ║
║  predict <frase>          predecir siguiente palabra                ║
║  attn <frase>             mapa de atención: qué palabras importan  ║
║  add <frase>              añadir frase al corpus                    ║
║  temp <n>  /  topk <n>    temperatura y top-k                      ║
║  vocab  /  corpus         ver vocabulario o corpus                  ║
║  save [f]  /  load [f]    persistencia                              ║
║  quit                     salir                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""




def print_prediction(results, prompt):
    best, bp = results[0]
    print(f"\n  Prompt     : « {prompt} »")
    print(f"  Predicción : « {best} »  ({bp*100:.1f}%)\n")
    print(f"  {'Palabra':<22} {'Prob':>8}   Distribución de probabilidad")
    print(f"  {'─'*22} {'─'*8}   {'─'*32}")
    for i, (w, p) in enumerate(results):
        bar  = "█" * max(1, int(p * 350))
        star = "  ◄ MEJOR" if i == 0 else ""
        print(f"  {w:<22} {p*100:>7.2f}%   {bar:<32}{star}")
    print()

def main():
    print(BANNER)
    engine      = TransformerEngine(d_model=64, n_heads=4, n_layers=2,
                                    d_ff=128, max_len=24, dropout=0.1)
    temperature = 1.0
    top_k       = 8
    trained     = False
    engine.build()

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  ¡Hasta luego!"); break
        if not raw: continue
        parts = raw.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts)>1 else "")

        if cmd == "train":
            try:
                p = arg.split()
                epochs = int(p[0]) if p else 100
                lr     = float(p[1]) if len(p)>1 else 5e-3
            except: epochs, lr = 100, 5e-3
            engine.train(epochs=epochs, lr=lr)
            trained = True

        elif cmd == "predict":
            if not arg: print("⚠️  Uso: predict <frase>"); continue
            if not trained: print("⚠️  Primero: train"); continue
            try: print_prediction(engine.predict(arg, top_k, temperature), arg)
            except Exception as e: print(f"❌  {e}")
            
        elif cmd == "p":
            if not arg: print("⚠️  Uso: predict <frase>"); continue
            if not trained: print("⚠️  Primero: train"); continue
            try:
                context = arg
                print(context, end=" ", flush=True)
                
                for i in range(20):  
                    result = engine.predict(context, top_k, temperature);
                    
                    
                    
                    token = result[0][0]
                    
                    #print(" " + token, end="", flush=True)
                    
                    context += f" {token}"
                    
                    print(context)
                    
                print("\n")
                
                
            except Exception as e: print(f"❌  {e}")
            
        elif cmd == "attn":
            if not arg: print("⚠️  Uso: attn <frase>"); continue
            if not trained: print("⚠️  Primero: train"); continue
            engine.show_attention(arg)

        elif cmd == "add":
            if not arg: print("⚠️  Uso: add <frase>"); continue
            engine.corpus.append(arg.lower())
            engine.build(); trained = False
            print(f"✅  {len(engine.corpus)} frases en corpus. Re-entrena: train")

        elif cmd == "vocab":
            words = [w for w in engine.tokenizer.word2idx if not w.startswith("<")]
            print(f"\n  📖  {len(words)} palabras:\n")
            for i in range(0, len(words), 7):
                print("    " + "  ".join(f"{w:<14}" for w in words[i:i+7]))
            print()

        elif cmd == "corpus":
            print(f"\n  📜  Corpus ({len(engine.corpus)} frases):")
            for i, s in enumerate(engine.corpus, 1): print(f"    {i:>3}. {s}")
            print()

        elif cmd == "save":
            if not trained: print("⚠️  Entrena antes de guardar."); continue
            engine.save(arg or "transformer_lm.pkl")

        elif cmd == "load":
            p = arg or "transformer_lm.pkl"
            if not os.path.exists(p): print(f"❌  No encontrado: {p}"); continue
            engine.load(p); trained = True

        elif cmd == "temp":
            try: temperature = float(arg); print(f"🌡️  Temperatura: {temperature}")
            except: print("⚠️  Uso: temp <valor>")

        elif cmd == "topk":
            try: top_k = int(arg); print(f"🎯  Top-K: {top_k}")
            except: print("⚠️  Uso: topk <n>")

        elif cmd in ("quit","exit","q"):
            print("👋  ¡Hasta luego!"); break

        elif cmd in ("help","?","h"):
            print(BANNER)

        else:
            print(f"❓  Comando desconocido: '{cmd}'  →  escribe help")

if __name__ == "__main__":
    main()