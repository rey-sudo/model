╔══════════════════════════════════════════════════════════════════════════════════╗
║                        FLUJO BAN — IMÁGENES DE IGUAL TAMAÑO                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  ENTRADA — 3 imágenes cuadradas (196×196 px)
  ─────────────────────────────────────────────────────────────────────────────
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│
  │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│
  │░░░░  a  ░░░│      │░a car is░░░│      │░a car is a░│
  │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│
  │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│      │░░░░░░░░░░░░░│
  └─────────────┘      └─────────────┘      └─────────────┘
   ρ₁ = 0.02 (baja)     ρ₂ = 0.08 (media)   ρ₃ = 0.12 (alta)
   d=38416              d=38416              d=38416
   x₁ ∈ {0,1}³⁸⁴¹⁶     x₂ ∈ {0,1}³⁸⁴¹⁶     x₃ ∈ {0,1}³⁸⁴¹⁶


  PREPROCESAMIENTO — _preprocess()
  ─────────────────────────────────────────────────────────────────────────────
       x₁ (1, 38416)         x₂ (1, 38416)         x₃ (1, 38416)
       sparse CSR             sparse CSR             sparse CSR
           │                      │                      │
           └──────────────────────┴──────────────────────┘
                                  │
                                  ▼
  ══════════════════════════════════════════════════════════════════════════════
                              BAN1  —  W_fwd (38416 × 352)
  ══════════════════════════════════════════════════════════════════════════════

   train_from_("img.png", "a")          train_from_("img.png", "a car is")      train_from_("img.png", "a car is a")
           │                                       │                                        │
           ▼                                       ▼                                        ▼
     x₁ @ W_fwd → sign()              x₂ @ W_fwd → sign()              x₃ @ W_fwd → sign()
           │                                       │                                        │
           ▼                                       ▼                                        ▼
   b̂₁⁽¹⁾ ∈ {-1,+1}³⁵²           b̂₁⁽²⁾ ∈ {-1,+1}³⁵²           b̂₁⁽³⁾ ∈ {-1,+1}³⁵²
  [+1,-1,+1,-1,+1,+1,...]       [-1,+1,-1,+1,-1,+1,...]       [-1,-1,+1,+1,-1,-1,...]
           │                                       │                                        │
           └───────────────────── dH(b̂₁⁽¹⁾, b̂₁⁽²⁾) > 0 ────────────────────────────────┘
                                  dH(b̂₁⁽²⁾, b̂₁⁽³⁾) > 0
                                  dH(b̂₁⁽¹⁾, b̂₁⁽³⁾) > 0
                                  ↑ firmas separables gracias al tamaño fijo
                                  │
                                  ▼
  ══════════════════════════════════════════════════════════════════════════════
                              BAN2  —  W_fwd (352 × 352)
  ══════════════════════════════════════════════════════════════════════════════

   train_from_upstream_("img.png", "a",         upstream=[BAN1])
   train_from_upstream_("img.png", "a car is",  upstream=[BAN1])
   train_from_upstream_("img.png", "a car is a",upstream=[BAN1])
           │                                       │                                        │
           ▼                                       ▼                                        ▼
   b̂₁⁽¹⁾ @ W_fwd → sign()       b̂₁⁽²⁾ @ W_fwd → sign()       b̂₁⁽³⁾ @ W_fwd → sign()
           │                                       │                                        │
           ▼                                       ▼                                        ▼
   b̂₂⁽¹⁾ ∈ {-1,+1}³⁵²           b̂₂⁽²⁾ ∈ {-1,+1}³⁵²           b̂₂⁽³⁾ ∈ {-1,+1}³⁵²
           │                                       │                                        │
           └──────────────── dH aumenta respecto a BAN1 ──────────────────────────────────┘
                                  │
                                  ▼
  ══════════════════════════════════════════════════════════════════════════════
                              BAN3  —  W_fwd (352 × 352)
  ══════════════════════════════════════════════════════════════════════════════

   train_from_upstream_("img.png", "a",         upstream=[BAN1, BAN2])
   train_from_upstream_("img.png", "a car is",  upstream=[BAN1, BAN2])
   train_from_upstream_("img.png", "a car is a",upstream=[BAN1, BAN2])
           │                                       │                                        │
           ▼                                       ▼                                        ▼
   b̂₂⁽¹⁾ @ W_fwd → sign()       b̂₂⁽²⁾ @ W_fwd → sign()       b̂₂⁽³⁾ @ W_fwd → sign()
           │                                       │                                        │
           ▼                                       ▼                                        ▼
   b̂₃⁽¹⁾ ∈ {-1,+1}³⁵²           b̂₃⁽²⁾ ∈ {-1,+1}³⁵²           b̂₃⁽³⁾ ∈ {-1,+1}³⁵²
           │                                       │                                        │
           └─────────────── dH máximo — mayor discriminación ──────────────────────────────┘


  CLASIFICACIÓN — classify_chained_("img.png", upstream=[BAN1, BAN2])
  ─────────────────────────────────────────────────────────────────────────────

                    imagen de prueba "a car is a"
                              │
                    x_test @ W_fwd(BAN1) → sign() → b̂₁
                              │
                    b̂₁ @ W_fwd(BAN2) → sign() → b̂₂
                              │
                    b̂₂ @ W_fwd(BAN3) → sign() → b̂₃
                              │
                    cosine_similarity(b̂₃, ℓ_c) para cada label c
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         "a"              "a car is"     "a car is a"
        s = +0.12          s = +0.61      s = +0.97  ← ▶ GANADOR











        imagen "a car is"  (196×196 px)
        │
        ▼  BAN1._forward()
        │
        │  BAN1 vio "a car is" durante entrenamiento ✅
        │  produce firma conocida: b̂₁⁽²⁾
        │
        ▼  BAN2._forward()
        │
        │  BAN2 también vio "a car is" ✅
        │  produce firma conocida: b̂₂⁽²⁾
        │
        ▼  BAN3.classify_()
        │
        │  BAN3 también conoce "a car is" ✅
        │  compara b̂₂⁽²⁾ contra los 3 label_vecs
        │
        ▼
   "a"          s = +0.12
   "a car is a" s = +0.61
   "a car is"   s = +0.97  ← ▶ GANADOR








   imagen nueva de "a car is" con fuente distinta / diferente padding
        │
        ▼  BAN1._forward()
        │
        │  firma b̂₁ ≈ b̂₁⁽²⁾  pero no idéntica
        │  pequeña desviación del vector conocido
        │
        ▼  BAN2._forward()
        │
        │  la desviación se amplifica o amortigua
        │  depende de W_fwd de BAN2
        │
        ▼  BAN3.classify_()
        │
        │  b̂₂ está cerca de b̂₂⁽²⁾ pero no igual
        │
   "a"          s = +0.09
   "a car is a" s = +0.55   ← ⚠️ compite más fuerte
   "a car is"   s = +0.71   ← sigue ganando pero con menos margen

   