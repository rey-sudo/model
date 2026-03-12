imagen de prueba
       │
       ├──► LINEAS[1].classify_()   worker 1  ┐
       ├──► LINEAS[2].classify_()   worker 2  ├──► max(scores) → ganador
       ├──► LINEAS[3].classify_()   worker 3  │
       ├──► LINEAS[4].classify_()   worker 4  │
       └──► LINEAS[N].classify_()   worker N  ┘

tiempo total = tiempo de 1 sola BAN  ← sin importar N