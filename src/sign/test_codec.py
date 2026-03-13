import time
import numpy as np
from src.sign.codec import index_to_sign

def test_collisions(limit=1_200_000):
    print(f"🕵️ Checking for collisions up to {limit:,}...")
    
    # Usamos un set para almacenar las representaciones binarias ya vistas
    # Un set en Python tiene una búsqueda de O(1), es extremadamente rápido.
    seen_patterns = set()
    start_time = time.time()
    
    for i in range(limit + 1):
        # 1. Obtenemos la imagen
        img = index_to_sign(i)
        
        # 2. Convertimos la matriz de datos en una tupla (que es hashable)
        # Solo nos interesa el core de 5x5
        canvas = np.array(img)
        data_core = tuple(canvas[2:7, 2:7].flatten())
        
        # 3. Verificamos si este patrón ya existe
        if data_core in seen_patterns:
            print(f"❌ COLLISION DETECTED at index {i}!")
            # Buscamos qué valor generó este patrón antes (opcional)
            return False
        
        # 4. Guardamos el patrón
        seen_patterns.add(data_core)
        
        if i % 200_000 == 0 and i > 0:
            print(f"Verified {i:,} unique patterns...")

    end_time = time.time()
    print("-" * 45)
    print(f"✨ PERFECT: No collisions found in {limit:,} nodes.")
    print(f"Total unique patterns stored: {len(seen_patterns):,}")
    print(f"Time: {end_time - start_time:.2f}s")
    return True

if __name__ == "__main__":
    test_collisions(1_200_000)