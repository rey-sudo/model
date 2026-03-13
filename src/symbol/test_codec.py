import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from src.symbol.codec import decodificar_aztec, word_to_aztec
# Importamos tus funciones del archivo principal (asumiendo que se llama generador.py)

# --- CONFIGURACIÓN ---
RANGO_TEST = 10000
CARPETA_TEST = Path("./runas_test_batch")
ARCHIVO_LOG = Path("errores_validacion.log")

def worker_validar_id(unicode_id: int):
    """
    Función que ejecuta el ciclo completo para un ID: 
    Generar -> Resize -> Decodificar -> Comparar -> Borrar.
    """
    nombre_temp = f"test_node_{unicode_id}"
    ruta_svg = CARPETA_TEST / f"{nombre_temp}.svg"
    ruta_png = CARPETA_TEST / f"{nombre_temp}.png"
    
    try:
        # 1. Generar (SVG y PNG 60x60)
        word_to_aztec(CARPETA_TEST, unicode_id, nombre_temp)
        
        # 2. Decodificar desde el PNG generado
        id_recuperado = decodificar_aztec(ruta_png)
        
        # 3. Validar integridad de los datos
        if id_recuperado is None:
            return unicode_id, "ERROR: No se pudo decodificar el código generado."
        
        if id_recuperado != unicode_id:
            return unicode_id, f"ERROR: Mismatch de datos (Esperado: {unicode_id}, Obtenido: {id_recuperado})"
        
        return None # Éxito total

    except Exception as e:
        return unicode_id, f"EXCEPCIÓN: {str(e)}"
    
    finally:
        # 4. Limpieza absoluta para no agotar el espacio en disco
        if ruta_svg.exists():
            ruta_svg.unlink()
        if ruta_png.exists():
            ruta_png.unlink()

def ejecutar_test_paralelo():
    # Asegurar que la carpeta de test exista
    CARPETA_TEST.mkdir(parents=True, exist_ok=True)
    
    num_procesos = cpu_count()
    print(f"🚀 Iniciando validación de {RANGO_TEST:,} elementos.")
    print(f"💻 Usando {num_procesos} núcleos en paralelo.")
    
    inicio_tiempo = time.time()
    fallos = []
    
    # Pool de procesos para ejecución paralela
    with Pool(processes=num_procesos) as pool:
        # imap_unordered es el más eficiente para grandes flujos de datos
        for i, resultado in enumerate(pool.imap_unordered(worker_validar_id, range(RANGO_TEST), chunksize=500)):
            if resultado:
                fallos.append(resultado)
                id_err, msg_err = resultado
                print(f"\n❌ Fallo en ID {id_err}: {msg_err}")
            
            # Progreso cada 10.000 elementos
            if i % 10000 == 0 and i > 0:
                progreso = (i / RANGO_TEST) * 100
                tiempo_transcurrido = time.time() - inicio_tiempo
                velocidad = i / tiempo_transcurrido
                print(f"⏳ Procesados: {i:,} / {RANGO_TEST:,} ({progreso:.2f}%) | Velocidad: {velocidad:.2f} n/s", end="\r")

    # --- REPORTE FINAL ---
    duracion_total = time.time() - inicio_tiempo
    print(f"\n\n{'='*40}")
    print(f"🏁 TEST FINALIZADO en {duracion_total/60:.2f} minutos.")
    print(f"✅ Éxitos: {RANGO_TEST - len(fallos):,}")
    print(f"❌ Fallos: {len(fallos):,}")
    
    if fallos:
        print(f"📝 Detalles de errores guardados en: {ARCHIVO_LOG}")
        with open(ARCHIVO_LOG, "w") as f:
            for id_f, msg_f in fallos:
                f.write(f"ID {id_f}: {msg_f}\n")
    else:
        print("🎉 ¡Increíble! Todos los códigos son 100%.")
    print(f"{'='*40}")

if __name__ == "__main__":
    # Evitar problemas de recursión en Windows/Debian con multiprocessing
    ejecutar_test_paralelo()