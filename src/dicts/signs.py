from pathlib import Path
from src.dicts import english

SIGN_COLLECTION_RAW = { 
        **english.words.vocabulary_a,      
        **english.words.other,
}

class SignManager:
    def __init__(self, raw_collection: dict):
        self.SIGN_COLLECTION_RAW = raw_collection
        self.SIGN_COLLECTION = {}
        self._SIGN_REVERSE = {}
        
    def build(self):
        self.generate_index_map()
        
    def generate_index_map(self):
        """
        Convierte RAW en formato "word": index_int
        Ejemplo: "abundant": 0, "accept": 1...
        """
        
        for i, key in enumerate(self.SIGN_COLLECTION_RAW.keys()):
                    self.SIGN_COLLECTION[key] = i
                    self._SIGN_REVERSE[i] = key
                    
        print(f"Diccionario indexado con {len(self.SIGN_COLLECTION)} entradas.")
        
    def get_index_from_sign(self, sign: str) -> int:
        """
        Busca el signo (llave) y retorna su índice entero.
        """
        # Usamos .get() para evitar que el programa se rompa si el signo no existe
        index = self.SIGN_COLLECTION.get(sign)
        if index is None:
            # Podrías retornar -1 o lanzar un error según prefieras
            print(f"Error: El signo '{sign}' no existe en la colección.")
            return None
            
        return index        
        
    def get_sign_from_index(self, index: int) -> str:
        """
        Retorna la palabra (str) a partir de su índice entero (int).
        """
        # Buscamos en el mapa inverso para máxima velocidad
        sign = self._SIGN_REVERSE.get(index)
        
        if sign is None:
            print(f"Error: El índice '{index}' no existe.")
            return None
            
        return sign
    
    def apply_index_to_block(self, array: list[str]):
        return [self.get_index_from_sign(word.lower()) for word in array]
    
    def block_to_bam_dict(self, array: list[int]):
        return {i: array[:i+1] for i in range(len(array))}
    
    def decode_labels(self, label_str):
        # 1. Hacemos el split para obtener ['10', '11', '12']
        signs = label_str.split(',')
        
        # 2. Iteramos, convertimos a int y aplicamos get_sign_from_index
        # Asumiendo que get_sign_from_index recibe un entero
        resultado = [self.get_sign_from_index(int(idx)) for idx in signs]
        
        return " ".join(resultado)
    
    def load_block_file(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                contenido = f.read()
            return contenido
        except FileNotFoundError:
            return f"Error: El archivo en '{path}' no fue encontrado."
        except Exception as e:
            return f"Ocurrió un error inesperado: {e}"    