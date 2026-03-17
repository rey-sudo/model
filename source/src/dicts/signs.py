from pathlib import Path
from src.dicts.codec import create_canvas_row
import re

class SignManager:
    def __init__(self, collection_paths: list[Path]):
        self.collection_paths = collection_paths
        
        self.SIGN_COLLECTION_RAW = {}
        
        self.SIGN_COLLECTION = {}
        self._SIGN_REVERSE = {}
        
    def build(self):
        self.load_files()
        self.generate_index_map()
    
    def load_files(self):
        for path in self.collection_paths:    
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    if word:
                        self.SIGN_COLLECTION_RAW[word] = word        
    
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
    
    def clean_paragraph(self, line: str):
        return re.findall(r'\b\d{4}\b|[a-zA-Z]{2,}', line)
    
    def paragraph_to_indices(self, array: list[str]):
        return [self.get_index_from_sign(word.lower()) for word in array]
                
    def paragraph_to_bam_dict(self, paragraph: str):
        cleaned = self.clean_paragraph(paragraph)

        block = self.paragraph_to_indices(cleaned)
        return {i: block[:i+1] for i in range(len(block))}
    
    def decode_labels(self, label_str):
        # 1. Hacemos el split para obtener ['10', '11', '12']
        signs = label_str.split(',')
        
        # 2. Iteramos, convertimos a int y aplicamos get_sign_from_index
        # Asumiendo que get_sign_from_index recibe un entero
        resultado = [self.get_sign_from_index(int(idx)) for idx in signs]
        return " ".join(resultado)
    
    def load_paragraph_file(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                contenido = f.read()
            return contenido
        except FileNotFoundError:
            return f"Error: El archivo en '{path}' no fue encontrado."
        except Exception as e:
            return f"Ocurrió un error inesperado: {e}"  
        
    def paragraph_to_canvas(self, paragraph: str, sign_size_px: int, total_signs: int):
        cleaned = self.clean_paragraph(paragraph)
        block = self.paragraph_to_indices(cleaned)
        
        canvas = create_canvas_row(value=block, sign_size_px=sign_size_px, total_signs=total_signs)
        return canvas
        
